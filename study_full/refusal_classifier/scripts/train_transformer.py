"""Train one transformer candidate. GPU jobs must be run SEQUENTIALLY, never in parallel.

Modes:
  --split-mode holdout   train on data/processed/train.parquet, early-stop on val.parquet
                         (used for hyperparameter freezing; test never touched)
  --split-mode cv        grouped 5-fold CV over data/processed/cv_pool.parquet, writing
                         out-of-fold predictions

Usage (PowerShell):
  python -X utf8 scripts/train_transformer.py --config configs/deberta_v3.yaml --smoke
  python -X utf8 scripts/train_transformer.py --config configs/deberta_v3.yaml --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data as D       # noqa: E402
from src import metrics as M    # noqa: E402
from src import train as T      # noqa: E402


class Tee:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()


def subsample_smoke(df, n=200):
    per = max(1, n // 3)
    return (df.groupby("label_id", group_keys=False, sort=True)
            .head(per).reset_index(drop=True))


def run_holdout(cfg, seed, smoke, run_dir, log):
    proc = D.repo_path("data/processed")
    tr = pd.read_parquet(proc / "train.parquet")
    va = pd.read_parquet(proc / "val.parquet")
    if smoke:
        tr, va = subsample_smoke(tr), subsample_smoke(va)
    log(f"  train rows={len(tr)}  val rows={len(va)}  (test NOT loaded)")
    data = {"train": (tr["response_text"].tolist(), tr["label_id"].to_numpy()),
            "val": (va["response_text"].tolist(), va["label_id"].to_numpy())}

    if cfg["kind"] == "frozen_embed":
        t0 = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        bs = int(cfg["model"].get("embed_batch_size", 16))
        e_tr, n_enc = T.embed_frozen(data["train"][0], cfg, batch_size=bs, smoke=smoke)
        e_va, _ = T.embed_frozen(data["val"][0], cfg, batch_size=bs, smoke=smoke)
        embed_s = time.perf_counter() - t0
        peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        log(f"  embeddings: train{e_tr.shape} val{e_va.shape} in {embed_s:.1f}s "
            f"peakVRAM={peak / 1e9:.2f}GB  (encoder params {n_enc:,}, frozen)")
        clf, mt, info = T.train_frozen_head(cfg, e_tr, data["train"][1],
                                            e_va, data["val"][1], seed, log)
        geom = M.label_geometry(e_va, data["val"][1], pooling="masked_mean_l2")
        log("    geometry: " + "  ".join(
            f"{k} d={v['euclidean']:.2f} cos={v['cosine']:.3f}"
            for k, v in geom["pairs"].items() if "euclidean" in v))
        np.save(run_dir / "val_emb.npy", e_va)
        return {"val_metrics": mt, "geometry": geom,
                "wall_seconds": embed_s + info["fit_seconds"],
                "peak_vram_bytes": peak, "n_trainable_params": info["n_head_params"],
                "n_frozen_encoder_params": n_enc,
                "throughput_per_sec": float(len(data["val"][0]) / max(embed_s, 1e-9)),
                "best_config": {"head": cfg["head"]},
                "_preds": clf.predict(e_va).tolist(),
                "_probs": clf.predict_proba(e_va).tolist()}

    best = T.train_encoder(cfg, data, seed, run_dir, smoke=smoke, log=log)
    out = {k: v for k, v in best.items() if not k.startswith("_")}
    out["_preds"] = best["_preds"]
    out["_probs"] = best["_probs"]
    out["best_config"] = {"lr": best["lr"], "best_epoch": best["best_epoch"],
                          "max_length": 64 if smoke else cfg["model"]["max_length"],
                          "truncation": cfg["train"].get("truncation", "head"),
                          "class_weight": cfg["train"].get("class_weight", "none")}
    if not smoke:
        torch.save(best["_state"], run_dir / "best_state.pt")
        best["_tok"].save_pretrained(run_dir / "tokenizer")
    return out


def load_all_labelled(log):
    """train + val + test concatenated. Used only by --split-mode all."""
    proc = D.repo_path("data/processed")
    parts = []
    for split in ("train", "val", "test"):
        df = pd.read_parquet(proc / f"{split}.parquet")
        log(f"  {split:5} rows={len(df)}")
        parts.append(df)
    allx = pd.concat(parts, ignore_index=True)
    log(f"  TOTAL rows={len(allx)}  (test IS included: this is a fit-on-everything run)")
    counts = allx.groupby("label_3").size().to_dict()
    log(f"  class counts: {counts}")
    return allx


def resolve_hparams(args, cfg, log):
    """lr/epochs for --split-mode all: explicit flags, or read from a prior run."""
    lr, epochs, src = args.lr, args.epochs, "command line"
    if args.from_run:
        rd = D.repo_path(args.from_run)
        bc = json.loads((rd / "metrics.json").read_text(encoding="utf-8"))["best_config"]
        lr = lr if lr is not None else float(bc["lr"])
        epochs = epochs if epochs is not None else int(bc["best_epoch"])
        src = f"{args.from_run} (lr={bc['lr']}, best_epoch={bc['best_epoch']})"
    if lr is None or epochs is None:
        raise SystemExit(
            "--split-mode all needs frozen hyperparameters: pass --lr and --epochs, "
            "or --from-run <holdout run dir> to read them from its metrics.json. "
            "There is no validation split in this mode, so nothing can be tuned here.")
    log(f"  hyperparameters from {src}: lr={lr:g} epochs={epochs}")
    return float(lr), int(epochs)


def run_all(cfg, args, run_dir, log):
    df = load_all_labelled(log)
    texts = df["response_text"].tolist()
    labels = df["label_id"].to_numpy()
    if args.smoke:
        df = subsample_smoke(df)
        texts, labels = df["response_text"].tolist(), df["label_id"].to_numpy()
        log(f"  SMOKE subsample rows={len(texts)}")

    if cfg["kind"] == "frozen_embed":
        import pickle
        bs = int(cfg["model"].get("embed_batch_size", 16))
        t0 = time.perf_counter()
        emb, n_enc = T.embed_frozen(texts, cfg, batch_size=bs, smoke=args.smoke)
        embed_s = time.perf_counter() - t0
        clf, mt, info = T.train_frozen_head(cfg, emb, labels, emb, labels, args.seed, log)
        geom = M.label_geometry(emb, labels, pooling="masked_mean_l2")
        log("    geometry: " + "  ".join(
            f"{k} d={v['euclidean']:.2f} cos={v['cosine']:.3f}"
            for k, v in geom["pairs"].items() if "euclidean" in v))
        with open(run_dir / "final_head.pkl", "wb") as f:
            pickle.dump(clf, f)
        (run_dir / "label_map.json").write_text(json.dumps(
            {"id2label": {i: l for i, l in enumerate(M.LABELS3)},
             "label2id": {l: i for i, l in enumerate(M.LABELS3)},
             "max_length": 64 if args.smoke else cfg["model"]["max_length"],
             "prefix": cfg["model"].get("prefix", ""),
             "model_name": cfg["model"]["name"], "head": "final_head.pkl"},
            indent=2), encoding="utf-8")
        return {"fit_metrics": mt, "geometry": geom,
                "wall_seconds": embed_s + info["fit_seconds"],
                "n_trainable_params": info["n_head_params"],
                "n_frozen_encoder_params": n_enc, "n_train_rows": len(texts),
                "lr": None, "epochs": None}

    lr, epochs = resolve_hparams(args, cfg, log)
    if args.smoke:
        epochs = 1
    return T.train_final(cfg, texts, labels, args.seed, run_dir, lr, epochs,
                         smoke=args.smoke, log=log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true",
                    help="200 rows/split, 1 epoch, max_length 64, first LR only")
    ap.add_argument("--split-mode", default="holdout", choices=["holdout", "cv", "all"])
    ap.add_argument("--lr", type=float, default=None,
                    help="--split-mode all only: fixed learning rate (no sweep)")
    ap.add_argument("--epochs", type=int, default=None,
                    help="--split-mode all only: fixed epoch count (no early stopping)")
    ap.add_argument("--from-run", default=None,
                    help="--split-mode all only: holdout run dir to read lr/best_epoch from")
    ap.add_argument("--runs-dir", default="results/runs")
    args = ap.parse_args()

    cfg = D.load_config(D.repo_path(args.config))
    name = cfg["candidate"]
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    prefix = "smoke_" if args.smoke else ""
    run_dir = D.repo_path(args.runs_dir) / f"{prefix}{name}_{args.seed}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log = Tee(run_dir / "train.log")

    log("=" * 78)
    log(f"CANDIDATE {name}  kind={cfg['kind']}  seed={args.seed}  "
        f"smoke={args.smoke}  split_mode={args.split_mode}")
    T.print_env()
    env = T.env_snapshot()
    log(json.dumps(env))
    log(f"  model      : {cfg['model']['name']}")
    log(f"  max_length : {64 if args.smoke else cfg['model']['max_length']}"
        f"{'  (smoke override)' if args.smoke else ''}")
    log(f"  run_dir    : {run_dir}")

    (run_dir / "config_snapshot.json").write_text(
        json.dumps({"config_path": args.config, "config": cfg, "seed": args.seed,
                    "smoke": args.smoke, "split_mode": args.split_mode,
                    "env": env}, indent=2), encoding="utf-8")

    t0 = time.perf_counter()
    if args.split_mode == "holdout":
        res = run_holdout(cfg, args.seed, args.smoke, run_dir, log)
    elif args.split_mode == "all":
        res = run_all(cfg, args, run_dir, log)
    else:
        raise NotImplementedError("cv mode is wired up after hyperparameters are frozen")
    total = time.perf_counter() - t0

    res["candidate"] = name
    res["seed"] = args.seed
    res["smoke"] = args.smoke
    res["total_wall_seconds"] = total
    res["env"] = env
    preds = res.pop("_preds", None)
    probs = res.pop("_probs", None)
    if preds is not None:
        np.save(run_dir / "val_preds.npy", np.array(preds))
        np.save(run_dir / "val_probs.npy", np.array(probs))
    M.write_json(res, run_dir / "metrics.json")

    log("-" * 78)
    if args.split_mode == "all":
        v = res["fit_metrics"]
        log(f"RESULT {name} seed={args.seed}: FIT-ON-TRAINING-DATA "
            f"macroF1={v['macro_f1']:.4f} DISC_F1={v['disclaimer_f1']:.4f} "
            f"acc={v['accuracy']:.4f}")
        log("  NOTE: these are in-sample numbers on data the model was fitted to.")
        log("        They measure fit, not generalisation. Do not quote them.")
        log(f"  rows fitted      : {res.get('n_train_rows', 0):,}")
    else:
        v = res["val_metrics"]
        log(f"RESULT {name} seed={args.seed}: val macroF1={v['macro_f1']:.4f} "
            f"DISC_F1={v['disclaimer_f1']:.4f} acc={v['accuracy']:.4f}")
    log(f"  trainable params : {res.get('n_trainable_params', 0):,}")
    log(f"  wall seconds     : {total:.1f}")
    log(f"  peak VRAM        : {res.get('peak_vram_bytes', 0) / 1e9:.2f} GB")
    log(f"  throughput       : {res.get('throughput_per_sec', float('nan')):.1f} resp/s")
    g = res.get("geometry")
    if g:
        log(f"  label geometry ({g['pooling']}, dim={g['dim']}):")
        for k, v in g["pairs"].items():
            if "euclidean" not in v:
                log(f"    {k:26} insufficient_evidence (n={v['n_a']}/{v['n_b']})")
                continue
            log(f"    {k:26} dist={v['euclidean']:8.3f}  cos={v['cosine']:+.4f}  "
                f"sep={v['separation']:.2f}"
                + ("" if v["evidence"] == "ok" else "  [insufficient_evidence]"))
    log(f"  artefacts        : {run_dir}")


if __name__ == "__main__":
    main()
