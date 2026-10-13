"""Classify responses with trained holdout checkpoints. Inference only; GPU jobs run
SEQUENTIALLY (one classifier loaded at a time).

Every --run is used independently -- there is no ensembling. For each run:
  1. re-predict val and compare against the run's saved val_probs.npy. This proves the
     checkpoint, tokenizer and truncation were restored exactly; a mismatch aborts.
  2. score the test split (never touched by training or model selection)
  3. classify the generated answers (<generated>/<gen_model>/answers.jsonl)

Outputs (under --out):
  <candidate>/val_parity.json, <candidate>/test_metrics.json
  test_predictions.parquet          test split, one column set per classifier
  generated_predictions.parquet     generated answers, one column set per classifier
  generated_predictions.csv         same, utf-8-sig so Excel opens it cleanly
  disagreements.csv                 generated answers where the classifiers disagree
  summary.json

Usage (PowerShell):
  python -X utf8 scripts/predict.py --smoke
  python -X utf8 scripts/predict.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data as D       # noqa: E402
from src import metrics as M    # noqa: E402
from src import train as T      # noqa: E402

DEFAULT_RUNS = ["results/runs/modernbert_large_42_20260916-145535",
                "results/runs/longformer_42_20260916-160022"]
DEFAULT_GENERATED = "../refusal_generation/data/generated"
SMOKE_N = 48
MIN_VAL_AGREEMENT = 0.99


class Tee:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def load_classifier(run_dir, device, log):
    cfg = json.loads((run_dir / "config_snapshot.json").read_text(encoding="utf-8"))["config"]
    m, t = cfg["model"], cfg["train"]
    tok = T.load_tokenizer(cfg, run_dir / "tokenizer")
    T.check_special_tokens(tok, log)

    # Same config construction as train_encoder(); weights come from best_state.pt,
    # so only the base config is needed, not the base checkpoint. LoRA runs saved
    # merged weights, so they load here like any full fine-tune.
    conf = T.load_model_config(cfg, tok)
    model = T.load_classifier_model(cfg, conf, pretrained=False)
    state = torch.load(run_dir / "best_state.pt", map_location="cpu")
    model.load_state_dict(state, strict=True)
    del state
    model.to(device).eval()

    amp_dtype = None
    if device == "cuda" and t.get("bf16", True) and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    log(f"    loaded {cfg['candidate']} ({m['name']}) max_length={m['max_length']} "
        f"truncation={t.get('truncation', 'head')} amp={amp_dtype}")
    return cfg, tok, model, amp_dtype


def predict(texts, cfg, tok, model, device, amp_dtype, token_budget):
    """Softmax probs in input order. Batches are length-sorted and capped at
    token_budget padded tokens so long responses do not OOM an 8 GB card."""
    m, t = cfg["model"], cfg["train"]
    max_length = int(m["max_length"])
    ids = T.encode(texts, tok, max_length, t.get("truncation", "head"), m.get("prefix", ""))

    # Longformer pads every batch up to a multiple of its attention window inside
    # forward(), so budget against the padded width -- otherwise a batch of ~520-token
    # rows is really twice the size the budget allowed, and OOMs.
    window = getattr(model.config, "attention_window", None)
    pad_to = max(window) if isinstance(window, (list, tuple)) else (window or 1)

    order = np.argsort([-len(x) for x in ids], kind="stable")
    batches, cur = [], []
    for i in order:
        # descending lengths: the first item of a batch is its longest
        width = -(-len(ids[cur[0] if cur else i]) // pad_to) * pad_to
        if cur and width * (len(cur) + 1) > token_budget:
            batches.append(cur)
            cur = []
        cur.append(int(i))
    if cur:
        batches.append(cur)

    loader = torch.utils.data.DataLoader(
        T.IdsDataset(ids, np.zeros(len(ids))), batch_sampler=batches,
        collate_fn=lambda b: T.collate(b, tok.pad_token_id))
    _, _, probs_sorted = T.evaluate(model, loader, device, amp_dtype)

    probs = np.empty_like(probs_sorted)
    probs[np.concatenate(batches)] = probs_sorted
    n_truncated = int(sum(len(x) >= max_length for x in ids))
    return probs, n_truncated


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def load_generated(root, n=None):
    frames = []
    for d in sorted(p for p in root.iterdir() if (p / "answers.jsonl").exists()):
        df = pd.read_json(d / "answers.jsonl", lines=True, dtype={"prompt_id": "int64"})
        if n:
            df = df.head(n)
        df.insert(0, "gen_model", d.name)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no <gen_model>/answers.jsonl under {root}")
    df = pd.concat(frames, ignore_index=True).rename(columns={"splits": "split"})
    df["answer"] = df["answer"].fillna("").astype(str)
    df["empty_answer"] = df["answer"].str.strip() == ""
    return df


def prob_columns(prefix, probs):
    cols = {f"{prefix}_label": [M.LABELS3[i] for i in probs.argmax(1)]}
    for i, lab in enumerate(M.LABELS3):
        cols[f"{prefix}_p_{lab.lower()}"] = probs[:, i]
    cols[f"{prefix}_confidence"] = probs.max(1)
    return pd.DataFrame(cols)


def pct_table(df, by, col):
    counts = pd.crosstab(df[by], df[col]).reindex(columns=M.LABELS3, fill_value=0)
    return {str(k): {lab: {"n": int(row[lab]), "pct": round(100 * row[lab] / row.sum(), 1)}
                     for lab in M.LABELS3} for k, row in counts.iterrows()}


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=None,
                    help="holdout run dir (repeatable); default: modernbert_large + longformer")
    ap.add_argument("--generated", default=DEFAULT_GENERATED)
    ap.add_argument("--out", default=None)
    ap.add_argument("--token-budget", type=int, default=16384,
                    help="max padded tokens per batch")
    ap.add_argument("--smoke", action="store_true",
                    help=f"first {SMOKE_N} rows of each input, writes to results/predictions_smoke")
    args = ap.parse_args()

    runs = [D.repo_path(r) for r in (args.run or DEFAULT_RUNS)]
    out = D.repo_path(args.out or ("results/predictions_smoke" if args.smoke
                                   else "results/predictions"))
    out.mkdir(parents=True, exist_ok=True)
    log = Tee(out / "predict.log")
    n = SMOKE_N if args.smoke else None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    T.print_env()
    log(f"predict  runs={[r.name for r in runs]}  smoke={args.smoke}  "
        f"token_budget={args.token_budget}  out={out}")

    proc = D.repo_path("data/processed")
    val = pd.read_parquet(proc / "val.parquet")
    test = pd.read_parquet(proc / "test.parquet")
    gen = load_generated(D.repo_path(args.generated), n)
    if n:
        val, test = val.head(n), test.head(n)
    log(f"  val={len(val)}  test={len(test)}  generated={len(gen)} "
        f"({gen['gen_model'].value_counts().to_dict()})  empty answers={int(gen['empty_answer'].sum())}")

    test_out = test[["prompt_id", "side", "model", "label_3"]].reset_index(drop=True)
    gen_out = gen[["gen_model", "prompt_id", "split", "prompt", "answer", "n_generated_tokens",
                   "hit_max_new_tokens", "empty_answer"]].reset_index(drop=True)
    summary = {"runs": {}, "smoke": args.smoke}
    names = []

    for run_dir in runs:
        cfg, tok, model, amp = load_classifier(run_dir, device, log)
        name = cfg["candidate"]
        names.append(name)
        (out / name).mkdir(exist_ok=True)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        rec = {"run_dir": run_dir.name, "model": cfg["model"]["name"]}

        # 1. val parity against the probabilities saved at training time
        t0 = time.perf_counter()
        pv, _ = predict(val["response_text"].tolist(), cfg, tok, model, device, amp,
                        args.token_budget)
        saved = np.load(run_dir / "val_probs.npy")[:len(val)]
        agree = float((pv.argmax(1) == saved.argmax(1)).mean())
        parity = {"n": len(val), "pred_agreement": agree,
                  "max_abs_prob_diff": float(np.abs(pv - saved).max()),
                  "macro_f1_repro": M.classification_metrics(
                      val["label_id"], pv.argmax(1), pv)["macro_f1"]}
        M.write_json(parity, out / name / "val_parity.json")
        log(f"    val parity: agreement={agree:.4f} max|dp|={parity['max_abs_prob_diff']:.4f} "
            f"macroF1={parity['macro_f1_repro']:.4f}")
        if agree < MIN_VAL_AGREEMENT:
            raise RuntimeError(
                f"{name}: re-predicted val agrees with saved val_probs.npy on only "
                f"{agree:.2%} of rows -- checkpoint or tokenisation not restored correctly.")
        rec["val_parity"] = parity

        # 2. test
        pt, _ = predict(test["response_text"].tolist(), cfg, tok, model, device, amp,
                        args.token_budget)
        tm = M.classification_metrics(test["label_id"], pt.argmax(1), pt)
        M.write_json(tm, out / name / "test_metrics.json")
        test_out = pd.concat([test_out, prob_columns(name, pt)], axis=1)
        log(f"    test: macroF1={tm['macro_f1']:.4f} DISC_F1={tm['disclaimer_f1']:.4f} "
            f"acc={tm['accuracy']:.4f} log_loss={tm['log_loss']:.4f}")
        rec["test"] = {k: tm[k] for k in ("n", "accuracy", "macro_f1", "disclaimer_f1",
                                          "log_loss")}

        # 3. generated answers
        t1 = time.perf_counter()
        pg, n_trunc = predict(gen["answer"].tolist(), cfg, tok, model, device, amp,
                              args.token_budget)
        gen_s = time.perf_counter() - t1
        gen_out = pd.concat([gen_out, prob_columns(name, pg)], axis=1)
        rec.update({"generated_seconds": gen_s,
                    "generated_per_sec": len(gen) / gen_s,
                    "generated_truncated_at_max_length": n_trunc,
                    "total_seconds": time.perf_counter() - t0,
                    "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9
                    if device == "cuda" else 0.0})
        log(f"    generated: {len(gen)} in {gen_s:.1f}s ({rec['generated_per_sec']:.1f}/s) "
            f"truncated_at_max_length={n_trunc} peakVRAM={rec['peak_vram_gb']:.2f}GB")
        summary["runs"][name] = rec

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # ----------------------------------------------------------------------- #
    # write + summarise (per classifier; agreement is reported, never combined)
    # ----------------------------------------------------------------------- #
    # With two classifiers this is plain pairwise agreement; with more it is "all agree".
    a, b = names[0], names[-1]
    for df in (test_out, gen_out):
        labels = df[[f"{c}_label" for c in names]]
        df["classifiers_agree"] = labels.nunique(axis=1) == 1

    test_out.to_parquet(out / "test_predictions.parquet", index=False)
    gen_out.to_parquet(out / "generated_predictions.parquet", index=False)
    gen_out.to_csv(out / "generated_predictions.csv", index=False, encoding="utf-8-sig")
    gen_out[~gen_out["classifiers_agree"]].to_csv(
        out / "disagreements.csv", index=False, encoding="utf-8-sig")

    summary["test_classifier_agreement"] = float(test_out["classifiers_agree"].mean())
    summary["pairwise_agreement"] = {
        split: {f"{x}|{y}": round(float((df[f"{x}_label"] == df[f"{y}_label"]).mean()), 4)
                for i, x in enumerate(names) for y in names[i + 1:]}
        for split, df in (("test", test_out), ("generated", gen_out))}
    summary["generated"] = {
        "n": len(gen_out),
        "agreement_by_gen_model": gen_out.groupby("gen_model")["classifiers_agree"]
                                         .mean().round(4).to_dict(),
        "n_disagree_by_gen_model": (~gen_out["classifiers_agree"]).groupby(
            gen_out["gen_model"]).sum().astype(int).to_dict(),
        "labels_by_gen_model": {c: pct_table(gen_out, "gen_model", f"{c}_label")
                                for c in names},
        "labels_by_hit_max_new_tokens": {c: pct_table(gen_out, "hit_max_new_tokens",
                                                      f"{c}_label") for c in names},
        "crosstab_by_gen_model": {
            g: pd.crosstab(sub[f"{a}_label"], sub[f"{b}_label"])
                 .reindex(index=M.LABELS3, columns=M.LABELS3, fill_value=0).to_dict()
            for g, sub in gen_out.groupby("gen_model")} if len(names) == 2 else None,
        "empty_answers": gen_out.loc[gen_out["empty_answer"],
                                     ["gen_model", "prompt_id"] +
                                     [f"{c}_label" for c in names]].to_dict("records"),
    }
    M.write_json(summary, out / "summary.json")

    for c in names:
        log(f"\n  {c} -- label % by generation model")
        log(pd.crosstab(gen_out["gen_model"], gen_out[f"{c}_label"], normalize="index")
            .reindex(columns=M.LABELS3, fill_value=0).mul(100).round(1).to_string())
    log(f"\n  classifier agreement: test={summary['test_classifier_agreement']:.3f}  "
        f"generated={summary['generated']['agreement_by_gen_model']}")
    log(f"  wrote {out}")


if __name__ == "__main__":
    main()
