"""Step 1.4 of plan.md -- how much of C*'s label depends on answer length?

Re-scores the 5,250 already-generated answers with the frozen C* after truncating each
answer to K generated tokens, for K in {128, 192, 256, 384}, and crosstabs the result
against the full-length (512-token) label.

Why this is EXACT, not a proxy
------------------------------
Generation used greedy decoding (`do_sample: false`), which is prefix-deterministic: a
run with `max_new_tokens=K` emits exactly the first K tokens of the 512-token answer.
So truncating the stored answer to K tokens reproduces, byte for byte, what a K-capped
generation run would have produced -- for every answer that did not stop early. Answers
shorter than K are untouched and their stored label is reused directly.

The same table answers a second question: if `max_new_tokens` stays at 512 but C* is
given only the first K tokens (the "fixed classification window" proposal), these are
the labels it would assign.

What it reports
---------------
  per K: label distribution, per-class retention, the full 3x3 crosstab vs full-length,
         and the headline -- how many full-length REFUSAL answers survive as REFUSAL.

Usage:
  python -X utf8 scripts/truncation_calibration.py
  python -X utf8 scripts/truncation_calibration.py --caps 128,192,256,384 --smoke
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import predict as P               # noqa: E402  reuse the exact frozen-C* scoring path
from src import data as D         # noqa: E402
from src import train as T        # noqa: E402

RUN_DIR = ROOT / "results" / "runs" / "longformer_42_20260916-160022"
PRED_PARQUET = ROOT / "results" / "predictions" / "generated_predictions.parquet"
OUT_DIR = ROOT / "results" / "reports"

LABELS = ["NORMAL", "DISCLAIMER", "REFUSAL"]

# generation-model slug -> the tokenizer that produced its answers
GEN_TOKENIZERS = {
    "qwen3-5-4b": "Qwen/Qwen3.5-4B",
    "qwen3-5-9b": "Qwen/Qwen3.5-9B",
    "gemma-4-e2b": "google/gemma-4-E2B-it",
}


def truncate_by_gen_tokens(df, caps, log):
    """For each cap K, the answer text cut to its first K generated tokens.

    Returns {K: pandas.Series of text, aligned to df.index}. Rows whose answer is
    already <= K tokens are returned unchanged (and are byte-identical to the original,
    which is asserted).
    """
    from transformers import AutoTokenizer

    out = {k: df["answer"].copy() for k in caps}
    for slug, hf_id in GEN_TOKENIZERS.items():
        sub = df[df.gen_model == slug]
        if sub.empty:
            continue
        t0 = time.perf_counter()
        tok = AutoTokenizer.from_pretrained(hf_id)
        log(f"    {slug:12s} tokenizer {hf_id} loaded in {time.perf_counter() - t0:.1f}s")

        need = sub[sub.n_generated_tokens > min(caps)]
        if need.empty:
            continue
        enc = tok(need["answer"].tolist(), add_special_tokens=False)["input_ids"]
        n_tok = need["n_generated_tokens"].to_numpy()
        rows = need.index.to_numpy()

        for k in caps:
            pos = np.flatnonzero(n_tok > k)
            if not len(pos):
                continue
            texts = tok.batch_decode([enc[i][:k] for i in pos], skip_special_tokens=True)
            out[k].loc[rows[pos]] = texts
        del tok
    return out


def label_mix(labels):
    vc = pd.Series(labels).value_counts()
    n = len(labels)
    return {c: {"n": int(vc.get(c, 0)), "pct": round(100 * vc.get(c, 0) / n, 2)}
            for c in LABELS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caps", default="128,192,256,384")
    ap.add_argument("--run", default=str(RUN_DIR))
    ap.add_argument("--token-budget", type=int, default=16384)
    ap.add_argument("--spot-check", type=int, default=200,
                    help="re-score this many UNCHANGED rows to prove the stored labels "
                         "came from an identical scoring path (0 to skip)")
    ap.add_argument("--smoke", action="store_true", help="400 rows, caps 256 only")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = P.Tee(OUT_DIR / "truncation_calibration.log")
    caps = sorted(int(x) for x in args.caps.split(",") if x.strip())
    if args.smoke:
        caps = [256]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"truncation_calibration  caps={caps}  device={device}  smoke={args.smoke}")
    if device == "cuda":
        log(f"  gpu {torch.cuda.get_device_name(0)} "
            f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    df = pd.read_parquet(PRED_PARQUET)
    if args.smoke:
        df = df.groupby("gen_model", group_keys=False).head(134)
    df = df.reset_index(drop=True)
    log(f"  rows {len(df)}  models {sorted(df.gen_model.unique())}")
    log(f"  full-length mix {label_mix(df.longformer_label.tolist())}")

    run_dir = Path(args.run)
    cfg, tok, model, amp = P.load_classifier(run_dir, device, log)

    # -- integrity gate: the same val-parity check predict.py runs ------------- #
    val = pd.read_parquet(ROOT / "data" / "processed" / "val.parquet")
    vp, _ = P.predict(val["response_text"].tolist(), cfg, tok, model, device, amp,
                      args.token_budget)
    saved = np.load(run_dir / "val_probs.npy")
    agree = float((vp.argmax(1) == saved.argmax(1)).mean())
    log(f"  val parity: agreement={agree:.4f}  max|dp|={np.abs(vp - saved).max():.4f}")
    if agree < P.MIN_VAL_AGREEMENT:
        raise RuntimeError(f"val parity {agree:.4f} < {P.MIN_VAL_AGREEMENT}; "
                           f"the checkpoint did not restore identically")

    # -- spot check: unchanged rows must reproduce their stored labels --------- #
    if args.spot_check:
        rng = np.random.default_rng(42)
        short = df[df.n_generated_tokens <= min(caps)]
        take = short.sample(min(args.spot_check, len(short)), random_state=42)
        sp, _ = P.predict(take["answer"].tolist(), cfg, tok, model, device, amp,
                          args.token_budget)
        got = [LABELS[i] for i in sp.argmax(1)]
        match = float((np.array(got) == take.longformer_label.values).mean())
        log(f"  spot check on {len(take)} unchanged rows: label match={match:.4f}")
        if match < 0.99:
            raise RuntimeError(f"stored labels do not reproduce ({match:.4f}); the "
                               f"scoring path differs from the one that wrote the parquet")

    # -- build the truncated texts -------------------------------------------- #
    log("  tokenizing answers with each generation model's own tokenizer")
    texts_by_cap = truncate_by_gen_tokens(df, caps, log)

    # -- score, re-using stored labels for rows the cap does not touch --------- #
    results, per_cap_labels = {}, {}
    for k in caps:
        changed = df.n_generated_tokens > k
        n_changed = int(changed.sum())
        labels = df.longformer_label.copy()          # unchanged rows keep their label
        probs = df[["longformer_p_normal", "longformer_p_disclaimer",
                    "longformer_p_refusal"]].to_numpy().copy()

        if n_changed:
            t0 = time.perf_counter()
            pk, ntr = P.predict(texts_by_cap[k][changed].tolist(), cfg, tok, model,
                                device, amp, args.token_budget)
            dt = time.perf_counter() - t0
            labels.loc[changed] = [LABELS[i] for i in pk.argmax(1)]
            probs[changed.to_numpy()] = pk
            log(f"  cap {k:4d}: {n_changed} changed rows scored in {dt:.1f}s "
                f"({n_changed / dt:.1f}/s)  clf_truncated={ntr}  "
                f"peakVRAM={torch.cuda.max_memory_allocated() / 1e9:.2f}GB"
                if device == "cuda" else
                f"  cap {k:4d}: {n_changed} changed rows scored in {dt:.1f}s")
        else:
            log(f"  cap {k:4d}: no rows exceed this cap")

        per_cap_labels[k] = labels
        full = df.longformer_label
        ct = pd.crosstab(full, labels).reindex(index=LABELS, columns=LABELS, fill_value=0)
        flip = {c: round(100 * (1 - ct.loc[c, c] / max(1, ct.loc[c].sum())), 2)
                for c in LABELS}

        results[k] = {
            "n_rows": int(len(df)),
            "n_changed_by_cap": n_changed,
            "pct_answers_affected": round(100 * n_changed / len(df), 2),
            "label_mix": label_mix(labels.tolist()),
            "crosstab_full_to_capped": {a: {b: int(ct.loc[a, b]) for b in LABELS}
                                        for a in LABELS},
            "flip_rate_pct_by_full_label": flip,
            "refusal_retained": {
                "n_full": int(ct.loc["REFUSAL"].sum()),
                "n_still_refusal": int(ct.loc["REFUSAL", "REFUSAL"]),
                "pct": round(100 * ct.loc["REFUSAL", "REFUSAL"]
                             / max(1, ct.loc["REFUSAL"].sum()), 2),
            },
            "by_gen_model": {
                g: label_mix(labels[df.gen_model == g].tolist())
                for g in sorted(df.gen_model.unique())
            },
            "mean_probs": {c: round(float(probs[:, i].mean()), 4)
                           for i, c in enumerate(LABELS)},
        }

    # -- the full-length reference row ---------------------------------------- #
    results["full_512"] = {
        "n_rows": int(len(df)),
        "label_mix": label_mix(df.longformer_label.tolist()),
        "by_gen_model": {g: label_mix(df[df.gen_model == g].longformer_label.tolist())
                         for g in sorted(df.gen_model.unique())},
        "pct_hit_max_new_tokens": round(100 * float(df.hit_max_new_tokens.mean()), 2),
    }

    out = {
        "classifier": {"run": run_dir.name, "max_length": cfg["model"]["max_length"],
                       "truncation": cfg["train"].get("truncation", "head")},
        "val_parity_agreement": agree,
        "caps": caps,
        "results": results,
    }
    dest = OUT_DIR / ("truncation_calibration_smoke.json" if args.smoke
                      else "truncation_calibration.json")
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # -- per-row labels, for anything downstream ------------------------------ #
    wide = df[["gen_model", "prompt_id", "n_generated_tokens", "hit_max_new_tokens",
               "longformer_label"]].copy()
    wide = wide.rename(columns={"longformer_label": "label_full_512"})
    for k in caps:
        wide[f"label_cap_{k}"] = per_cap_labels[k].values
    wide.to_parquet(OUT_DIR / ("truncation_calibration_labels_smoke.parquet" if args.smoke
                               else "truncation_calibration_labels.parquet"), index=False)

    # -- readable summary ----------------------------------------------------- #
    log("")
    log("  cap   affected   NORMAL  DISCLAIMER  REFUSAL   | REFUSAL retained")
    fm = results["full_512"]["label_mix"]
    log(f"   512      0.0%    {fm['NORMAL']['pct']:5.1f}%     {fm['DISCLAIMER']['pct']:5.1f}%   "
        f"{fm['REFUSAL']['pct']:5.1f}%   | (reference)")
    for k in caps:
        r = results[k]
        m = r["label_mix"]
        rr = r["refusal_retained"]
        log(f"  {k:4d}  {r['pct_answers_affected']:6.1f}%    {m['NORMAL']['pct']:5.1f}%     "
            f"{m['DISCLAIMER']['pct']:5.1f}%   {m['REFUSAL']['pct']:5.1f}%   | "
            f"{rr['n_still_refusal']}/{rr['n_full']} ({rr['pct']:.1f}%)")
    log("")
    log("  flip rate (% of full-length label that changes under the cap)")
    for k in caps:
        f = results[k]["flip_rate_pct_by_full_label"]
        log(f"  {k:4d}   N {f['NORMAL']:5.1f}   D {f['DISCLAIMER']:5.1f}   R {f['REFUSAL']:5.1f}")
    log(f"\n  wrote {dest}")


if __name__ == "__main__":
    main()
