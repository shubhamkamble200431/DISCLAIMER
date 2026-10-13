"""Build the committed 500-prompt screening set (plan_model_screening.md §3.1).

Two groups, kept separate on purpose:

  random   300 prompts drawn uniformly from the 1,750. This is the ONLY group the
           natural-label-mix estimate (Gate 1) may be computed from. At n=300 a rate
           near 15% carries a 95% CI of about +/-4 pp, which is enough to tell a
           25%-disclaimer model from a 10%-disclaimer one -- the distinction that
           matters -- but not enough to rank two models a few points apart.

  enriched 200 prompts where at least 2 of the 3 current generation models produced
           DISCLAIMER or REFUSAL. These top up the class counts so a candidate's
           per-layer probe and difference-in-means direction are estimable at all.
           They are BIASED toward hedging by construction and must never enter the
           label-mix estimate.

Drawn with a fixed seed and committed with a sha256, the same way study_splits.json is.

Usage:
  python -X utf8 scripts/build_screen_prompts.py
  python -X utf8 scripts/build_screen_prompts.py --n-random 300 --n-enriched 200 --verify
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT.parent / "refusal_classifier" / "results" / "predictions" / "generated_predictions.parquet"
PROMPTS = ROOT / "data" / "generated" / "prompts.parquet"
OUT_DIR = ROOT / "data" / "screen"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-random", type=int, default=300)
    ap.add_argument("--n-enriched", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUT_DIR / "screen_prompts.json"

    pred = pd.read_parquet(PRED)
    prompts = pd.read_parquet(PROMPTS)[["prompt_id", "prompt", "n_chars"]]

    # how many of the 3 current models hedged or refused on each prompt
    hedge = (pred.longformer_label.isin(["DISCLAIMER", "REFUSAL"])
             .groupby(pred.prompt_id).sum().rename("n_hedged"))

    rng = np.random.default_rng(args.seed)
    all_ids = np.sort(prompts.prompt_id.to_numpy())

    rand_ids = np.sort(rng.choice(all_ids, size=args.n_random, replace=False))

    pool = np.sort(hedge[hedge >= 2].index.to_numpy())
    pool = np.setdiff1d(pool, rand_ids)          # no overlap: groups stay disjoint
    if len(pool) < args.n_enriched:
        raise SystemExit(f"only {len(pool)} enriched candidates available after removing "
                         f"the random draw; lower --n-enriched")
    enr_ids = np.sort(rng.choice(pool, size=args.n_enriched, replace=False))

    sel = pd.concat([
        prompts[prompts.prompt_id.isin(rand_ids)].assign(group="random"),
        prompts[prompts.prompt_id.isin(enr_ids)].assign(group="enriched"),
    ]).sort_values(["group", "prompt_id"]).reset_index(drop=True)

    pairs = "\n".join(f"{r.prompt_id},{r.group}" for r in sel.itertuples())
    digest = hashlib.sha256(pairs.encode()).hexdigest()

    if args.verify:
        old = json.loads(dest.read_text(encoding="utf-8"))
        ok = old["sha256"] == digest
        print(f"committed {old['sha256'][:16]}  recomputed {digest[:16]}  "
              f"{'MATCH' if ok else 'MISMATCH'}")
        sys.exit(0 if ok else 1)

    # what the current models do on each group -- the yardstick a candidate is read against
    ref = {}
    for g, ids in (("random", rand_ids), ("enriched", enr_ids)):
        sub = pred[pred.prompt_id.isin(ids)]
        ref[g] = {}
        for m, d in sub.groupby("gen_model"):
            vc = d.longformer_label.value_counts()
            ref[g][m] = {c: {"n": int(vc.get(c, 0)), "pct": round(100 * vc.get(c, 0) / len(d), 1)}
                         for c in ["NORMAL", "DISCLAIMER", "REFUSAL"]}

    out = {
        "seed": args.seed,
        "sha256": digest,
        "n_total": int(len(sel)),
        "groups": {"random": int(len(rand_ids)), "enriched": int(len(enr_ids))},
        "enriched_rule": "at least 2 of the 3 current models produced DISCLAIMER or REFUSAL",
        "warning": "Gate 1 (natural label mix) may ONLY be computed on the 'random' group. "
                   "The enriched group is biased toward hedging by construction.",
        "incumbent_reference": ref,
        "prompt_ids": {"random": [int(x) for x in rand_ids],
                       "enriched": [int(x) for x in enr_ids]},
    }
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    sel.to_parquet(OUT_DIR / "screen_prompts.parquet", index=False)

    print(f"seed={args.seed}  sha256={digest[:16]}...")
    print(f"random={len(rand_ids)}  enriched={len(enr_ids)}  total={len(sel)}  "
          f"(disjoint, enriched pool was {len(pool)})")
    print(f"prompt chars: median {int(sel.n_chars.median())}  p90 {int(sel.n_chars.quantile(.9))}  "
          f"max {int(sel.n_chars.max())}")
    print("\nincumbent label mix on the RANDOM group (the Gate 1 yardstick):")
    for m, v in ref["random"].items():
        print(f"  {m:13s} N {v['NORMAL']['pct']:5.1f}%  D {v['DISCLAIMER']['pct']:5.1f}%  "
              f"R {v['REFUSAL']['pct']:5.1f}%")
    print("\nincumbent label mix on the ENRICHED group (class-supply yardstick):")
    for m, v in ref["enriched"].items():
        print(f"  {m:13s} N {v['NORMAL']['pct']:5.1f}%  D {v['DISCLAIMER']['pct']:5.1f}%  "
              f"R {v['REFUSAL']['pct']:5.1f}%")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
