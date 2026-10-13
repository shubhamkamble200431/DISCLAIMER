"""Step 1.3 of plan.md -- the pre-registered 4-way prompt split.

  Direction       35%  (612)  -- where d_R / d_D / d_D_perp are estimated
  Layer-Probing   15%  (263)  -- where the top-10 layers are selected
  Causal Steering 40%  (700)  -- where steering is run and pp is measured
  Human Validation 10% (175)  -- where C* is checked against people

ONE shared partition for all three generation models (decision D5), so cross-model
claims are possible. Stratified jointly on:

  * the MAJORITY C* label across the three models, with prompts where all three
    disagree (1/1/1) kept as their own TIE stratum rather than silently folded in
  * C*'s OWN train/val/test split, because 1,223 of the 1,750 prompts are prompts the
    frozen classifier was trained on. Making it a stratification variable is what lets
    every headline be reported a second time on the C*-test-clean subset.

Class means are always computed from a model's OWN label inside the Direction split --
the majority label is used for stratification only, never as a target.

The assignment is committed: the output JSON carries the seed, the stratum table and a
sha256 over the sorted (prompt_id, split) pairs. Re-running with the same seed
reproduces the same hash; any later change to the partition is therefore visible.

Usage:
  python -X utf8 scripts/make_study_splits.py
  python -X utf8 scripts/make_study_splits.py --seed 42 --verify
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
OUT_DIR = ROOT / "data" / "study"

SPLITS = {"direction": 0.35, "probing": 0.15, "steering": 0.40, "human": 0.10}
LABELS = ["NORMAL", "DISCLAIMER", "REFUSAL"]


def majority_label(row):
    """Majority across the 3 models; 'TIE' when all three disagree."""
    vc = pd.Series(row).value_counts()
    return vc.index[0] if vc.iloc[0] > 1 else "TIE"


def allocate(n, fracs, rng):
    """Largest-remainder allocation of n items to the split fractions."""
    names = list(fracs)
    exact = np.array([n * fracs[k] for k in names])
    base = np.floor(exact).astype(int)
    rem = n - base.sum()
    if rem:
        # ties in the remainder broken randomly, so no split is systematically favoured
        order = np.lexsort((rng.random(len(names)), -(exact - base)))
        base[order[:rem]] += 1
    return dict(zip(names, base.tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verify", action="store_true",
                    help="recompute and compare against the committed hash, then exit")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUT_DIR / "study_splits.json"

    pred = pd.read_parquet(PRED)
    models = sorted(pred.gen_model.unique())

    wide = pred.pivot(index="prompt_id", columns="gen_model", values="longformer_label")
    cstar = pred.groupby("prompt_id")["split"].first()          # C*'s own train/val/test
    maj = wide.apply(majority_label, axis=1).rename("majority_label")

    tbl = pd.concat([maj, cstar.rename("cstar_split")], axis=1).reset_index()
    tbl["stratum"] = tbl.majority_label + "|" + tbl.cstar_split

    rng = np.random.default_rng(args.seed)
    assign = pd.Series(index=tbl.prompt_id, dtype=object)

    for stratum, grp in tbl.groupby("stratum", sort=True):
        ids = grp.prompt_id.to_numpy()
        ids = rng.permutation(ids)                               # seeded shuffle
        counts = allocate(len(ids), SPLITS, rng)
        i = 0
        for name in SPLITS:                                      # insertion order, stable
            assign.loc[ids[i:i + counts[name]]] = name
            i += counts[name]

    assign = assign.sort_index()
    if assign.isna().any():
        raise RuntimeError(f"{int(assign.isna().sum())} prompts unassigned")

    pairs = "\n".join(f"{p},{s}" for p, s in assign.items())
    digest = hashlib.sha256(pairs.encode()).hexdigest()

    if args.verify:
        old = json.loads(dest.read_text(encoding="utf-8"))
        ok = old["sha256"] == digest
        print(f"committed {old['sha256'][:16]}  recomputed {digest[:16]}  "
              f"{'MATCH' if ok else 'MISMATCH'}")
        sys.exit(0 if ok else 1)

    # ---- report ------------------------------------------------------------ #
    print(f"seed={args.seed}  prompts={len(assign)}  sha256={digest[:16]}...")
    print(f"\nsplit sizes: {assign.value_counts().reindex(SPLITS).to_dict()}")

    tbl["split"] = assign.loc[tbl.prompt_id].values
    print("\nstratification check -- majority label x study split")
    print(pd.crosstab(tbl.majority_label, tbl.split).reindex(columns=list(SPLITS)).to_string())
    print("\nC* split x study split (the circularity control)")
    print(pd.crosstab(tbl.cstar_split, tbl.split).reindex(columns=list(SPLITS)).to_string())

    print("\nPER-MODEL class counts by study split (that model's OWN C* label)")
    per_model = {}
    for m in models:
        lab = wide[m].reindex(assign.index)
        ct = pd.crosstab(assign, lab).reindex(index=list(SPLITS), columns=LABELS, fill_value=0)
        per_model[m] = {s: {c: int(ct.loc[s, c]) for c in LABELS} for s in SPLITS}
        print(f"\n  {m}")
        print("  " + ct.to_string().replace("\n", "\n  "))

    out = {
        "seed": args.seed,
        "sha256": digest,
        "n_prompts": int(len(assign)),
        "fractions": SPLITS,
        "stratified_on": ["majority_label (TIE = all three models disagree)",
                          "cstar_split (C*'s own train/val/test)"],
        "note": "Majority label is for STRATIFICATION ONLY. Class means inside the "
                "direction split use each model's own C* label.",
        "split_sizes": {k: int(v) for k, v in assign.value_counts().reindex(SPLITS).items()},
        "majority_label_counts": {k: int(v) for k, v in maj.value_counts().items()},
        "per_model_class_counts": per_model,
        "splits": {k: sorted(int(p) for p in assign[assign == k].index) for k in SPLITS},
    }
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")

    flat = pd.DataFrame({"prompt_id": assign.index, "study_split": assign.values})
    flat = flat.merge(tbl[["prompt_id", "majority_label", "cstar_split"]], on="prompt_id")
    flat.to_parquet(OUT_DIR / "study_splits.parquet", index=False)
    print(f"\nwrote {dest}")
    print(f"wrote {OUT_DIR / 'study_splits.parquet'}")


if __name__ == "__main__":
    main()
