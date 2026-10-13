"""One table for the two chosen classifiers on the three generation models.

Two ways to feed it:

  --pred <dir>                one predict.py run that held BOTH classifiers
  --pred <a> --pred <b>       two separate predict.py runs, merged on (gen_model,
                              prompt_id). Same labels, so the same table -- use this
                              when one classifier's checkpoint is no longer on the box
                              but its predictions are.

Use --only to pick classifiers out of a dir that holds more than the two you want.
The two stay independent -- this reports how often they agree, it never combines them.

Outputs (under --out):
  pair_table.md     the tables below
  pair_table.csv    the label-mix table, machine-readable
  pair.json         every number on this page

Usage:
  python -X utf8 scripts/pair_report.py --pred results/predictions_pair
  python -X utf8 scripts/pair_report.py --pred results/predictions \
      --pred results/predictions_wide/bge_m3 --only longformer bge_m3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data as D       # noqa: E402
from src import metrics as M    # noqa: E402

N_BOOT = 2000
SHORT = {"NORMAL": "N", "DISCLAIMER": "D", "REFUSAL": "R"}
GEN_KEYS = ["gen_model", "prompt_id"]
TEST_KEYS = ["prompt_id", "side"]
GEN_META = ["split", "prompt", "answer", "n_generated_tokens", "hit_max_new_tokens",
            "empty_answer"]


def macro_f1(y, p, k=3):
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm)
    denom = cm.sum(0) + cm.sum(1)
    return float(np.mean(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)))


def boot_ci(y, p, seed=0):
    """95% bootstrap interval. The test split is 528 rows, so small gaps are noise."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(N_BOOT, len(y)))
    s = np.array([macro_f1(y[i], p[i]) for i in idx])
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def md_table(df, floatfmt=".3f"):
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = [format(v, floatfmt) if isinstance(v, float) and not np.isnan(v)
                 else ("" if isinstance(v, float) else str(v)) for v in r]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def load_predictions(preds, only):
    """Collect one column set per classifier from one or more predict.py output dirs.

    Merging separate dirs is safe because every run scored the same rows: the merge
    keys are the row identity, and a missing label after the merge aborts rather than
    quietly dropping a classifier's column.
    """
    runs, gen, test = {}, None, None
    for d in preds:
        spath = d / "summary.json"
        if not spath.exists():
            sys.exit(f"{d} has no summary.json -- not a predict.py output dir")
        summary = json.loads(spath.read_text(encoding="utf-8"))
        if summary.get("smoke"):
            sys.exit(f"{d} is a smoke run -- rerun predict.py without --smoke")
        g = pd.read_parquet(d / "generated_predictions.parquet")
        t = pd.read_parquet(d / "test_predictions.parquet")

        for name, rec in summary["runs"].items():
            if only and name not in only:
                continue
            if name in runs:
                continue                       # first dir listed wins
            runs[name] = rec
            gcols = [c for c in g.columns if c.startswith(f"{name}_")]
            tcols = [c for c in t.columns if c.startswith(f"{name}_")]
            if gen is None:
                gen = g[GEN_KEYS + GEN_META + gcols].copy()
                test = t[TEST_KEYS + ["label_3"] + tcols].copy()
            else:
                gen = gen.merge(g[GEN_KEYS + gcols], on=GEN_KEYS, how="left")
                test = test.merge(t[TEST_KEYS + tcols], on=TEST_KEYS, how="left")

    if gen is None:
        sys.exit(f"no classifiers found in {[str(p) for p in preds]}"
                 + (f" matching --only {only}" if only else ""))
    names = [c for c in (only or runs) if c in runs]
    for c in names:
        if gen[f"{c}_label"].isna().any() or test[f"{c}_label"].isna().any():
            sys.exit(f"{c}: rows did not line up across the prediction dirs -- these "
                     "runs scored different inputs, so they cannot be compared row by row")
    label_cols = [f"{c}_label" for c in names]
    gen["classifiers_agree"] = gen[label_cols].nunique(axis=1) == 1
    test["classifiers_agree"] = test[label_cols].nunique(axis=1) == 1
    return runs, names, gen, test


def label_mix(gen, names):
    """Rows = generation model, columns = <classifier> <label> as % of that model's rows.
    Both classifiers side by side so the difference reads across a single row."""
    blocks = []
    for c in names:
        pct = (pd.crosstab(gen["gen_model"], gen[f"{c}_label"], normalize="index")
                 .reindex(columns=M.LABELS3, fill_value=0).mul(100))
        pct.columns = [f"{c} %{SHORT[lab]}" for lab in M.LABELS3]
        blocks.append(pct)
    return pd.concat(blocks, axis=1).reset_index(names="gen model")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", action="append", default=None,
                    help="predict.py out dir (repeatable); default results/predictions_pair")
    ap.add_argument("--only", nargs="*", default=None,
                    help="classifier names to keep, in table order")
    ap.add_argument("--runs-dir", default="results/runs")
    ap.add_argument("--out", default="results/pair")
    ap.add_argument("--split", default="all", choices=["all", "train", "val", "test"],
                    help="keep only generated answers whose PROMPT came from this split. "
                         "The prompts were reused for generation, so for 70%% of them the "
                         "classifier trained on some other model's answer to the same "
                         "prompt. --split test is the clean subset (264 prompts/model); "
                         "it does not change the held-out scores in the Classifiers table.")
    args = ap.parse_args()

    preds = [D.repo_path(p) for p in (args.pred or ["results/predictions_pair"])]
    out = D.repo_path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    runs, names, gen, test = load_predictions(preds, args.only)
    if len(names) < 2:
        sys.exit(f"only {names} found -- this report compares two classifiers. Add a "
                 "second --pred dir, or widen --only.")
    if args.split != "all":
        gen = gen[gen["split"] == args.split].reset_index(drop=True)
        if gen.empty:
            sys.exit(f"no generated answers with split={args.split}")
    y = test["label_3"].map(M.LABELS3.index).to_numpy()

    # ------------------------------------------------------------------ scores
    rows = []
    for name in names:
        rec = runs[name]
        mpath = D.repo_path(args.runs_dir) / rec["run_dir"] / "metrics.json"
        mt = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {}
        val, bc = mt.get("val_metrics", {}), mt.get("best_config", {})
        lo, hi = boot_ci(y, test[f"{name}_label"].map(M.LABELS3.index).to_numpy())
        rows.append({
            "classifier": name,
            "base model": rec["model"],
            "max len": bc.get("max_length"),
            "val F1": val.get("macro_f1", np.nan),
            "val DISC F1": val.get("disclaimer_f1", np.nan),
            "test F1": rec["test"]["macro_f1"],
            "test 95% CI": f"{lo:.3f}-{hi:.3f}",
            "test DISC F1": rec["test"]["disclaimer_f1"],
            "log loss": rec["test"]["log_loss"],
            "val parity": rec["val_parity"]["pred_agreement"],
            "truncated": rec.get("generated_truncated_at_max_length", 0),
            "gen/s": f"{rec['generated_per_sec']:.1f}" if "generated_per_sec" in rec else "",
        })
    scores = pd.DataFrame(rows)

    # ------------------------------------------------------- generated answers
    mix = label_mix(gen, names)
    counts = pd.concat({c: pd.crosstab(gen["gen_model"], gen[f"{c}_label"])
                           .reindex(columns=M.LABELS3, fill_value=0) for c in names},
                       names=["classifier"]).reset_index()

    agree = pd.DataFrame([
        {"gen model": g, "n": len(sub),
         "agree %": 100 * float(sub["classifiers_agree"].mean()),
         "n disagree": int((~sub["classifiers_agree"]).sum())}
        for g, sub in gen.groupby("gen_model")])
    test_agreement = float(test["classifiers_agree"].mean())

    a, b = names[0], names[1]
    crosstabs = {g: pd.crosstab(sub[f"{a}_label"], sub[f"{b}_label"])
                      .reindex(index=M.LABELS3, columns=M.LABELS3, fill_value=0)
                 for g, sub in gen.groupby("gen_model")}
    disagree = gen.loc[~gen["classifiers_agree"],
                       GEN_KEYS + ["split", "prompt", "answer"]
                       + [f"{c}_label" for c in names]
                       + [f"{c}_confidence" for c in names]]

    # Generations that ran out of budget rarely end in a refusal -- a refusal is short,
    # so hitting max_new_tokens is itself evidence the model complied. Kept visible
    # because it shifts the mix for any generation model that truncates often.
    trunc = pd.concat({c: pd.crosstab(gen["hit_max_new_tokens"], gen[f"{c}_label"],
                                      normalize="index")
                          .reindex(columns=M.LABELS3, fill_value=0).mul(100)
                       for c in names}, names=["classifier"]).reset_index()

    # ------------------------------------------------------------------- write
    mix.to_csv(out / "pair_table.csv", index=False)
    disagree.to_csv(out / "disagreements.csv", index=False, encoding="utf-8-sig")
    M.write_json({"classifiers": names,
                  "sources": [str(p) for p in preds],
                  "split": args.split,
                  "scores": scores.to_dict("records"),
                  "label_mix_pct": mix.to_dict("records"),
                  "label_counts": counts.to_dict("records"),
                  "agreement_by_gen_model": agree.to_dict("records"),
                  "test_agreement": test_agreement,
                  "crosstabs_by_gen_model": {g: t.to_dict() for g, t in crosstabs.items()},
                  "labels_by_hit_max_new_tokens_pct": trunc.to_dict("records")},
                 out / "pair.json")

    num = lambda df: df.map(lambda v: round(v, 3) if isinstance(v, float) else v)  # noqa: E731
    per_model = ", ".join(f"{k}: {v}" for k, v in
                          gen["gen_model"].value_counts().sort_index().items())
    scope = ("" if args.split == "all"
             else f", restricted to prompts from the {args.split} split")
    lines = [
        f"# {a} vs {b} on the three generation models", "",
        f"Both classifiers labelled {len(gen)} generated answers{scope} ({per_model}). "
        "Each is used on its own -- agreement is reported, the two are never combined. "
        f"Test scores are the {len(test)}-row held-out split, which neither run trained "
        "or selected on; overlapping 95% intervals mean the gap between them is not "
        "established. Single seed (42).", ""]
    if len(preds) > 1:
        lines += [f"Labels come from {len(preds)} separate predict.py runs "
                  f"({', '.join(p.name for p in preds)}), merged on "
                  "`(gen_model, prompt_id)`. Each run verified its own checkpoint against "
                  "its training-time val probabilities when it was made.", ""]
    if args.split == "all":
        lines += [
            "The generation prompts are the dataset's own prompts, so for about 70% of "
            "them the classifier trained on a *different* model's answer to the same "
            "prompt. That is fine for measuring how often each generation model refuses "
            "-- the classifier is being applied, not scored, and it never sees the prompt "
            "text, only the response. `--split test` reruns these tables on the 264 "
            "prompts per model that no classifier trained on.", ""]
    lines += [
        "## Classifiers", "", md_table(num(scores)), "",
        "`val parity` is the fraction of val rows where re-predicting reproduced the "
        "label saved at training time; anything below 1.000 is bf16 nondeterminism on "
        "near-tie rows. `truncated` counts generated answers that filled the model's "
        "max length.", "",
        "## Label mix on generated answers (% of each generation model)", "",
        md_table(mix, ".1f"), "",
        "The same numbers as counts:", "", md_table(counts, ".0f"), "",
        "## Agreement between the two classifiers", "",
        md_table(num(agree), ".1f"), "",
        f"On the test split they agree on {100 * test_agreement:.1f}% of rows. "
        f"The {len(disagree)} disagreeing generated answers are in `disagreements.csv`.",
        "", f"### Where they disagree (rows = {a}, columns = {b})", "",
    ]
    for g, t in crosstabs.items():
        lines += [f"**{g}**", "", md_table(t.reset_index(names=f"{a} / {b}"), ".0f"), ""]
    lines += [
        "## Label mix by whether generation hit max_new_tokens (%)", "",
        md_table(num(trunc), ".1f"), "",
        "An answer that ran out of token budget is almost never a refusal -- refusals are "
        "short, so truncation is itself evidence of compliance. This shifts the mix for "
        "any generation model that truncates often.", "",
    ]
    (out / "pair_table.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
