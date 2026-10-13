"""Side-by-side comparison of every classifier that has predictions. Reads only; no GPU.

For each predict.py output dir (--pred / --pred-glob) it joins the run's training
metrics (results/runs/<run_dir>/metrics.json) with its test metrics, bootstraps a 95%
CI for test macro-F1 (528 rows: small differences are noise), and lines up every
classifier's labels on the generated answers. Classifiers stay independent -- this
reports agreement, it never combines them.

Outputs (under --out):
  comparison.md                      the tables below, ranked by test macro-F1
  comparison.csv / comparison.json   same numbers, machine-readable
  generated_all_classifiers.parquet  one row per generated answer, a label column per
  generated_all_classifiers.csv      classifier (utf-8-sig for Excel)

Usage:
  python -X utf8 scripts/compare_runs.py --pred results/predictions \
      --pred-glob "results/predictions_wide/*" --out results/wide
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data as D       # noqa: E402
from src import metrics as M    # noqa: E402

N_BOOT = 2000
GEN_KEYS = ["gen_model", "prompt_id"]
TEST_KEYS = ["prompt_id", "side"]


def macro_f1(y, p, k=3):
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm)
    denom = cm.sum(0) + cm.sum(1)
    return float(np.mean(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)))


def boot_ci(y, p, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(N_BOOT, len(y)))
    s = np.array([macro_f1(y[i], p[i]) for i in idx])
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def md_table(df, floatfmt=".3f"):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = [format(v, floatfmt) if isinstance(v, float) and not np.isnan(v)
                 else ("" if isinstance(v, float) else str(v)) for v in r]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", action="append", default=[])
    ap.add_argument("--pred-glob", action="append", default=[])
    ap.add_argument("--expect", nargs="*", default=[],
                    help="candidates that should be present; missing ones are listed")
    ap.add_argument("--runs-dir", default="results/runs")
    ap.add_argument("--out", default="results/wide")
    args = ap.parse_args()

    dirs = [D.repo_path(p) for p in args.pred]
    for g in args.pred_glob:
        dirs += sorted(Path(p) for p in glob.glob(str(D.repo_path(g))))
    dirs = [d for d in dirs if (d / "summary.json").exists()]
    out = D.repo_path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows, tests, gens, geos = [], [], [], []
    short = {"NORMAL": "N", "DISCLAIMER": "D", "REFUSAL": "R"}
    for d in dirs:
        summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        if summary.get("smoke"):
            continue
        test = pd.read_parquet(d / "test_predictions.parquet")
        gen = pd.read_parquet(d / "generated_predictions.parquet")
        for name, rec in summary["runs"].items():
            mpath = D.repo_path(args.runs_dir) / rec["run_dir"] / "metrics.json"
            mt = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {}
            val, bc = mt.get("val_metrics", {}), mt.get("best_config", {})
            y = test["label_3"].map(M.LABELS3.index).to_numpy()
            p = test[f"{name}_label"].map(M.LABELS3.index).to_numpy()
            lo, hi = boot_ci(y, p)
            collapsed = bool(val) and (val.get("disclaimer_f1", 1) == 0
                                       or val.get("macro_f1", 1) < 0.3)
            lab = gen[f"{name}_label"].value_counts(normalize=True)
            rows.append({
                "classifier": name,
                "base model": rec["model"],
                "trained on": mt.get("env", {}).get("device_name", ""),
                "max len": bc.get("max_length"),
                "lr": f"{bc['lr']:g}" if bc.get("lr") is not None else "",
                "best epoch": bc.get("best_epoch"),
                "trainable M": mt.get("n_trainable_params", 0) / 1e6,
                "val macroF1": val.get("macro_f1", np.nan),
                "val DISC F1": val.get("disclaimer_f1", np.nan),
                "test macroF1": rec["test"]["macro_f1"],
                "test 95% CI": f"{lo:.3f}-{hi:.3f}",
                "test DISC F1": rec["test"]["disclaimer_f1"],
                "test acc": rec["test"]["accuracy"],
                "test log loss": rec["test"]["log_loss"],
                "train min": mt.get("total_wall_seconds", np.nan) / 60,
                "train peak GB": mt.get("peak_vram_bytes", np.nan) / 1e9,
                "gen %DISC": 100 * float(lab.get("DISCLAIMER", 0)),
                "gen %REF": 100 * float(lab.get("REFUSAL", 0)),
                "val parity": rec["val_parity"]["pred_agreement"],
                "flag": "COLLAPSED" if collapsed else "",
            })
            tests.append(test[TEST_KEYS + ["label_3", f"{name}_label"]])
            gens.append(gen[GEN_KEYS + [f"{name}_label", f"{name}_confidence"]])
            geom = mt.get("geometry", {})
            for pair, v in geom.get("pairs", {}).items():
                if "euclidean" not in v:
                    continue
                a, b = pair.split("|")
                geos.append({"classifier": name, "pooled": geom.get("pooling", ""),
                             "pair": f"{short.get(a, a)}-{short.get(b, b)}",
                             "centroid dist": v["euclidean"], "cosine": v["cosine"],
                             "separation": v["separation"]})

    if not rows:
        sys.exit("no non-smoke predictions found")
    table = (pd.DataFrame(rows).drop_duplicates("classifier", keep="last")
             .sort_values("test macroF1", ascending=False).reset_index(drop=True))
    names = table["classifier"].tolist()
    missing = [c for c in args.expect if c not in names]

    base_gen = pd.read_parquet(dirs[0] / "generated_predictions.parquet")[
        GEN_KEYS + ["split", "prompt", "answer", "hit_max_new_tokens", "empty_answer"]]
    all_gen = base_gen
    for g in gens:
        all_gen = all_gen.merge(g.loc[:, ~g.columns.duplicated()], on=GEN_KEYS, how="left",
                                suffixes=("", "_dup"))
    all_gen = all_gen[[c for c in all_gen.columns if not c.endswith("_dup")]]
    all_test = tests[0][TEST_KEYS + ["label_3"]]
    for t in tests:
        all_test = all_test.merge(t.drop(columns="label_3"), on=TEST_KEYS, how="left",
                                  suffixes=("", "_dup"))

    def agreement(df):
        return pd.DataFrame([[float((df[f"{a}_label"] == df[f"{b}_label"]).mean())
                              for b in names] for a in names], index=names, columns=names)

    agree_gen, agree_test = agreement(all_gen), agreement(all_test)
    by_gen_model = pd.concat(
        {c: pd.crosstab(all_gen["gen_model"], all_gen[f"{c}_label"], normalize="index")
             .reindex(columns=M.LABELS3, fill_value=0).mul(100) for c in names},
        names=["classifier"]).reset_index()

    table.to_csv(out / "comparison.csv", index=False)
    all_gen.to_parquet(out / "generated_all_classifiers.parquet", index=False)
    all_gen.to_csv(out / "generated_all_classifiers.csv", index=False, encoding="utf-8-sig")
    M.write_json({"table": table.to_dict("records"), "missing": missing,
                  "agreement_generated": agree_gen.round(4).to_dict(),
                  "agreement_test": agree_test.round(4).to_dict()},
                 out / "comparison.json")

    num = lambda df: df.map(lambda v: round(v, 3) if isinstance(v, float) else v)  # noqa: E731
    lines = [
        "# Refusal classifier comparison", "",
        f"{len(names)} classifiers, each used on its own (no ensembling). Ranked by test "
        f"macro-F1 on the {len(all_test)}-row test split, which no run trained or selected "
        "on. Overlapping 95% bootstrap intervals mean the ranking between those rows is "
        "not established. Single seed (42) throughout.", "",
        "- Wall time and peak VRAM are only comparable between runs on the same device.",
        "- `max len` differs: 512-position models cut ~5.5% of responses.",
        "- `tinyllama_lora` trains LoRA adapters + head, not every weight.",
        "- `COLLAPSED` = val DISCLAIMER F1 of 0 or macro-F1 < 0.3: a failed run, not a score.",
    ]
    if missing:
        lines += ["", f"**No predictions for:** {', '.join(missing)} "
                      "(see results/wide/status.tsv and logs)."]
    quality = table[["classifier", "val macroF1", "test macroF1", "test 95% CI",
                     "test DISC F1", "test log loss", "flag"]].rename(columns={
        "val macroF1": "val F1", "test macroF1": "test F1", "test DISC F1": "DISC F1",
        "test log loss": "log loss"})
    cost = table[["classifier", "trainable M", "max len", "train min", "train peak GB",
                  "trained on", "gen %DISC", "gen %REF"]].rename(columns={
        "trainable M": "trained M", "train peak GB": "VRAM GB"}).sort_values("train min")
    cost = cost.round({"trained M": 1, "train min": 1, "VRAM GB": 1,
                       "gen %DISC": 1, "gen %REF": 1})
    geo = (pd.DataFrame(geos).drop_duplicates(["classifier", "pair"], keep="last")
           if geos else pd.DataFrame())
    if not geo.empty:
        geo["_rank"] = geo["classifier"].map({c: i for i, c in enumerate(names)})
        geo = geo.sort_values(["_rank", "pair"]).drop(columns="_rank")
    off_parity = table.loc[table["val parity"] < 1, ["classifier", "val parity"]]

    lines += ["", "## Scores", "", md_table(num(quality)),
              "", "## Cost and downstream label mix", "", md_table(cost, ".1f"),
              "", "The full wide table, including every column, is in `comparison.csv`."]
    if not off_parity.empty:
        lines += ["", "Re-predicted val does not match training-time probabilities "
                      "exactly for: "
                  + ", ".join(f"{c} ({v:.3f})" for c, v in
                              zip(off_parity["classifier"], off_parity["val parity"]))
                  + " -- bf16 nondeterminism on near-tie rows, within the 0.99 gate."]
    if not geo.empty:
        lines += [
            "", "## Label geometry (val split, final-layer vectors)", "",
            "Centroid distance between each pair of classes, the cosine between those "
            "centroids, and `separation` = distance divided by the mean width of the two "
            "classes. **Read separation**: distances are not comparable across models, "
            "since each has its own embedding scale. Above 1 means the centroids sit "
            "further apart than the classes are wide. `pooled` says which vector was "
            "measured -- [CLS] for encoders, the last real token for decoders.", "",
            md_table(num(geo))]
    lines += ["", "## Label mix on generated answers (% of each generation model)", "",
              md_table(num(by_gen_model), ".1f"),
              "", "## Agreement on generated answers (fraction of identical labels)", "",
              md_table(num(agree_gen.reset_index(names="classifier"))),
              "", "## Agreement on the test split", "",
              md_table(num(agree_test.reset_index(names="classifier"))), ""]
    (out / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
