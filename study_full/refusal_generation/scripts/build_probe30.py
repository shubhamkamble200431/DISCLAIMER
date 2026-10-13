"""Pick the 30 screening prompts: 10 NORMAL, 10 DISCLAIMER, 10 REFUSAL.

A prompt is eligible only if all four judgements concur:
  - longformer and modernbert_large agree, on all 3 generation models
  - all 3 generation models produced the same label

The shared label travels with the prompt as `group`, so the runner writes it into
answers.jsonl as splits=screen_<LABEL> and the candidate's answer can be crosstabbed
against it later.

  python -X utf8 scripts/build_probe30.py
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT.parent / "refusal_classifier" / "results" / "predictions" / "generated_predictions.parquet"
OUT = ROOT / "data" / "screen" / "probe30.parquet"

ap = argparse.ArgumentParser()
ap.add_argument("--per-class", type=int, default=10)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

g = pd.read_parquet(PRED)

agree = g.groupby("prompt_id").classifiers_agree.all()
lab = g.pivot(index="prompt_id", columns="gen_model", values="longformer_label")
eligible = agree & (lab.nunique(axis=1) == 1)

pool = pd.DataFrame({"prompt_id": lab.index[eligible],
                     "group": lab[eligible].iloc[:, 0].values})
pool = pool.merge(g[["prompt_id", "prompt"]].drop_duplicates("prompt_id"), on="prompt_id")

print("eligible pool:", pool.group.value_counts().to_dict())

sel = pd.concat([d.sample(min(args.per_class, len(d)), random_state=args.seed)
                 for _, d in pool.groupby("group")])
sel = sel.sort_values(["group", "prompt_id"]).reset_index(drop=True)

sel["n_chars"] = sel.prompt.str.len()
OUT.parent.mkdir(parents=True, exist_ok=True)
sel.to_parquet(OUT, index=False)

print("picked:", sel.group.value_counts().to_dict())
print(f"chars: median {int(sel.n_chars.median())}  max {int(sel.n_chars.max())}")
print(f"wrote {OUT}")
