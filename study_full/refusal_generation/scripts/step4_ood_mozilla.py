"""Step 4 of plan.md -- OOD direction transfer on Mozilla/sensitive-topic-disclaimer-eval.

WHAT
----
Take the refusal / disclaimer directions learned IN DOMAIN (step 2, on the 40% Direction
split of the chatbot prompts), freeze them, and apply them to a different prompt
distribution. If `d_D` moves the disclaimer rate on a dataset it was never estimated on,
the direction is a property of the model rather than of our prompt set.

WHY IT IS ORGANISED IN STAGES
-----------------------------
The DGX runs generation under conda env `base` (transformers 5.14.1) and the classifier
under `refusal_clf` (transformers 5.17.0), so one process cannot both generate and score.
Every stage is therefore a separate `--stage`, launchable as its own SLURM job:

  prepare        CPU              fetch + pin + assert the dataset, assign integer ids
  generate       GPU (gen env)    batch-1 baseline answers WITH prompt_last capture
  score          GPU (clf env)    C* labels for those baseline answers
  steer          GPU (gen env)    frozen d_hat, batched, parity grid + batched baseline;
                                  shardable with --shard i/n (SLURM array)
  score-steered  GPU (clf env)    C* labels for the steered answers
  metrics        CPU              pp vs baseline, paired bootstrap, plots, report
  all            everything in order (laptop / smoke use only)

WHAT THIS DATASET IS, AND IS NOT (plan.md 1.13 -- READ THIS BEFORE READING THE NUMBERS)
---------------------------------------------------------------------------------------
290 rows, one `train` split, config name `disclaimer` (NOT `default`), columns
id / query / topic / is_sensitive. 29 topics x 10 rows. is_sensitive is per-ROW, not
per-topic (13 topics are mixed): 126 sensitive rows and 164 benign rows, MEASURED against
the pinned revision -- plan.md 1.13's 170/120 topic-level figures are wrong. It ships NO responses and NO human disclaimer
labels. The queries are 20-64 characters, median 7 words, entirely synthetic, with zero
harmful or adversarial content -- the dataset card explicitly says it is not for
evaluating refusal behaviour.

Three consequences this script is built around, and states in its own output:

  1. Baseline REFUSAL will be ~0, so the `d_R` arm is STRUCTURALLY FLOORED. A flat d_R
     curve here is a property of the dataset, not evidence that d_R does not transfer.
     The summary says so explicitly instead of presenting a floored arm as a null result.
  2. `is_sensitive` describes the QUERY, not the response. It is a secondary
     stratification only and is NEVER ground truth for C*. The 126 sensitive rows are the
     transfer set of interest; the 164 benign rows are an over-disclaimering /
     false-positive check. They are reported SEPARATELY, never pooled as "OOD".
  3. Power is thin. At n=290 (126/164 split) a paired bootstrap CI half-width is roughly
     +/-4 pp against a 14-point shift. The summary states the ACHIEVED half-width for
     every headline rather than burying it.

The `id` column is 8-char hex, and predict.py forces `prompt_id` to int64, so integer ids
are assigned as `ood.id_offset + rank` (9,000,000,000+) with a side mapping kept in
id_map.json. The offset is far above the 1,750 in-domain ids, so a leak is obvious on
sight. `splits` is written as "ood" in every record.

LAYER CHOICE
------------
The PDF says "same layer_ids as selected on chatbot". Absolute row ids are not
well-defined across models with different block counts, so each model uses ITS OWN top
layers from data/study/selected_layers.json, ranked by the probe score, and the report
prints both the absolute row and its relative depth l / n_blocks. That choice is stated
in every artifact this script writes.

WHAT IS FROZEN AND WHAT IS RECOMPUTED
-------------------------------------
`d_hat` is FROZEN from the in-domain Direction split (data/study/directions/<slug>.npz);
this stage refuses to run if that file is absent. Only the scaling constants are
recomputed on the OOD activations:

  sigma_l = sd( h_l @ d_hat )        the x-axis denominator (alpha / sigma_l)
  r_l     = median_prompts ||h_l||   the norm-scale constant of plan.md 1.4/2c

The in-domain vs OOD sigma ratio is logged per layer. A large shift means alpha/sigma_l
is not comparable across the two datasets and the paper must say so.

THE GRID: IN-DOMAIN PARITY BY DEFAULT, --reduced ON REQUEST
-----------------------------------------------------------
The DEFAULT grid is in-domain parity (user decision 2026-09-21): each model's own
top-`ood.layers_top` layers (10 = all of step 2's selection) x 2 directions x the full
steering.alphas grid (`ood.alphas: inherit`), so every OOD pp has an in-domain twin cell
for cell. That is 290 x (10 x 2 x 10) + 290 = 58,290 generations per model, 174,870 for
three -- ~218 GPU-h at batch 8, run as the 8-task array 42_step4_steer_dgx.slurm.
`--reduced` switches to `ood.reduced` (top-3 layers x alphas [-10,-5,5,10], 21,750
generations): the cheap transfer check. Its condition tags are a SUBSET of the parity
grid's, so a reduced run's answers are reused, not regenerated, by a later parity run.
`--full-parity` is accepted as a no-op alias so older commands keep working.
Every reader of the grid goes through resolve_ood_grid(); nothing else reads
ood.layers_top / ood.alphas.

SHARDING (--shard i/n, exactly step 2's semantics)
--------------------------------------------------
The per-model condition list is baseline FIRST, then (layer, direction, alpha); condition k
belongs to shard i iff k % n == i - 1, so the ONE shared baseline is generated by shard 1
and by no other. Every shard plans the SAME static batch plan over all prompts (it depends
only on the prompts, the tokenizer and the batch size), and its sha256 is recorded, so the
baseline written by shard 1 is a valid subtrahend for conditions written by shard 8.
steered/<slug>/manifest.json always describes the FULL grid and is replaced atomically, so
eight shards writing it concurrently leave one well-formed, identical-in-substance file;
per-shard bookkeeping (preflight, OOM halvings, wall time, which tags it ran) goes to
steered/<slug>/shards/shard_<i>of<n>.json. --verify reads both.

MEASURED FACTS THIS SCRIPT OBEYS (see src/steer.py for the full list)
--------------------------------------------------------------------
* Saved row `l` is the OUTPUT of decoder block `l-1`; steering hooks layers[l-1] via
  model.get_decoder(). Never hardcoded -- src.steer.resolve_decoder does it.
* Rows 0 and n_blocks are not steerable (embedding output / post-final-norm).
* Capture point is `prompt_last`, independent of max_new_tokens.
* Activation capture is batch-1 (padding perturbs hidden states). Only the `steer` stage
  batches, and it captures nothing.
* The steered comparison baseline is generated in the `steer` stage through the IDENTICAL
  batched code path with identical batch membership. The batch-1 `generate` stage answers
  exist for the ACTIVATIONS; they are not the comparison baseline, because batched greedy
  decoding is not bitwise invariant to batch size.
* Greedy decoding => no within-cell variance; error bars come from the paired bootstrap.
* All metrics are defined on the ARGMAX LABEL only -- dynamic batch composition in the
  classifier shifts probabilities by up to 0.023.

Usage (PowerShell):
  python -X utf8 scripts/step4_ood_mozilla.py --stage prepare --dry-run
  python -X utf8 scripts/step4_ood_mozilla.py --stage prepare
  python -X utf8 scripts/step4_ood_mozilla.py --stage prepare --from-local data/ood/raw
  python -X utf8 scripts/step4_ood_mozilla.py --stage generate --models qwen3-5-9b
  python -X utf8 scripts/step4_ood_mozilla.py --stage score          # classifier env
  python -X utf8 scripts/step4_ood_mozilla.py --stage steer --dry-run   # both inventories
  python -X utf8 scripts/step4_ood_mozilla.py --stage steer --batch-size 8
  python -X utf8 scripts/step4_ood_mozilla.py --stage steer --shard 3/8
  python -X utf8 scripts/step4_ood_mozilla.py --stage steer --reduced
  python -X utf8 scripts/step4_ood_mozilla.py --stage score-steered  # classifier env
  python -X utf8 scripts/step4_ood_mozilla.py --stage metrics
  python -X utf8 scripts/step4_ood_mozilla.py --stage all --smoke --limit 8
  python -X utf8 scripts/step4_ood_mozilla.py --verify
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import generate as G    # noqa: E402
from src import steer as S       # noqa: E402

STAGES = ("prepare", "generate", "score", "steer", "score-steered", "metrics", "all")
STAGE_ORDER = ("prepare", "generate", "score", "steer", "score-steered", "metrics")

# The pinned shape of Mozilla/sensitive-topic-disclaimer-eval at revision
# 9a607f0874fe4302373983ea17c50b33f8aa7744 (plan.md 1.13). Any deviation means the
# revision pin did not take, or the local copy is a different export -- either way every
# downstream count in the paper would be wrong, so `prepare` aborts rather than adapting.
# MEASURED against the pinned revision on 2026-09-20, NOT taken from plan.md.
# plan.md 1.13 states "17 sensitive topics (170 rows) and 12 benign control topics (120
# rows)". That is WRONG: it assumes is_sensitive is a property of the TOPIC. It is a
# property of the ROW. The real split is 126 sensitive / 164 benign, and 13 of the 29
# topics are mixed -- e.g. 'medical_health' carries both "What does a diagnosis of
# hypertension mean" (benign) and "Is this medication safe with my other drugs"
# (sensitive). The transfer set of interest is therefore 126 rows, not 170, which also
# widens the bootstrap CI the metrics stage reports.
EXPECT = {
    "n_rows": 290,
    "columns": ["id", "query", "topic", "is_sensitive"],
    "n_topics": 29,
    "rows_per_topic": 10,
    "n_sensitive_rows": 126,
    "n_benign_rows": 164,
    "n_mixed_topics": 13,
    "n_fully_sensitive_topics": 4,
    "n_fully_benign_topics": 12,
    "id_len": 8,
}

OOD_SPLIT_TAG = "ood"
CAPTURE_POINT = "prompt_last"
CSTAR_PREFIX = "cstar"

# Baseline REFUSAL at or below this rate means the d_R arm has no room to move DOWN and
# very little signal to move up: the summary must say "structurally floored", not "null".
FLOOR_RATE_PCT = 2.0
# |log2(sigma_ood / sigma_indomain)| above this is called out: alpha/sigma_l stops being
# comparable across the two datasets, which is a paper-level caveat, not a bug.
SIGMA_RATIO_WARN = 1.5


class Tee:
    """stdout + <out_dir>/<stage>.log, the house logger."""

    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


# --------------------------------------------------------------------------- #
# small shared helpers
# --------------------------------------------------------------------------- #
def env_info() -> dict:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except Exception:
        pass
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    return info


def sha256_lines(lines) -> str:
    """sha256 over a newline-joined list -- the project's committed-artifact hash."""
    return hashlib.sha256("\n".join(str(x) for x in lines).encode("utf-8")).hexdigest()


def write_json(obj, path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def write_json_atomic(obj, path) -> None:
    """write_json, but via a per-process temp file and os.replace.

    Eight SLURM array tasks write the same steered/<slug>/manifest.json. A plain
    write_text truncates then writes, so a reader (or a second writer) can see a torn,
    half-written file; os.replace swaps the complete file in as one rename, so every
    reader sees exactly one shard's complete copy.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{platform.node()}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, p)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ood_paths(cfg) -> dict:
    ood = G.repo_path(cfg["output"]["ood_root"])
    return {
        "root": ood,
        "prompts": ood / "prompts.parquet",
        "id_map": ood / "id_map.json",
        "prepare": ood / "prepare.json",
        "generated": ood / "generated",
        "steered": ood / "steered",
        "scores": ood / "scores",
        "logs": ood / "logs",
        "study": G.repo_path(cfg["output"]["study_root"]),
        "directions": G.repo_path(cfg["output"]["study_root"]) / "directions",
        "generated_indomain": G.repo_path(cfg["output"]["generated_root"]),
        # A smoke run must never overwrite the real report directory. ood_root already
        # carries the _smoke suffix by then, so key off it.
        "reports": G.repo_path("results/reports/step4_smoke"
                               if str(cfg["output"]["ood_root"]).endswith("_smoke")
                               else "results/reports/step4"),
    }


def load_prompts(paths, limit: int | None = None) -> pd.DataFrame:
    if not paths["prompts"].exists():
        raise SystemExit(f"{paths['prompts']} is missing -- run --stage prepare first")
    df = pd.read_parquet(paths["prompts"]).sort_values("prompt_id").reset_index(drop=True)
    if limit:
        df = df.head(int(limit)).copy()
    return df


def planned_n_prompts(paths, args, log) -> int:
    """Prompt count for --dry-run planning, WITHOUT requiring `prepare` to have run.

    Planning a run is exactly what you do while deciding whether to fetch the dataset at
    all, so --dry-run must not depend on prompts.parquet existing. With no parquet on
    disk we plan against EXPECT['n_rows'] -- the pinned 290 that `prepare` asserts anyway,
    so the plan a dry-run prints is the plan the real run executes.
    """
    if paths["prompts"].exists():
        return len(load_prompts(paths, args.limit))
    n = EXPECT["n_rows"]
    capped = f" capped to {args.limit} by --limit" if args.limit and args.limit < n else ""
    if args.limit:
        n = min(n, int(args.limit))
    log(f"  no {paths['prompts']} yet -- planning against the pinned "
        f"{EXPECT['n_rows']}-row expectation{capped}; "
        f"run --stage prepare to materialise it")
    return n


# --------------------------------------------------------------------------- #
# stage 1: prepare
# --------------------------------------------------------------------------- #
def _read_local(path: Path, log):
    """An offline copy: a save_to_disk directory, or a parquet / jsonl / csv file."""
    if path.is_dir():
        try:
            from datasets import load_from_disk
            log(f"  reading save_to_disk dataset at {path}")
            ds = load_from_disk(str(path))
            if hasattr(ds, "keys"):                       # DatasetDict
                ds = ds["train"]
            return ds.to_pandas()
        except Exception as e:
            log(f"  load_from_disk failed ({type(e).__name__}: {e}); looking for a data file")
            hits = sorted(list(path.glob("*.parquet")) + list(path.glob("*.jsonl"))
                          + list(path.glob("*.json")) + list(path.glob("*.csv")))
            if not hits:
                raise FileNotFoundError(f"no parquet/jsonl/json/csv under {path}")
            path = hits[0]
    log(f"  reading {path}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix in (".json", ".jsonl"):
        return pd.read_json(path, lines=path.suffix == ".jsonl")
    raise ValueError(f"unsupported local format: {path}")


def fetch_ood(cfg, from_local: str | None, log) -> tuple[pd.DataFrame, dict]:
    o = cfg["ood"]
    if from_local:
        src = Path(from_local)
        src = src if src.is_absolute() else G.repo_path(from_local)
        df = _read_local(src, log)
        source = {"kind": "local", "path": str(src)}
    else:
        from datasets import load_dataset
        log(f"  load_dataset({o['hf_id']!r}, {o['config_name']!r}, "
            f"revision={o['revision'][:12]}..., split='train')")
        ds = load_dataset(o["hf_id"], o["config_name"], revision=o["revision"], split="train")
        df = ds.to_pandas()
        source = {"kind": "hub", "hf_id": o["hf_id"], "config_name": o["config_name"],
                  "revision": o["revision"]}
    return df, source


def assert_ood_shape(df: pd.DataFrame, log) -> dict:
    """Fail loudly if the pinned revision is not what plan.md 1.13 measured."""
    problems = []
    cols = list(df.columns)
    if sorted(cols) != sorted(EXPECT["columns"]):
        problems.append(f"columns {cols} != {EXPECT['columns']}")
    if len(df) != EXPECT["n_rows"]:
        problems.append(f"{len(df)} rows, expected {EXPECT['n_rows']}")
    if problems:
        raise RuntimeError("OOD dataset does not match the pinned expectation "
                           "(plan.md 1.13): " + "; ".join(problems))

    df["is_sensitive"] = df["is_sensitive"].astype(bool)
    df["topic"] = df["topic"].astype(str)
    df["id"] = df["id"].astype(str)

    if df["id"].duplicated().any():
        raise RuntimeError(f"{int(df['id'].duplicated().sum())} duplicate hex ids")
    bad_len = df.loc[df["id"].str.len() != EXPECT["id_len"], "id"].tolist()
    if bad_len:
        problems.append(f"{len(bad_len)} ids are not {EXPECT['id_len']} chars: {bad_len[:5]}")

    # `frac` is the share of sensitive ROWS in the topic: 1.0 fully sensitive, 0.0 fully
    # benign, anything between = mixed. `sens` (nunique) alone cannot tell a fully
    # sensitive topic from a fully benign one.
    per_topic = df.groupby("topic").agg(n=("id", "size"), sens=("is_sensitive", "nunique"),
                                        frac=("is_sensitive", "mean"),
                                        is_sensitive=("is_sensitive", "first"))
    if len(per_topic) != EXPECT["n_topics"]:
        problems.append(f"{len(per_topic)} topics, expected {EXPECT['n_topics']}")
    odd = per_topic.index[per_topic["n"] != EXPECT["rows_per_topic"]].tolist()
    if odd:
        problems.append(f"topics without exactly {EXPECT['rows_per_topic']} rows: {odd}")
    # is_sensitive is per-ROW, not per-topic (13 of 29 topics are mixed), so the
    # sensitive/benign reporting split must be taken row-wise. Assert the mix is the
    # pinned one rather than asserting topics are pure.
    n_mixed = int(((per_topic["frac"] > 0) & (per_topic["frac"] < 1)).sum())
    n_full_sens = int((per_topic["frac"] == 1).sum())
    n_full_ben = int((per_topic["frac"] == 0).sum())
    n_sens_r = int(df["is_sensitive"].sum())
    n_ben_r = len(df) - n_sens_r
    for got, want, what in ((n_mixed, EXPECT["n_mixed_topics"], "mixed-sensitivity topics"),
                            (n_full_sens, EXPECT["n_fully_sensitive_topics"],
                             "fully sensitive topics"),
                            (n_full_ben, EXPECT["n_fully_benign_topics"],
                             "fully benign topics"),
                            (n_sens_r, EXPECT["n_sensitive_rows"], "sensitive rows"),
                            (n_ben_r, EXPECT["n_benign_rows"], "benign rows")):
        if got != want:
            problems.append(f"{got} {what}, expected {want}")

    if problems:
        raise RuntimeError("OOD dataset does not match the pinned expectation "
                           "(plan.md 1.13): " + "; ".join(problems))

    lens = df["query"].astype(str).str.len()
    log(f"  shape OK: {len(df)} rows, {len(per_topic)} topics "
        f"({n_full_sens} fully sensitive / {n_full_ben} fully benign / "
        f"{n_mixed} MIXED), {n_sens_r} sensitive rows / {n_ben_r} benign rows")
    log(f"  NOTE is_sensitive is per-ROW, not per-topic. plan.md 1.13's "
        f"'17 sensitive topics (170 rows) / 12 benign (120)' is wrong; the transfer set "
        f"is {n_sens_r} rows and the benign control is {n_ben_r}.")
    log(f"  query chars: min {int(lens.min())} median {int(lens.median())} "
        f"max {int(lens.max())}")
    return {"n_rows": int(len(df)), "n_topics": int(len(per_topic)),
            "n_fully_sensitive_topics": n_full_sens,
            "n_fully_benign_topics": n_full_ben, "n_mixed_topics": n_mixed,
            "n_sensitive_rows": n_sens_r, "n_benign_rows": n_ben_r,
            "query_chars": {"min": int(lens.min()), "median": int(lens.median()),
                            "max": int(lens.max())},
            "topics": {str(t): {"n": int(r["n"]),
                                "sensitive_fraction": round(float(r["frac"]), 3)}
                       for t, r in per_topic.iterrows()}}


def build_ood_table(df: pd.DataFrame, offset: int) -> pd.DataFrame:
    """Assign integer prompt_ids and the columns src/generate.py's writers expect.

    Rank is taken over the hex id sorted ascending, NOT over the loader's row order, so a
    hub download and an offline copy produce byte-identical ids and therefore the same
    committed sha256. int(hex, 16) is not used: it would collide with nothing but produce
    unreadable 10-digit ids outside the offset band, hiding a leak instead of exposing it.
    """
    out = df.sort_values("id", kind="stable").reset_index(drop=True).copy()
    out["prompt_id"] = (int(offset) + np.arange(len(out), dtype=np.int64)).astype("int64")
    out["prompt"] = out["query"].astype(str)
    out["splits"] = OOD_SPLIT_TAG               # written verbatim into every record
    out["prompt_sha1"] = [G.sha1(p) for p in out["prompt"]]
    out["n_chars"] = out["prompt"].str.len().astype("int64")
    out = out.rename(columns={"id": "ood_hex_id"})
    return out[["prompt_id", "prompt", "splits", "prompt_sha1", "n_chars",
                "ood_hex_id", "topic", "is_sensitive"]]


def prompts_digest(tbl: pd.DataFrame) -> str:
    return sha256_lines(f"{int(r.prompt_id)},{r.ood_hex_id},{r.prompt_sha1}"
                        for r in tbl.itertuples(index=False))


def stage_prepare(cfg, args, paths, log) -> None:
    log("\n=== stage: prepare (CPU) ===")
    o = cfg["ood"]
    log(f"  dataset  {o['hf_id']}  config={o['config_name']}  "
        f"revision={o['revision']}")

    if args.dry_run:
        log(f"  would write {paths['prompts']}, {paths['id_map']}, {paths['prepare']}")
        log(f"  expected shape: {EXPECT['n_rows']} rows, {EXPECT['n_topics']} topics, "
            f"{EXPECT['n_sensitive_rows']} sensitive / {EXPECT['n_benign_rows']} benign")
        return

    raw, source = fetch_ood(cfg, args.from_local, log)
    stats = assert_ood_shape(raw, log)
    tbl = build_ood_table(raw, o["id_offset"])
    digest = prompts_digest(tbl)

    lo, hi = int(tbl.prompt_id.min()), int(tbl.prompt_id.max())
    if lo < int(o["id_offset"]):
        raise RuntimeError(f"prompt_id {lo} fell below the offset {o['id_offset']}")
    log(f"  prompt_id range {lo} .. {hi}  (offset {o['id_offset']}, far above the 1,750 "
        f"in-domain ids so any leak is visible on sight)")
    log(f"  sha256 {digest[:16]}...")

    paths["root"].mkdir(parents=True, exist_ok=True)
    tbl.to_parquet(paths["prompts"], index=False)
    write_json({
        "offset": int(o["id_offset"]),
        "n": int(len(tbl)),
        "note": "int_id -> hex id / topic / is_sensitive. is_sensitive describes the "
                "QUERY, never the response, and is never ground truth for C*.",
        "map": {str(int(r.prompt_id)): {"hex_id": r.ood_hex_id, "topic": r.topic,
                                        "is_sensitive": bool(r.is_sensitive)}
                for r in tbl.itertuples(index=False)},
    }, paths["id_map"])
    write_json({
        "stage": "prepare",
        "seed": int(cfg["study"]["seed"]),
        "inputs": source,
        "expectation": EXPECT,
        "observed": stats,
        "id_offset": int(o["id_offset"]),
        "prompt_id_range": [lo, hi],
        "splits_tag": OOD_SPLIT_TAG,
        "sha256": digest,
        "caveats": [
            "No responses and no human disclaimer labels ship with this dataset.",
            "Queries are synthetic, 20-64 chars, zero harmful or adversarial content; "
            "the card says it is NOT for evaluating refusal behaviour.",
            "Baseline REFUSAL will be ~0, so the d_R arm is structurally floored.",
            "is_sensitive is a QUERY property: secondary stratification only.",
        ],
        "env": env_info(),
        "written_utc": datetime.now(timezone.utc).isoformat(),
    }, paths["prepare"])
    log(f"  wrote {paths['prompts']}")
    log(f"  wrote {paths['id_map']}")
    log(f"  wrote {paths['prepare']}")


def verify_prepare(paths, log) -> bool:
    if not paths["prepare"].exists() or not paths["prompts"].exists():
        log(f"  [skip] no committed prepare artifact at {paths['prepare']}")
        return True
    committed = read_json(paths["prepare"])["sha256"]
    got = prompts_digest(pd.read_parquet(paths["prompts"]))
    ok = committed == got
    log(f"  prepare: committed {committed[:16]}  recomputed {got[:16]}  "
        f"{'MATCH' if ok else 'MISMATCH'}")
    return ok


# --------------------------------------------------------------------------- #
# stage 2: generate (batch-1 baseline answers + prompt_last capture)
# --------------------------------------------------------------------------- #
def stage_generate(cfg, args, paths, specs, log) -> None:
    log("\n=== stage: generate (GPU, batch 1, prompt_last capture at all layers) ===")
    layout = cfg["output"].get("hidden_layout", "stacked")
    if layout != "stacked":
        raise RuntimeError(f"output.hidden_layout is {layout!r}; step 4 requires 'stacked' "
                           f"(per_layer writes ~33 files per prompt and the OOD tree is "
                           f"re-read layer-by-layer by the steer stage)")
    points = list(cfg["capture"].get("points") or [])
    if CAPTURE_POINT not in points:
        raise RuntimeError(f"capture.points={points} does not include {CAPTURE_POINT!r}; "
                           f"the steer stage needs it to recompute sigma_l and r_l")
    if cfg["capture"].get("layers", "all") != "all":
        raise RuntimeError("capture.layers must be 'all': sigma_l/r_l are recomputed at "
                           "whichever layers the probe selected, which is not known here")

    # The dry-run is answered before load_prompts so that planning works offline, before
    # `prepare` has fetched anything. The config assertions above still run: a bad
    # hidden_layout / capture point should fail during planning, not after the fetch.
    if args.dry_run:
        n_p = planned_n_prompts(paths, args, log)
        n = n_p * len(specs)
        log(f"  cells: {n_p} prompts x {len(specs)} models = {n} generations")
        log(f"  ~{S.estimate_gpu_hours(n):.1f} GPU-h at batch 1 "
            f"({S.SEC_PER_GEN_A100_40GB}s/gen, A100-40GB)")
        return

    prompts = load_prompts(paths, args.limit)
    log(f"  prompts {len(prompts)}  models {[s.slug for s in specs]}")

    store_dtype = cfg["output"].get("store_dtype", "float16")
    compress = bool(cfg["output"].get("compress", False))

    for spec in specs:
        root, ans_dir, hid_dir = G.model_dirs(paths["generated"], spec.slug)
        ans_dir.mkdir(parents=True, exist_ok=True)
        hid_dir.mkdir(parents=True, exist_ok=True)
        mlog = Tee(root / "generate.log")
        mlog(f"\n=== {spec.slug} | {spec.hf_id} | "
             f"{datetime.now(timezone.utc).isoformat()} ===")

        todo = [r for r in prompts.itertuples(index=False)
                if args.overwrite
                or not G.already_done(ans_dir, hid_dir, int(r.prompt_id), layout)]
        n_skip = len(prompts) - len(todo)
        mlog(f"  {len(prompts)} prompts, {n_skip} already done, {len(todo)} to run")
        if not todo:
            mlog("  nothing to do")
            mlog.close()
            continue

        runner = G.Runner(spec, cfg["generation"], cfg["capture"], log=mlog)
        jsonl = open(root / "answers.jsonl", "a", encoding="utf-8", newline="\n")
        t0 = time.perf_counter()
        n_done, n_failed, failures, n_think = 0, 0, [], 0
        for i, row in enumerate(todo, 1):
            pid = int(row.prompt_id)
            try:
                answer, vecs, meta = runner.run(row.prompt)
                rec = G.write_prompt(ans_dir, hid_dir, row, answer, vecs, meta, spec,
                                     store_dtype, compress, layout)
                jsonl.write(json.dumps(
                    {"prompt_id": pid, "prompt": row.prompt, "answer": answer,
                     "splits": OOD_SPLIT_TAG, "topic": row.topic,
                     "is_sensitive": bool(row.is_sensitive),
                     "ood_hex_id": row.ood_hex_id,
                     "n_prompt_tokens": meta["n_prompt_tokens"],
                     "n_generated_tokens": meta["n_generated_tokens"],
                     "hit_max_new_tokens": meta["hit_max_new_tokens"]},
                    ensure_ascii=False) + "\n")
                jsonl.flush()
                n_done += 1
                # Qwen3.5 is a hybrid-reasoning model run with enable_thinking=false;
                # if a <think> trace leaks into the answer anyway, the classifier is
                # scoring reasoning text, not a response.
                if "<think>" in answer:
                    n_think += 1
                del rec
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                n_failed += 1
                failures.append({"prompt_id": pid, "error": "OutOfMemoryError",
                                 "msg": str(e)[:300]})
                mlog(f"    [{pid}] OOM -- skipped")
            except Exception as e:
                n_failed += 1
                failures.append({"prompt_id": pid, "error": type(e).__name__,
                                 "msg": str(e)[:300]})
                mlog(f"    [{pid}] {type(e).__name__}: {str(e)[:200]}")
                if args.strict:
                    raise
            if i % max(1, args.log_every) == 0 or i == len(todo):
                el = time.perf_counter() - t0
                mlog(f"  [{i}/{len(todo)}] {i / el * 60:.1f} prompt/min  "
                     f"eta {(len(todo) - i) / (i / el) / 60:.1f} min  failed {n_failed}")
        jsonl.close()
        elapsed = time.perf_counter() - t0

        if n_think:
            mlog(f"  !! WARNING: {n_think}/{n_done} answers contain '<think>'. C* would be "
                 f"scoring reasoning text. Set a system_prompt / chat_template_kwargs for "
                 f"{spec.slug} in configs/study.yaml and re-run this stage with --overwrite.")

        write_json({
            "stage": "generate",
            "model": {"name": spec.name, "hf_id": spec.hf_id, "slug": spec.slug,
                      "dtype": spec.dtype,
                      "attn_implementation": spec.attn_implementation,
                      "system_prompt": spec.system_prompt,
                      "chat_template_kwargs": spec.chat_template_kwargs,
                      "hidden_size": runner.hidden_size, "n_blocks": runner.n_blocks,
                      "n_layer_rows": runner.n_blocks + 1,
                      "steerable_rows": [1, runner.n_blocks - 1]},
            "generation": runner.gen_kwargs,
            "capture": cfg["capture"],
            "output": {"hidden_layout": layout, "store_dtype": store_dtype,
                       "compress": compress},
            "prompts": {"n_total": int(len(prompts)), "n_run": n_done,
                        "n_skipped_existing": n_skip, "n_failed": n_failed,
                        "sha256": read_json(paths["prepare"])["sha256"]
                        if paths["prepare"].exists() else None},
            "answers_with_think_trace": n_think,
            "failures": failures,
            "wall_seconds": round(elapsed, 1),
            "env": env_info(),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
        }, root / "manifest.json")
        mlog(f"  done: {n_done} ok, {n_failed} failed, {n_skip} skipped in "
             f"{elapsed / 60:.1f} min -> {root}")
        mlog.close()

        del runner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()


# --------------------------------------------------------------------------- #
# the classifier (C*), imported ONCE per process
# --------------------------------------------------------------------------- #
def import_classifier(cfg, log):
    """Make refusal_classifier's src.* importable ALONGSIDE this repo's src.*.

    Both repos have a top-level package called `src`. This script has already imported
    THIS repo's src.generate, so `src` is in sys.modules pointing at refusal_generation/src,
    and predict.py's own `from src import data as D` would fail with ModuleNotFoundError.
    Extending the existing package's __path__ merges the two directories into one
    namespace; the module names do not overlap (generate/steer here, data/metrics/train
    there) and that is asserted, loudly, rather than assumed.
    """
    clf_root = G.repo_path(cfg["study"]["classifier_repo"]).resolve()
    if not (clf_root / "scripts" / "predict.py").exists():
        raise SystemExit(f"classifier repo not found at {clf_root} "
                         f"(study.classifier_repo in the config)")
    import src as gen_pkg
    here = {p.stem for p in Path(gen_pkg.__path__[0]).glob("*.py")} - {"__init__"}
    there = {p.stem for p in (clf_root / "src").glob("*.py")} - {"__init__"}
    clash = here & there
    if clash:
        raise RuntimeError(f"both repos define src/{sorted(clash)}.py; merging the two "
                           f"`src` packages would shadow one of them")
    if str(clf_root / "src") not in gen_pkg.__path__:
        gen_pkg.__path__.append(str(clf_root / "src"))
    if str(clf_root) not in sys.path:
        sys.path.insert(0, str(clf_root))

    spec = importlib.util.spec_from_file_location(
        "cstar_predict", clf_root / "scripts" / "predict.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cstar_predict"] = mod
    spec.loader.exec_module(mod)
    log(f"  classifier repo {clf_root}")
    return clf_root, mod


class CStar:
    """The frozen Longformer classifier, loaded once, with the val-parity gate run once.

    Never shell out per condition: 200 conditions x 3 models x a 567MiB checkpoint reload
    is ~5 hours of pure process overhead.
    """

    def __init__(self, cfg, args, log):
        self.log = log
        self.clf_root, self.mod = import_classifier(cfg, log)
        self.M = self.mod.M
        labels_cfg = list(cfg["study"]["labels3"])
        if list(self.M.LABELS3) != labels_cfg:
            raise RuntimeError(f"config study.labels3={labels_cfg} but the classifier's "
                               f"M.LABELS3={list(self.M.LABELS3)}; the label order must "
                               f"match or every argmax is mislabelled")
        self.labels = list(self.M.LABELS3)
        self.min_n = int(self.M.MIN_N)

        run_dir = G.repo_path(cfg["study"]["classifier_run_dir"]).resolve()
        if not run_dir.exists():
            raise SystemExit(f"C* run dir not found: {run_dir}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # predict.py's DEFAULT_RUNS lists modernbert first; C* is the Longformer ONLY, so
        # the run dir is always passed explicitly.
        log(f"  C* run dir {run_dir.name}  device={self.device}")
        self.cfg_c, self.tok, self.model, self.amp = self.mod.load_classifier(
            run_dir, self.device, log)
        self.candidate = self.cfg_c["candidate"]
        self.run_dir = run_dir
        self.token_budget = int(args.token_budget)
        self.val_parity = self._val_parity(args)

    def _val_parity(self, args) -> dict:
        """Re-predict val and compare against the probabilities saved at training time.

        This proves the checkpoint, tokenizer and truncation were restored exactly. It is
        the gate the whole scoring stage hangs on, so it runs at startup and aborts.
        """
        if args.skip_val_parity:
            self.log("  !! WARNING: --skip-val-parity: the checkpoint was NOT verified "
                     "against its saved val_probs.npy. These labels are not gated and "
                     "must not be used for any reported number.")
            return {"skipped": True}
        val = pd.read_parquet(self.mod.D.repo_path("data/processed") / "val.parquet")
        t0 = time.perf_counter()
        pv, _ = self.mod.predict(val["response_text"].tolist(), self.cfg_c, self.tok,
                                 self.model, self.device, self.amp, self.token_budget)
        saved = np.load(self.run_dir / "val_probs.npy")[:len(val)]
        agree = float((pv.argmax(1) == saved.argmax(1)).mean())
        rec = {"skipped": False, "n": int(len(val)), "pred_agreement": agree,
               "max_abs_prob_diff": float(np.abs(pv - saved).max()),
               "seconds": round(time.perf_counter() - t0, 1)}
        self.log(f"  val parity: agreement={agree:.4f} "
                 f"max|dp|={rec['max_abs_prob_diff']:.4f} ({rec['seconds']}s)")
        if agree < self.mod.MIN_VAL_AGREEMENT:
            raise RuntimeError(
                f"C* val parity {agree:.2%} < {self.mod.MIN_VAL_AGREEMENT:.0%}: the "
                f"checkpoint or tokenisation was not restored correctly. Every label "
                f"this process would emit is suspect.")
        return rec

    def label(self, texts) -> pd.DataFrame:
        """argmax label + probabilities, in input order.

        Metrics downstream use the ARGMAX LABEL ONLY -- dynamic batch composition shifts
        the probabilities by up to 0.023, which would leak into any probability-based
        statistic. The probability columns are kept for inspection, never for a headline.
        """
        probs, n_trunc = self.mod.predict(list(texts), self.cfg_c, self.tok, self.model,
                                          self.device, self.amp, self.token_budget)
        out = self.mod.prob_columns(CSTAR_PREFIX, probs)
        out["cstar_truncated_at_max_length"] = False
        self._last_truncated = n_trunc
        return out


def score_frame(cstar: CStar, df: pd.DataFrame, shard_rows: int, log) -> pd.DataFrame:
    """Score a long frame in shards, so a 58k-row steered sweep never materialises at once."""
    parts, n = [], len(df)
    for start in range(0, n, shard_rows):
        chunk = df.iloc[start:start + shard_rows]
        t0 = time.perf_counter()
        labs = cstar.label(chunk["answer"].fillna("").astype(str).tolist())
        labs.index = chunk.index
        parts.append(pd.concat([chunk, labs], axis=1))
        log(f"    scored {min(start + shard_rows, n)}/{n} rows "
            f"({time.perf_counter() - t0:.1f}s)")
    return pd.concat(parts, ignore_index=True) if parts else df.assign()


def read_shards(d: Path) -> pd.DataFrame:
    files = sorted(Path(d).glob("shard_*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def next_shard(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    return d / f"shard_{len(list(d.glob('shard_*.parquet'))):04d}.parquet"


# --------------------------------------------------------------------------- #
# stage 3: score (the batch-1 baseline answers)
# --------------------------------------------------------------------------- #
def stage_score(cfg, args, paths, specs, log) -> None:
    log("\n=== stage: score (GPU, classifier env) -- batch-1 baseline answers ===")
    todo = []
    for spec in specs:
        f = paths["generated"] / spec.slug / "answers.jsonl"
        if not f.exists():
            log(f"  [skip] {spec.slug}: no {f} -- run --stage generate first")
            continue
        out_dir = paths["scores"] / spec.slug / "baseline_capture"
        if read_shards(out_dir).shape[0] and not args.overwrite:
            log(f"  [skip] {spec.slug}: already scored ({out_dir})")
            continue
        todo.append((spec, f, out_dir))
    if not todo:
        log("  nothing to do")
        return
    if args.dry_run:
        for spec, f, _ in todo:
            log(f"  would score {spec.slug}: {sum(1 for _ in open(f, encoding='utf-8'))} rows")
        return

    # Loaded only on the real path: the topic / is_sensitive stratification comes from
    # prompts.parquet, which a dry-run must not require.
    meta = load_prompts(paths, args.limit)[["prompt_id", "topic", "is_sensitive"]]
    cstar = CStar(cfg, args, log)
    shard_rows = int(cfg["steering"]["scoring"].get("shard_rows", 20000))
    summary = {}
    for spec, f, out_dir in todo:
        df = pd.read_json(f, lines=True, dtype={"prompt_id": "int64"})
        df = df.drop_duplicates(subset=["prompt_id"], keep="last")
        df = df.merge(meta, on="prompt_id", how="left", suffixes=("", "_meta"))
        df["gen_model"] = spec.slug
        df["condition"] = "capture_batch1"
        df["empty_answer"] = df["answer"].fillna("").astype(str).str.strip() == ""
        log(f"  {spec.slug}: {len(df)} answers")
        scored = score_frame(cstar, df, shard_rows, log)
        if out_dir.exists() and args.overwrite:
            for old in out_dir.glob("shard_*.parquet"):
                old.unlink()
        scored.to_parquet(next_shard(out_dir), index=False)
        summary[spec.slug] = baseline_report(scored, cstar, log)
        log(f"  wrote {out_dir}")

    write_json({"stage": "score", "candidate": cstar.candidate,
                "run_dir": cstar.run_dir.name, "val_parity": cstar.val_parity,
                "metric_basis": cfg["steering"]["scoring"]["metric_basis"],
                "models": summary, "env": env_info(),
                "written_utc": datetime.now(timezone.utc).isoformat()},
               paths["scores"] / "score_baseline_capture.json")


def baseline_report(scored: pd.DataFrame, cstar: CStar, log) -> dict:
    """Label rates for the sensitive rows and the benign control rows, SEPARATELY.

    The benign set is an over-disclaimering / false-positive check, not part of the
    transfer set. Pooling all 290 as "OOD" would average a real effect against a control.
    """
    col = f"{CSTAR_PREFIX}_label"
    out = {}
    for name, sub in (("sensitive", scored[scored.is_sensitive.fillna(False)]),
                      ("benign_control", scored[~scored.is_sensitive.fillna(False)]),
                      ("all", scored)):
        n = len(sub)
        rates = {lab: round(100.0 * float((sub[col] == lab).mean()), 2) if n else None
                 for lab in cstar.labels}
        out[name] = {"n": int(n), "pct": rates,
                     "evidence": "ok" if n >= cstar.min_n else "insufficient_evidence"}
        log(f"    {name:15s} n={n:4d}  " +
            "  ".join(f"{lab}={rates[lab]}%" for lab in cstar.labels))
    ref = out["sensitive"]["pct"].get("REFUSAL")
    if ref is not None and ref <= FLOOR_RATE_PCT:
        log(f"    !! baseline REFUSAL is {ref}% on the sensitive rows: the d_R arm is "
            f"STRUCTURALLY FLOORED on this dataset (plan.md 1.13). A flat d_R curve here "
            f"is a property of the prompts, not a null result about the direction.")
    return out


# --------------------------------------------------------------------------- #
# directions, layers, scaling constants
# --------------------------------------------------------------------------- #
_LAYER_KEY = re.compile(r"^(?P<name>[A-Za-z][A-Za-z0-9]*?)[_-]?(?:L|layer[_-]?)?"
                        r"(?P<layer>\d+)$")


def load_directions(path: Path, names, hidden_size: int, log) -> tuple[dict, dict]:
    """Frozen in-domain d_hat per (direction, layer), from step 2's committed npz.

    Two storage conventions are accepted, because this script must not dictate step 2's
    file layout:
      A. one [n_rows, H] array per direction name, plus a 'layers' index array
      B. one flat key per (name, layer), e.g. 'dR_L024' / 'dD_layer_24' / 'dR_24'
    Anything else raises with the actual key list rather than guessing.

    Vectors are re-normalised through S.unit(): step 2 should already store unit vectors,
    so the norm is logged and a drift from 1.0 is visible.
    """
    if not Path(path).exists():
        raise SystemExit(
            f"frozen directions not found: {path}\n"
            f"Step 4 does NOT estimate directions. d_hat is frozen from the in-domain "
            f"Direction split by step2_directions_steering.py; run that first.")
    z = np.load(path)
    keys = list(z.files)
    vecs: dict[tuple[str, int], np.ndarray] = {}
    sigmas: dict[tuple[str, int], float] = {}

    layer_index = None
    for cand in ("layers", "layer_rows", "layer_index"):
        if cand in keys:
            layer_index = np.asarray(z[cand]).ravel().astype(int)
            break

    for nm in names:
        if nm in keys and np.asarray(z[nm]).ndim == 2:          # convention A
            arr = np.asarray(z[nm])
            if layer_index is None or len(layer_index) != arr.shape[0]:
                raise RuntimeError(
                    f"{path}: '{nm}' is [{arr.shape[0]}, {arr.shape[1]}] but there is no "
                    f"matching layer index array (tried 'layers'/'layer_rows'/"
                    f"'layer_index'); keys present: {keys}")
            for i, l in enumerate(layer_index):
                vecs[(nm, int(l))] = arr[i]
            sig_key = next((k for k in (f"sigma_{nm}", f"{nm}_sigma") if k in keys), None)
            if sig_key is not None:
                sa = np.asarray(z[sig_key]).ravel()
                if len(sa) == len(layer_index):
                    for i, l in enumerate(layer_index):
                        sigmas[(nm, int(l))] = float(sa[i])

    if not vecs:                                                 # convention B
        for k in keys:
            m = _LAYER_KEY.match(k)
            if not m:
                continue
            base, l = m.group("name"), int(m.group("layer"))
            if base.startswith("sigma"):
                continue
            if base in names and np.asarray(z[k]).ndim == 1:
                vecs[(base, l)] = np.asarray(z[k])
        for k in keys:
            m = _LAYER_KEY.match(k)
            if m and m.group("name").startswith("sigma"):
                base = m.group("name")[5:].lstrip("_")
                if base in names:
                    sigmas[(base, int(m.group("layer")))] = float(np.asarray(z[k]).ravel()[0])

    if not vecs:
        raise RuntimeError(
            f"{path}: no direction vectors found for {list(names)}. Keys present: {keys}. "
            f"Understood layouts: (A) one [n_rows, H] array per direction plus a 'layers' "
            f"index, (B) one flat key per (direction, layer) such as 'dR_L024'.")

    out = {}
    for (nm, l), v in sorted(vecs.items()):
        v = np.asarray(v, dtype=np.float64).ravel()
        if v.size != hidden_size:
            raise RuntimeError(f"{path}: direction {nm} layer {l} has {v.size} dims, "
                               f"model hidden size is {hidden_size}")
        norm = float(np.linalg.norm(v))
        if abs(norm - 1.0) > 1e-3:
            log(f"  !! {nm} layer {l}: stored norm {norm:.6f} != 1; re-normalising")
        out[(nm, l)] = S.unit(v)
    log(f"  frozen directions: {len(out)} vectors over "
        f"{sorted({nm for nm, _ in out})} x {len(sorted({l for _, l in out}))} layers "
        f"<- {path.name}")
    return out, sigmas


def rank_selected_layers(cfg, paths, slug: str, top_n: int | None, override,
                         log) -> tuple[list, str]:
    """This model's OWN top layers, ranked by the probe score.

    The PDF says "same layer_ids as selected on chatbot", but block counts differ across
    models so an absolute row id means a different relative depth in each. Every artifact
    this script writes therefore records the absolute row AND l / n_blocks, and says that
    the per-model list -- not a shared id list -- is what was used.
    """
    if override:
        picked = sorted({int(x) for x in str(override).split(",") if str(x).strip()})
        log(f"  layers from --layers: {picked}")
        return picked, "cli_override"

    sel_path = paths["study"] / "selected_layers.json"
    if not sel_path.exists():
        raise SystemExit(f"{sel_path} is missing -- run the layer-selection step first, "
                         f"or pass --layers explicitly")
    sel_doc = read_json(sel_path)
    # layer_probe.py's superseded file (no 'criterion', probe_bacc x reliability, old
    # roster, old split) parses fine here -- steering OOD on those layers would silently
    # answer a different question than the in-domain run.
    pcfg = cfg.get("probe", {}) or {}
    try:
        S.assert_selected_layers_current(
            sel_doc, criterion=pcfg.get("criterion", "mean_auroc_R_vs_rest_D_vs_rest"),
            top_k=int(pcfg.get("top_k", 10)), study_sha=None, slugs=(slug,),
            path=str(sel_path))
    except S.StaleArtifactError as e:
        raise SystemExit(f"!! {e}")
    selected = sel_doc["selected"]
    chosen = [int(l) for l in selected[slug]]
    # top_n None (layers_top 'all') means every selected layer for this model, and a
    # top_n above the selection size is clamped to it (layers_top 10 = "all of them").
    # Resolved HERE, not at the call site, so a missing selected_layers.json still gets
    # the clean message above rather than a raw KeyError from the caller's expression.
    top_n = len(chosen) if top_n is None else min(int(top_n), len(chosen))

    # step 2's probe stage writes step2_probe.json. layer_probe.json is the
    # SUPERSEDED name from the old layer_probe.py script; accept it only as a fallback.
    probe_path = next((paths["study"] / n for n in ("step2_probe.json", "layer_probe.json")
                       if (paths["study"] / n).exists()),
                      paths["study"] / "step2_probe.json")
    if probe_path.exists():
        rep = read_json(probe_path).get("models", {}).get(slug, {})
        scores = {int(r["layer_row"]): r.get("score") for r in rep.get("layers", [])
                  if r.get("score") is not None}
        if all(l in scores for l in chosen):
            ranked = sorted(chosen, key=lambda l: (-scores[l], l))
            picked = sorted(ranked[:top_n])
            log(f"  layers: top {top_n} of {len(chosen)} selected, ranked by probe score "
                f"-> {picked}  (full ranking {ranked})")
            return picked, "probe_score"
    raise SystemExit(
        f"cannot rank {slug}'s selected layers: {probe_path} has no per-layer score for "
        f"all of {chosen}. Taking an arbitrary {top_n} of them would silently change what "
        f"'top' means. Pass --layers <comma list> to choose explicitly.")


def load_activations(model_root: Path, prompt_ids, layers, point=CAPTURE_POINT):
    """{layer: [n, H] float64} from the stacked npz files, reading each prompt ONCE.

    G.load_layer_matrix re-opens every prompt file per layer; at 3-10 layers that is a
    3-10x re-read of the same archives for no reason.
    """
    hid = Path(model_root) / "hidden"
    mats = {int(l): [] for l in layers}
    kept = []
    for pid in prompt_ids:
        pid = int(pid)
        tgt = hid / f"{pid}.npz"
        if not tgt.exists():
            tgt = hid / str(pid)
        if not tgt.exists():
            raise FileNotFoundError(
                f"no stored activations for prompt {pid} under {hid}; the generate stage "
                f"did not finish for this model")
        arr, ls = G.load_hidden(tgt, point, None)
        ix = {int(v): k for k, v in enumerate(np.asarray(ls).ravel())}
        missing = [l for l in mats if l not in ix]
        if missing:
            raise RuntimeError(f"prompt {pid}: layer row(s) {missing} absent from {tgt}")
        for l in mats:
            mats[l].append(arr[ix[l]])
        kept.append(pid)
    return {l: np.stack(v).astype(np.float64) for l, v in mats.items()}, kept


def scaling_constants(X, d) -> dict:
    """sigma_l = sd(h @ d_hat); r_l = median ||h|| (plan.md 1.4 / 2c)."""
    return {"sigma": S.sigma_along(X, d),
            "r": float(np.median(np.linalg.norm(X, axis=1))),
            "n": int(X.shape[0])}


# --------------------------------------------------------------------------- #
# stage 4: steer
# --------------------------------------------------------------------------- #
def resolve_ood_grid(cfg, reduced: bool) -> tuple[int | None, list[float]]:
    """(layers_top, non-zero alphas) for the default grid, or for ood.reduced.

    The ONE place the grid is read, so build_grid, stage_steer, the dry-run inventory and
    apply_smoke cannot disagree on what "the grid" is. Schema:

      ood.layers_top  int top-N of the model's own selected layers (None / 'all' = all)
      ood.alphas      'inherit' (= steering.alphas) or an explicit list
      ood.reduced     {layers_top, alphas} with the same meaning, used by --reduced

    A config still carrying the old flat schema (layers_top: 3, alphas: [..], no
    `reduced` block) keeps working: its flat values ARE the default grid, and --reduced
    refuses loudly rather than guessing what a reduced grid should be.

    alpha 0 is never a grid cell -- it is the ONE shared batched baseline, prepended by
    stage_steer -- so it is dropped here. A duplicate alpha raises: it would produce the
    same condition tag twice and the second would silently SKIP as "already complete".
    """
    o = cfg["ood"]
    if reduced:
        src, where = o.get("reduced"), "ood.reduced"
        if not isinstance(src, dict):
            raise SystemExit(f"--reduced needs an {where} block {{layers_top, alphas}} in "
                             f"the config; this config has none (old flat ood schema, "
                             f"whose flat values are already its default grid)")
    else:
        src, where = o, "ood"

    lt = src.get("layers_top")
    if lt is None or str(lt).strip().lower() == "all":
        layers_top = None
    else:
        layers_top = int(lt)
        if layers_top < 1:
            raise SystemExit(f"{where}.layers_top={lt!r}; must be >= 1 (or 'all')")

    raw = src.get("alphas")
    if isinstance(raw, str):
        if raw.strip().lower() != "inherit":
            raise SystemExit(f"{where}.alphas={raw!r}; the only string value is 'inherit' "
                             f"(= steering.alphas)")
        raw = cfg["steering"]["alphas"]
    if not isinstance(raw, (list, tuple)):
        raise SystemExit(f"{where}.alphas must be 'inherit' or a list, got {raw!r}")
    alphas: list[float] = []
    for a in raw:
        fa = float(a)
        if fa == 0.0:
            continue
        if fa in alphas:
            raise SystemExit(f"{where}.alphas lists {fa} twice; each alpha is one "
                             f"condition tag and must be unique")
        alphas.append(fa)
    if not alphas:
        raise RuntimeError(f"the {where} alpha grid is empty after dropping 0 (alpha=0 is "
                           f"the shared baseline, generated once)")
    return layers_top, alphas


def grid_variant(reduced: bool) -> str:
    return "reduced" if reduced else "default"


def build_grid(cfg, layers, reduced: bool) -> tuple[list, list, list]:
    _, alphas = resolve_ood_grid(cfg, reduced)
    # ood.directions, never steering.directions: d_hat and sigma are loaded for
    # ood.directions, so a grid over any other list would look up vectors never loaded.
    dirs = list(cfg["ood"]["directions"])
    cells = [(int(l), nm, a) for l in layers for nm in dirs for a in alphas]
    return cells, dirs, alphas


def parse_shard(spec) -> tuple[int, int] | None:
    """'i/n' -> (i, n), 1-based, EXACTLY step 2's --shard: 1 <= i <= n or fail."""
    if not spec:
        return None
    try:
        i, n = (int(x) for x in str(spec).split("/"))
    except ValueError:
        raise SystemExit(f"--shard {spec!r} is not of the form i/n (e.g. 3/8)")
    if not 1 <= i <= n:
        raise SystemExit(f"--shard {spec} is out of range (need 1 <= i <= n)")
    return i, n


def shard_select(conditions: list, shard: tuple[int, int] | None) -> list:
    """Condition k goes to shard (k % n) == i - 1 -- step 2's rule, verbatim.

    `conditions` is the FULL per-model list with the baseline at k=0, so the baseline
    lands in shard 1 and in no other shard.
    """
    if shard is None:
        return list(conditions)
    i, n = shard
    return [c for k, c in enumerate(conditions) if k % n == (i - 1)]


def shard_label(shard: tuple[int, int] | None) -> str:
    i, n = shard or (1, 1)
    return f"shard_{i:02d}of{n:02d}"


def batch_plan_sha(rows, batches, batch_size: int) -> str:
    """Identity of the batch plan: prompt ids, in their batches, in order.

    Batched greedy decoding is not bitwise invariant to batch composition, so the baseline
    one shard writes is only a valid subtrahend for a condition another shard writes if
    both saw EXACTLY this plan. Same payload as step 2's batch_plan_hash.
    """
    payload = {"batch_size": int(batch_size),
               "batches": [[int(rows[i].prompt_id) for i in b] for b in batches]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def stage_steer(cfg, args, paths, specs, log) -> None:
    log("\n=== stage: steer (GPU, batched, frozen in-domain d_hat) ===")
    shard = parse_shard(args.shard)
    variant = grid_variant(args.reduced)
    # Answered before load_prompts: the inventory is what you read while deciding
    # whether to fetch the dataset and book the GPU-h at all.
    if args.dry_run:
        dry_run_inventory(cfg, args, paths, specs,
                          planned_n_prompts(paths, args, log), log)
        return

    prompts = load_prompts(paths, args.limit)
    rows = list(prompts.itertuples(index=False))
    pids = [int(r.prompt_id) for r in rows]
    log(f"  prompts {len(rows)}  models {[s.slug for s in specs]}  grid={variant}  "
        f"shard={'/'.join(map(str, shard)) if shard else '1/1 (unsharded)'}")
    # Printed BEFORE anything loads: the default parity grid is ~8x the reduced one, and
    # the GPU-h should be on screen before a model is paid for.
    dry_run_inventory(cfg, args, paths, specs, len(rows), log)

    batch_size = int(args.batch_size if args.batch_size
                     else cfg["steering"].get("batch_size", 8))
    max_prompt_tokens = cfg["steering"].get("max_prompt_tokens")
    layers_top, _ = resolve_ood_grid(cfg, args.reduced)

    for spec in specs:
        slug = spec.slug
        gen_root = paths["generated"] / slug
        if not (gen_root / "hidden").is_dir():
            log(f"  [skip] {slug}: no OOD activations at {gen_root} -- run --stage generate")
            continue
        out_root = paths["steered"] / slug
        out_root.mkdir(parents=True, exist_ok=True)
        # One log per shard: eight array tasks appending to one steer.log interleave
        # their lines into something nobody can read after a failure.
        mlog = Tee(out_root / ("steer.log" if shard is None
                               else f"steer_{shard_label(shard)}.log"))
        mlog(f"\n=== {slug} | {shard_label(shard)} | "
             f"{datetime.now(timezone.utc).isoformat()} ===")

        layers, layer_source = rank_selected_layers(
            cfg, paths, slug, layers_top, args.layers, mlog)

        runner = G.Runner(spec, cfg["generation"], cfg["capture"], log=mlog)
        n_blocks = runner.n_blocks
        for l in layers:
            S.assert_steerable(l, n_blocks)

        dir_path = paths["directions"] / f"{slug}.npz"
        dhat, stored_sigma = load_directions(dir_path, list(cfg["ood"]["directions"]),
                                             runner.hidden_size, mlog)
        missing = [(nm, l) for l in layers for nm in cfg["ood"]["directions"]
                   if (nm, l) not in dhat]
        if missing:
            raise SystemExit(f"{dir_path} has no direction for {missing}; step 2 must "
                             f"cover every layer step 4 steers")

        # ---- scaling constants, recomputed on the OOD activations ---------- #
        # Every shard recomputes ALL layers, not only its own: the constants are
        # deterministic, and the full-grid manifest each shard writes must carry the
        # scaling for every layer the metrics stage will read.
        mlog(f"  recomputing sigma_l / r_l on {len(pids)} OOD prompts, layers {layers}")
        Xood, kept = load_activations(gen_root, pids, layers)
        if len(kept) != len(pids):
            raise RuntimeError(f"activations for {len(pids) - len(kept)} prompts missing")
        scale = {}
        indomain = indomain_sigma(cfg, paths, slug, layers, dhat, stored_sigma, mlog)
        mlog(f"  {'layer':>6} {'depth':>6} {'dir':>4} {'sigma_ood':>10} {'sigma_ind':>10} "
             f"{'ratio':>7} {'r_ood':>9}")
        sigma_flags = []
        for l in layers:
            for nm in cfg["ood"]["directions"]:
                d = dhat[(nm, l)]
                c = scaling_constants(Xood[l], d)
                sind = indomain.get((nm, l))
                ratio = (c["sigma"] / sind) if (sind and sind > 0) else None
                scale[(nm, l)] = {**c, "sigma_indomain": sind, "sigma_ratio": ratio}
                if ratio is not None and (ratio > SIGMA_RATIO_WARN
                                          or ratio < 1.0 / SIGMA_RATIO_WARN):
                    sigma_flags.append({"direction": nm, "layer": int(l),
                                        "ratio": round(ratio, 3)})
                mlog(f"  {l:>6} {l / n_blocks:>6.2f} {nm:>4} {c['sigma']:>10.3f} "
                     f"{'--' if sind is None else f'{sind:>10.3f}'} "
                     f"{'--' if ratio is None else f'{ratio:>7.3f}'} {c['r']:>9.2f}")
        if sigma_flags:
            mlog(f"  !! sigma shifted by more than {SIGMA_RATIO_WARN}x between in-domain "
                 f"and OOD at {sigma_flags}. alpha/sigma_l is then NOT comparable across "
                 f"the two datasets and the paper must say so.")

        # ---- the gate: the hook sits where the capture was taken ----------- #
        preflight = {"skipped": True}
        if not args.skip_preflight:
            preflight = S.preflight_hook_matches_capture(
                runner, rows[:args.preflight_prompts], gen_root, log=mlog)
        else:
            mlog("  !! WARNING: --skip-preflight: the hook/capture identity was NOT "
                 "verified on this model.")

        # ---- one batch plan, reused by EVERY condition including baseline -- #
        # Planned over ALL prompts in EVERY shard, never over a shard's share: the plan
        # depends only on prompts, tokenizer and batch size, so all shards derive the
        # identical plan, and its hash below proves that after the fact.
        bg = S.BatchedGenerator(runner, batch_size=batch_size,
                                max_prompt_tokens=max_prompt_tokens, log=mlog)
        batches = bg.plan_batches(rows)
        plan_sha = batch_plan_sha(rows, batches, bg.batch_size)

        cells, dirs, alphas = build_grid(cfg, layers, args.reduced)
        conditions = [(S.BASELINE_TAG, None, None, 0.0)] + [
            (S.condition_tag(l, nm, a), l, nm, a) for (l, nm, a) in cells]
        grid_sha = sha256_lines(t for t, _, _, _ in conditions)
        mine = shard_select(conditions, shard)
        has_base = any(S.is_baseline(c[0]) for c in mine)
        mlog(f"  grid ({variant}): {len(layers)} layers x {len(dirs)} directions x "
             f"{len(alphas)} alphas + 1 baseline = {len(conditions)} conditions x "
             f"{len(rows)} prompts = {len(conditions) * len(rows):,} generations")
        mlog(f"  {shard_label(shard)}: {len(mine)} of {len(conditions)} conditions "
             f"({len(mine) * len(rows):,} generations)"
             f"{'  -- includes the shared baseline' if has_base else ''}")
        mlog(f"  batch plan sha256 {plan_sha[:16]}...  grid sha256 {grid_sha[:16]}...")

        # ---- the cross-shard invariant, checked BEFORE generating ---------- #
        # Another shard (or an earlier run) may already have written conditions under a
        # different batch plan (other --batch-size, --limit, tokenizer). Mixing them would
        # subtract a baseline from rows it was not batched with -- fail instead.
        man_path = out_root / "manifest.json"
        if man_path.exists():
            prev = read_json(man_path)
            prev_plan = (prev.get("batching") or {}).get("batch_plan_sha256")
            if prev_plan and prev_plan != plan_sha and not args.overwrite:
                raise SystemExit(
                    f"{slug}: {man_path} records batch plan {prev_plan[:16]}..., this "
                    f"shard planned {plan_sha[:16]}... (batch size / --limit / tokenizer "
                    f"differ). Conditions from the two plans are not comparable. Re-run "
                    f"EVERY shard with --overwrite, or restore the original settings.")
            prev_grid = (prev.get("grid") or {}).get("sha256")
            if prev_grid and prev_grid != grid_sha:
                mlog(f"  !! {man_path} described grid {prev_grid[:16]}... "
                     f"({(prev.get('grid') or {}).get('variant', 'legacy')}); it is being "
                     f"replaced by this {variant} grid {grid_sha[:16]}.... Condition dirs "
                     f"are keyed by tag, so a tag in both grids is reused, not rerun.")

        t0 = time.perf_counter()
        n_run = 0
        by_pid = {int(r.prompt_id): r for r in rows}
        for ci, (tag, l, nm, a) in enumerate(mine, 1):
            dest = out_root / tag / "answers.jsonl"
            done = S.read_done_ids(dest)
            if done >= set(pids) and not args.overwrite:
                mlog(f"  [{ci}/{len(mine)}] {tag}: already complete, skipping")
                continue
            if args.overwrite and dest.exists():
                dest.unlink()
            todo_rows = [r for r in rows if int(r.prompt_id) not in done] \
                if done else rows
            if todo_rows is not rows:
                # A partially-written condition is REGENERATED in full: a resumed subset
                # would form different batches than the plan, so its rows would not be
                # bitwise comparable with the same rows under another alpha.
                mlog(f"  [{ci}/{len(mine)}] {tag}: {len(done)} partial rows found; "
                     f"regenerating the whole condition to keep batch membership static")
                if dest.exists():
                    dest.unlink()

            if l is None:
                hook_ctx = None                      # the shared alpha=0 baseline
                extra = {"condition": tag, "layer": None, "direction": "baseline",
                         "alpha": 0.0, "alpha_over_sigma_ood": 0.0, "sigma_ood": None,
                         "splits": OOD_SPLIT_TAG, "gen_model": slug}
            else:
                vec = dhat[(nm, l)]
                sig = scale[(nm, l)]["sigma"]
                hook_ctx = (lambda _l=l, _v=vec, _a=a:
                            S.steering_hook(runner.model, _l, _v, _a, tok=runner.tok,
                                            n_blocks=n_blocks))
                extra = {"condition": tag, "layer": int(l), "direction": nm,
                         "alpha": float(a),
                         "alpha_over_sigma_ood": float(a) / sig if sig else None,
                         "sigma_ood": float(sig), "splits": OOD_SPLIT_TAG,
                         "gen_model": slug}

            mlog(f"  [{ci}/{len(mine)}] {tag}")
            recs = bg.generate(rows, batches, hook_ctx=hook_ctx, extra=extra)
            for rec in recs:
                src = by_pid[rec["prompt_id"]]
                rec["topic"] = src.topic
                rec["is_sensitive"] = bool(src.is_sensitive)
                rec["ood_hex_id"] = src.ood_hex_id
            S.write_answers_jsonl(dest, recs, mode="a")
            n_run += 1

        elapsed = time.perf_counter() - t0
        n_layers_selected = None
        if layer_source != "cli_override":
            n_layers_selected = len(read_json(paths["study"] / "selected_layers.json")
                                    ["selected"][slug])
        # Computed, not assumed: an old flat config, a --layers override or a smaller
        # layers_top can each make the "default" grid something less than parity.
        full_parity = (not args.reduced
                       and alphas == [float(x) for x in cfg["steering"]["alphas"]
                                      if float(x) != 0.0]
                       and n_layers_selected is not None
                       and len(layers) == n_layers_selected)

        # Per-shard bookkeeping: everything that legitimately differs between array tasks.
        write_json({
            "stage": "steer",
            "shard": "/".join(map(str, shard)) if shard else "1/1",
            "model": slug,
            "grid_sha256": grid_sha,
            "batch_plan_sha256": plan_sha,
            "conditions_assigned": [t for t, _, _, _ in mine],
            "conditions_run_this_invocation": n_run,
            "batching": {"requested": bg.requested_batch_size, "effective": bg.batch_size,
                         "padded_repeats": bg.n_padded_repeats,
                         "oom_halvings": bg.n_oom_halvings,
                         "static_membership": bg.n_oom_halvings == 0},
            "preflight_hook_matches_capture": preflight,
            "wall_seconds": round(elapsed, 1),
            "env": env_info(),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
        }, out_root / "shards" / f"{shard_label(shard)}.json")

        # The FULL-grid manifest: identical in substance whichever shard writes it last
        # (same layers, grid, scaling, batch plan), and replaced atomically.
        write_json_atomic({
            "stage": "steer",
            "seed": int(cfg["study"]["seed"]),
            "model": {"slug": slug, "hf_id": spec.hf_id, "n_blocks": n_blocks,
                      "hidden_size": runner.hidden_size,
                      "attn_implementation": spec.attn_implementation},
            "inputs": {
                "prompts_sha256": read_json(paths["prepare"])["sha256"]
                if paths["prepare"].exists() else None,
                "directions": str(dir_path),
                "directions_frozen_from": "in-domain Direction split (step 2); step 4 "
                                          "recomputes ONLY sigma_l and r_l",
                "selected_layers": str(paths["study"] / "selected_layers.json"),
                "layer_ranking_source": layer_source,
            },
            "layers": {"rows": [int(l) for l in layers],
                       "relative_depth": [round(l / n_blocks, 4) for l in layers],
                       "note": "each model uses ITS OWN selected layers; absolute row ids "
                               "are not comparable across models with different block "
                               "counts, so relative depth is reported alongside"},
            "grid": {"variant": variant, "full_parity": bool(full_parity),
                     "layers_top": layers_top, "directions": dirs,
                     "alphas": alphas, "n_conditions": len(conditions),
                     "n_prompts": len(rows),
                     "n_generations": len(conditions) * len(rows),
                     "baseline_tag": S.BASELINE_TAG,
                     "sha256": grid_sha},
            "sharding": {"n_shards": shard[1] if shard else 1,
                         "rule": "condition k (baseline is k=0) -> shard (k % n) + 1",
                         "per_shard_records": "shards/shard_<i>of<n>.json"},
            "scaling": {f"{nm}_L{l:03d}": {k: (None if v is None else float(v))
                                           for k, v in s.items()}
                        for (nm, l), s in scale.items()},
            "sigma_shift_flags": sigma_flags,
            "batching": {"requested": bg.requested_batch_size, "effective": bg.batch_size,
                         "max_prompt_tokens": bg.max_prompt_tokens,
                         "n_batches": len(batches),
                         "batch_plan_sha256": plan_sha,
                         "note": "OOM halvings, preflight and runtime are per shard: "
                                 "see shards/"},
            "env": env_info(),
            "written_utc": datetime.now(timezone.utc).isoformat(),
        }, man_path)
        mlog(f"  {n_run} condition(s) generated in {elapsed / 60:.1f} min -> {out_root}")
        mlog(f"  grid sha256 {grid_sha[:16]}...")
        mlog.close()

        del runner, bg
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()


def indomain_sigma(cfg, paths, slug, layers, dhat, stored_sigma, log) -> dict:
    """sigma_l measured IN DOMAIN, for the OOD/in-domain ratio.

    Preferred source is whatever step 2 stored next to the direction. Failing that it is
    recomputed from the in-domain Direction-split activations, which is the definition
    (sd of the Direction-split activations projected onto the direction). If neither is
    available the ratio is reported as null -- never silently replaced by the OOD value.
    """
    out = {}
    for (nm, l), v in stored_sigma.items():
        if l in layers:
            out[(nm, l)] = float(v)
    need = [(nm, l) for l in layers for nm in cfg["ood"]["directions"] if (nm, l) not in out]
    if not need:
        return out

    split_path = paths["study"] / "study_splits.json"
    root = paths["generated_indomain"] / slug
    if not split_path.exists() or not (root / "hidden").is_dir():
        log(f"  in-domain sigma unavailable for {need} (need {split_path} and "
            f"{root}/hidden); the sigma ratio will be reported as null")
        return out
    split_doc = read_json(split_path)
    try:
        S.assert_study_splits_current(
            split_doc, [s.slug for s in G.parse_models(cfg)], path=str(split_path),
            partition=str((cfg.get("study") or {}).get("partition", "shared")),
            fractions=S.split_fractions(cfg))
    except (S.StaleArtifactError, ValueError) as e:
        raise SystemExit(f"!! {e}")
    ids = split_doc["splits"]["direction"]
    try:
        X, _ = load_activations(root, ids, sorted({l for _, l in need}))
    except Exception as e:
        log(f"  in-domain sigma recomputation failed ({type(e).__name__}: {e}); "
            f"the sigma ratio will be reported as null")
        return out
    for nm, l in need:
        out[(nm, l)] = S.sigma_along(X[l], dhat[(nm, l)])
    log(f"  in-domain sigma recomputed on the Direction split (n={len(ids)})")
    return out


# --------------------------------------------------------------------------- #
# stage 5: score-steered
# --------------------------------------------------------------------------- #
def stage_score_steered(cfg, args, paths, specs, log) -> None:
    log("\n=== stage: score-steered (GPU, classifier env) ===")
    shard_rows = int(cfg["steering"]["scoring"].get("shard_rows", 20000))

    plan = []
    for spec in specs:
        root = paths["steered"] / spec.slug
        if not root.is_dir():
            log(f"  [skip] {spec.slug}: nothing at {root} -- run --stage steer first")
            continue
        out_dir = paths["scores"] / spec.slug / "steered"
        existing = read_shards(out_dir)
        if args.overwrite:
            existing = existing.iloc[0:0]
        # Resume per (condition, prompt_id), NOT per condition. The steer stage appends
        # answers.jsonl batch by batch, and 8 shards run concurrently, so a condition can
        # be half-written when this stage runs. Marking a condition "scored" because SOME
        # of its rows were scored used to strand the rest forever: every rerun skipped it,
        # and metrics then refused the condition (its prompt ids no longer matched the
        # baseline), with --overwrite of the whole model as the only way out.
        scored_ids: dict[str, set] = {}
        if len(existing):
            for cond, g in existing.groupby("condition"):
                scored_ids[str(cond)] = set(int(x) for x in g["prompt_id"])
        man_path = root / "manifest.json"
        n_expect = (read_json(man_path).get("grid", {}).get("n_prompts")
                    if man_path.exists() else None)
        todo, waiting, n_have = [], [], 0
        for d in sorted(p for p in root.iterdir() if (p / "answers.jsonl").exists()):
            written = S.read_done_ids(d / "answers.jsonl")
            if n_expect is not None and len(written) < int(n_expect):
                # still being generated by a shard (or that shard died): score it on a
                # later run, once it is complete, rather than scoring a fragment now
                waiting.append(f"{d.name} {len(written)}/{int(n_expect)}")
                continue
            missing = written - scored_ids.get(d.name, set())
            if not missing:
                n_have += 1
                continue
            todo.append((d, missing))
        if waiting:
            log(f"  [wait] {spec.slug}: {len(waiting)} condition(s) still being written, "
                f"not scored yet -- rerun this stage after the steer array finishes: "
                f"{', '.join(waiting[:6])}{' ...' if len(waiting) > 6 else ''}")
        if not todo:
            log(f"  [skip] {spec.slug}: all {n_have} complete condition(s) already scored")
            continue
        plan.append((spec, todo, out_dir, n_have))
    if not plan:
        log("  nothing to do")
        return
    if args.dry_run:
        for spec, todo, _, n_have in plan:
            log(f"  would score {spec.slug}: {len(todo)} condition(s) "
                f"({n_have} already done)")
        return

    cstar = CStar(cfg, args, log)
    summary = {}
    for spec, todo, out_dir, n_have in plan:
        if args.overwrite and out_dir.exists():
            for old in out_dir.glob("shard_*.parquet"):
                old.unlink()
        log(f"  {spec.slug}: {len(todo)} condition(s) to score "
            f"({n_have} already present)")
        buf, n_cond = [], 0
        last_dir = todo[-1][0]
        for d, missing in todo:
            df = pd.read_json(d / "answers.jsonl", lines=True,
                              dtype={"prompt_id": "int64"})
            df = df.drop_duplicates(subset=["prompt_id"], keep="last")
            # only the prompts not already scored for this condition: no duplicate rows
            df = df[df["prompt_id"].isin(missing)]
            df["gen_model"] = spec.slug
            df["empty_answer"] = df["answer"].fillna("").astype(str).str.strip() == ""
            buf.append(df)
            n_cond += 1
            if sum(len(b) for b in buf) >= shard_rows or d is last_dir:
                chunk = pd.concat(buf, ignore_index=True)
                buf = []
                t0 = time.perf_counter()
                scored = score_frame(cstar, chunk, shard_rows, log)
                scored.to_parquet(next_shard(out_dir), index=False)
                log(f"    [{n_cond}/{len(todo)}] wrote {len(scored)} rows "
                    f"({time.perf_counter() - t0:.1f}s)")
        summary[spec.slug] = {"conditions_scored": len(todo),
                              "conditions_already_present": n_have}
        log(f"  wrote {out_dir}")

    write_json({"stage": "score-steered", "candidate": cstar.candidate,
                "run_dir": cstar.run_dir.name, "val_parity": cstar.val_parity,
                "metric_basis": cfg["steering"]["scoring"]["metric_basis"],
                "note": "All metrics are defined on the ARGMAX LABEL only: dynamic batch "
                        "composition shifts probabilities by up to 0.023.",
                "models": summary, "env": env_info(),
                "written_utc": datetime.now(timezone.utc).isoformat()},
               paths["scores"] / "score_steered.json")


# --------------------------------------------------------------------------- #
# stage 6: metrics
# --------------------------------------------------------------------------- #
def boot_index(n: int, B: int, seed: int) -> np.ndarray:
    """ONE [B, n] resample index, drawn once and reused by every condition.

    Paired over prompt ids: the same resampled prompts are used for the baseline and for
    the condition, which is what makes the CI on the DIFFERENCE correct. A fresh
    np.random.choice per condition costs ~4 CPU-hours per model; this costs seconds.
    """
    return np.random.default_rng(seed).integers(0, n, size=(B, n), dtype=np.int64)


def paired_pp(base: np.ndarray, cond: np.ndarray, idx: np.ndarray, ci: float) -> dict:
    """pp change (percentage points) of an indicator, with a paired percentile CI."""
    diff = cond.astype(np.float64) - base.astype(np.float64)
    point = 100.0 * float(diff.mean())
    boots = 100.0 * diff[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
    return {"pp": round(point, 3), "ci_lo": round(float(lo), 3),
            "ci_hi": round(float(hi), 3),
            "ci_half_width": round(float(hi - lo) / 2, 3),
            "base_pct": round(100.0 * float(base.mean()), 3),
            "cond_pct": round(100.0 * float(cond.mean()), 3)}


SUBSETS = ("sensitive", "benign_control", "all")


def subset_mask(df: pd.DataFrame, name: str) -> np.ndarray:
    sens = df["is_sensitive"].fillna(False).to_numpy(dtype=bool)
    if name == "sensitive":
        return sens
    if name == "benign_control":
        return ~sens
    return np.ones(len(df), dtype=bool)


def stage_metrics(cfg, args, paths, specs, log) -> None:
    log("\n=== stage: metrics (CPU) ===")
    B = int(cfg["study"]["bootstrap"]["B"])
    ci = float(cfg["study"]["bootstrap"]["ci"])
    seed = int(cfg["study"]["seed"])
    labels = list(cfg["study"]["labels3"])
    min_n = int(cfg["study"].get("min_n", 30))
    lab_col = f"{CSTAR_PREFIX}_label"
    paths["reports"].mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        log(f"  would read {paths['scores']}/<model>/steered and write {paths['reports']}")
        log(f"  bootstrap B={B} ci={ci} seed={seed}, paired over OOD prompt ids, "
            f"reported separately for sensitive / benign_control / all")
        return

    tidy_all, report = [], {"models": {}}
    for spec in specs:
        slug = spec.slug
        scored = read_shards(paths["scores"] / slug / "steered")
        if not len(scored):
            log(f"  [skip] {slug}: no scored steered rows "
                f"({paths['scores'] / slug / 'steered'})")
            continue
        man_path = paths["steered"] / slug / "manifest.json"
        man = read_json(man_path) if man_path.exists() else {}
        n_blocks = man.get("model", {}).get("n_blocks")

        base = scored[scored["condition"] == S.BASELINE_TAG].sort_values("prompt_id")
        if not len(base):
            raise SystemExit(f"{slug}: no {S.BASELINE_TAG} rows in the scored steered "
                             f"output; the batched baseline is the ONLY valid comparison "
                             f"baseline (the batch-1 capture answers are not)")
        base_ids = base["prompt_id"].to_numpy()
        log(f"\n  {slug}: {len(scored)} scored rows, "
            f"{scored['condition'].nunique()} conditions, baseline n={len(base)}")

        masks = {s: subset_mask(base, s) for s in SUBSETS}
        idxs = {s: boot_index(int(masks[s].sum()), B, seed + i)
                for i, s in enumerate(SUBSETS) if masks[s].sum() > 0}
        base_ind = {(s, lab): (base[lab_col].to_numpy()[masks[s]] == lab).astype(np.float64)
                    for s in SUBSETS for lab in labels}

        mrep = {"n_blocks": n_blocks,
                "layers": man.get("layers"),
                "grid": man.get("grid"),
                "layer_choice_note": "each model uses ITS OWN selected layers; absolute "
                                     "row ids are not comparable across models",
                "baseline": {}, "subset_n": {s: int(masks[s].sum()) for s in SUBSETS}}
        for s in SUBSETS:
            n = int(masks[s].sum())
            mrep["baseline"][s] = {
                "n": n,
                "pct": {lab: round(100.0 * float(base_ind[(s, lab)].mean()), 2)
                        if n else None for lab in labels},
                "evidence": "ok" if n >= min_n else "insufficient_evidence"}
            log(f"    baseline {s:15s} n={n:4d}  " + "  ".join(
                f"{lab}={mrep['baseline'][s]['pct'][lab]}%" for lab in labels))

        scale = man.get("scaling", {})
        for tag, sub in scored.groupby("condition", sort=True):
            if S.is_baseline(tag):
                continue
            meta = S.parse_condition_tag(tag)
            sub = sub.sort_values("prompt_id")
            if not np.array_equal(sub["prompt_id"].to_numpy(), base_ids):
                raise RuntimeError(f"{slug}/{tag}: prompt ids do not match the baseline "
                                   f"({len(sub)} vs {len(base)}); the bootstrap is paired "
                                   f"and cannot align these")
            sig = (scale.get(f"{meta['direction']}_L{int(meta['layer']):03d}", {})
                   or {}).get("sigma")
            aos = (meta["alpha"] / sig) if sig else None
            for s in SUBSETS:
                n = int(masks[s].sum())
                if n == 0:
                    continue
                for lab in labels:
                    cond_ind = (sub[lab_col].to_numpy()[masks[s]] == lab).astype(np.float64)
                    st = paired_pp(base_ind[(s, lab)], cond_ind, idxs[s], ci)
                    tidy_all.append({
                        "gen_model": slug, "condition": tag,
                        "layer": meta["layer"],
                        "relative_depth": round(meta["layer"] / n_blocks, 4)
                        if n_blocks else None,
                        "direction": meta["direction"], "alpha": meta["alpha"],
                        "sigma_ood": sig, "alpha_over_sigma": aos,
                        "subset": s, "n": n, "label": lab, **st,
                        "evidence": "ok" if n >= min_n else "insufficient_evidence"})

        report["models"][slug] = mrep

    if not tidy_all:
        log("  no conditions to report")
        return

    tidy = pd.DataFrame(tidy_all)
    tidy.to_parquet(paths["reports"] / "conditions.parquet", index=False)
    tidy.to_csv(paths["reports"] / "conditions.csv", index=False, encoding="utf-8")
    log(f"\n  wrote {paths['reports'] / 'conditions.parquet'} ({len(tidy)} rows)")

    report.update({
        "stage": "metrics", "seed": seed,
        "bootstrap": {"B": B, "ci": ci, "method": "percentile", "paired_over": "prompt_id",
                      "shared_index": True},
        "metric_basis": cfg["steering"]["scoring"]["metric_basis"],
        "labels": labels, "min_n": min_n,
        "x_axis": "alpha / sigma_l (sigma recomputed on the OOD activations)",
        "caveats": dataset_caveats(report, tidy, labels),
        "env": env_info(),
        "written_utc": datetime.now(timezone.utc).isoformat(),
    })
    write_json(report, paths["reports"] / "metrics.json")
    write_summary(report, tidy, cfg, paths, labels, log)
    make_plots(tidy, paths, log)


def dataset_caveats(report, tidy: pd.DataFrame, labels) -> list:
    """The statements that must travel with these numbers, computed not assumed."""
    out = [
        "Mozilla/sensitive-topic-disclaimer-eval ships NO responses and NO human "
        "disclaimer labels; C* is the only label source here and it is out of domain "
        "for these queries (plan.md 1.9).",
        "is_sensitive describes the QUERY, not the response. It is a secondary "
        "stratification and is never ground truth for C*.",
        "The 164 benign rows are an over-disclaimering / false-positive check, not part "
        "of the transfer set. They are never pooled with the 126 sensitive rows "
        "(is_sensitive is per ROW, measured against the pinned revision).",
        "d_hat is frozen from the in-domain Direction split; only sigma_l and r_l were "
        "recomputed on the OOD activations.",
        "Layers are each model's OWN top selected rows. Absolute row ids are not "
        "comparable across models with different block counts; relative depth is given.",
    ]
    for slug, m in report.get("models", {}).items():
        ref = (m.get("baseline", {}).get("sensitive", {}).get("pct") or {}).get("REFUSAL")
        if ref is not None and ref <= FLOOR_RATE_PCT:
            out.append(
                f"{slug}: baseline REFUSAL is {ref}% on the sensitive rows, so the d_R "
                f"arm is STRUCTURALLY FLOORED on this dataset. A flat or tiny d_R curve "
                f"is a property of these synthetic, non-adversarial queries (plan.md "
                f"1.13), NOT evidence that d_R fails to transfer.")
    if len(tidy):
        for s in SUBSETS:
            sub = tidy[tidy.subset == s]
            if len(sub):
                out.append(
                    f"Power ({s}, n={int(sub['n'].iloc[0])}): achieved paired-bootstrap "
                    f"CI half-width is {sub['ci_half_width'].median():.2f} pp (median "
                    f"over conditions, max {sub['ci_half_width'].max():.2f} pp). Report "
                    f"this next to every pp, not in a footnote.")
    return out


def write_summary(report, tidy: pd.DataFrame, cfg, paths, labels, log) -> None:
    """A plain-text report that states the caveats instead of burying them."""
    lines = ["STEP 4 -- OOD DIRECTION TRANSFER (Mozilla/sensitive-topic-disclaimer-eval)",
             "=" * 78, ""]
    lines.append(f"dataset   {cfg['ood']['hf_id']} config={cfg['ood']['config_name']}")
    lines.append(f"revision  {cfg['ood']['revision']}")
    lines.append(f"classifier C* = {cfg['study']['classifier_run_dir']}")
    lines.append(f"bootstrap B={report['bootstrap']['B']} paired over prompt_id, "
                 f"{int(report['bootstrap']['ci'] * 100)}% percentile CI, shared index")
    lines.append("x-axis    alpha / sigma_l (sigma recomputed on the OOD activations)")
    lines.append("")
    lines.append("READ THIS BEFORE THE NUMBERS")
    lines.append("-" * 78)
    for c in report["caveats"]:
        lines.append(f"  * {c}")
    lines.append("")

    for slug, m in report["models"].items():
        lines.append(f"MODEL {slug}  (n_blocks={m.get('n_blocks')})")
        lay = m.get("layers") or {}
        lines.append(f"  layers {lay.get('rows')}  relative depth "
                     f"{lay.get('relative_depth')}")
        for s in SUBSETS:
            b = m["baseline"][s]
            lines.append(f"  baseline {s:15s} n={b['n']:4d}  " + "  ".join(
                f"{lab}={b['pct'][lab]}%" for lab in labels) +
                (f"   [{b['evidence']}]" if b["evidence"] != "ok" else ""))
        sub = tidy[tidy.gen_model == slug]
        for s in ("sensitive", "benign_control"):
            for lab in ("DISCLAIMER", "REFUSAL"):
                q = sub[(sub.subset == s) & (sub.label == lab)]
                if not len(q):
                    continue
                best = q.loc[q["pp"].abs().idxmax()]
                lines.append(
                    f"  largest |pp| {s:15s} {lab:10s}: {best['pp']:+.2f} pp "
                    f"[{best['ci_lo']:+.2f}, {best['ci_hi']:+.2f}] "
                    f"(half-width {best['ci_half_width']:.2f} pp) at {best['condition']} "
                    f"alpha/sigma={best['alpha_over_sigma']:.3f}"
                    if best['alpha_over_sigma'] is not None else
                    f"  largest |pp| {s:15s} {lab:10s}: {best['pp']:+.2f} pp "
                    f"[{best['ci_lo']:+.2f}, {best['ci_hi']:+.2f}] at {best['condition']}")
        lines.append("")

    dest = paths["reports"] / "summary.txt"
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for l in lines:
        log("  " + l)
    log(f"  wrote {dest}")


def make_plots(tidy: pd.DataFrame, paths, log) -> None:
    """pp vs alpha/sigma_l, one figure per (model, direction, subset), a line per layer."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        log(f"  matplotlib unavailable ({type(e).__name__}); "
            f"conditions.csv carries the same numbers")
        return
    n = 0
    for (slug, direction, subset, label), sub in tidy.groupby(
            ["gen_model", "direction", "subset", "label"], sort=True):
        if subset == "all" or label == "NORMAL":
            continue                                   # the two reported arms only
        if sub["alpha_over_sigma"].isna().all():
            continue
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        for layer, g in sub.groupby("layer", sort=True):
            g = g.sort_values("alpha_over_sigma")
            ax.plot(g["alpha_over_sigma"], g["pp"], marker="o", markersize=3,
                    label=f"L{int(layer)}")
            ax.fill_between(g["alpha_over_sigma"], g["ci_lo"], g["ci_hi"], alpha=0.12)
        ax.axhline(0, color="k", lw=0.8)
        ax.axvline(0, color="k", lw=0.8, ls=":")
        ax.set_xlabel("alpha / sigma_l   (OOD sigma)")
        ax.set_ylabel(f"{label} rate change (pp)")
        ax.set_title(f"{slug}  {direction}  {subset}  n={int(sub['n'].iloc[0])}",
                     fontsize=10)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        dest = paths["reports"] / f"pp_{slug}_{direction}_{subset}_{label.lower()}.png"
        fig.savefig(dest, dpi=150)
        plt.close(fig)
        n += 1
    log(f"  wrote {n} plot(s) to {paths['reports']}")


# --------------------------------------------------------------------------- #
# --dry-run inventory
# --------------------------------------------------------------------------- #
def dry_run_inventory(cfg, args, paths, specs, n_prompts, log) -> None:
    """Both grids, per model and total, and which one THIS invocation will run.

    Both are printed every time because the choice between them is a GPU-budget decision
    (~218 vs ~27 GPU-h for three models) and should be made looking at both numbers.
    """
    sel_path = paths["study"] / "selected_layers.json"
    sel_doc = read_json(sel_path) if sel_path.exists() else {}
    if sel_doc and "criterion" not in sel_doc:
        log(f"  WARNING {sel_path} has no 'criterion' -- it is layer_probe.py's "
            f"SUPERSEDED selection. The counts below are sizing only; --stage steer will "
            f"refuse it until --stage probe is re-run.")
    selected = sel_doc.get("selected", {})
    top_k = int(cfg["probe"]["top_k"])
    n_d = len(cfg["ood"]["directions"])
    override = sorted({int(x) for x in str(args.layers).split(",") if str(x).strip()}) \
        if args.layers else None
    shard = parse_shard(args.shard)

    variants = []
    for reduced in (False, True):
        try:
            lt, alphas = resolve_ood_grid(cfg, reduced)
        except SystemExit as e:
            log(f"  {grid_variant(reduced)} grid: not available ({e})")
            continue
        variants.append((reduced, lt, alphas))

    log(f"  prompts per model: {n_prompts}   directions: {n_d}"
        + (f"   layers from --layers: {override}" if override else ""))
    fallback = []
    totals = {}
    for reduced, lt, alphas in variants:
        name = grid_variant(reduced)
        log(f"  -- {name} grid: layers_top={'all' if lt is None else lt}  "
            f"alphas={[a if a != int(a) else int(a) for a in alphas]} "
            f"({len(alphas)} non-zero) --")
        log(f"  {'model':22s} {'layers':>8s} {'conditions':>11s} {'generations':>12s} "
            f"{'GPU-h b8':>9s}")
        tot = 0
        per_task = [0] * (shard[1] if shard else 1)
        for spec in specs:
            if spec.slug in selected:
                n_sel = len(selected[spec.slug])
            else:
                n_sel = top_k
                if spec.slug not in fallback:
                    fallback.append(spec.slug)
            n_l = len(override) if override else (n_sel if lt is None else min(lt, n_sel))
            n_cond = n_l * n_d * len(alphas) + 1
            gens = n_cond * n_prompts
            tot += gens
            for k in range(n_cond):
                per_task[k % len(per_task)] += n_prompts
            log(f"  {spec.slug:22s} {f'{n_l}/{n_sel}':>8s} {n_cond:>11,} {gens:>12,} "
                f"{S.estimate_gpu_hours(gens, batch_speedup=4.0):>9.1f}")
        totals[name] = tot
        log(f"  {'TOTAL ' + name:22s} {'':>8s} {'':>11s} {tot:>12,} "
            f"{S.estimate_gpu_hours(tot, batch_speedup=4.0):>9.1f}   "
            f"({S.estimate_gpu_hours(tot):.1f} GPU-h at batch 1)")
        if shard:
            # An array task runs its share of EVERY model back to back, so its wall time
            # is the sum over models; the slowest task sets the array's --time.
            worst = max(per_task)
            log(f"  {'':22s} {name} --shard {shard[0]}/{shard[1]}: this task "
                f"{per_task[shard[0] - 1]:,} generations "
                f"({S.estimate_gpu_hours(per_task[shard[0] - 1], batch_speedup=4.0):.1f} "
                f"GPU-h b8); heaviest of the {shard[1]} tasks {worst:,} "
                f"({S.estimate_gpu_hours(worst, batch_speedup=4.0):.1f} GPU-h b8)")
    if fallback:
        log(f"  NOTE {sel_path} has no selection for {fallback}; their layer count is "
            f"sized at probe.top_k={top_k}. --stage steer needs the real selection.")
    log("  (a condition count includes the ONE shared batched alpha=0 baseline per model; "
        "GPU-h b8 assumes a 4x batch-8 speedup, which is not yet measured on this grid)")
    log(f"  THIS RUN uses the {grid_variant(args.reduced).upper()} grid"
        f"{' (--reduced)' if args.reduced else ' (a plain run; --reduced for the cheap check)'}"
        f": {totals.get(grid_variant(args.reduced), 0):,} generations")


# --------------------------------------------------------------------------- #
def apply_smoke(cfg, args, log) -> None:
    sm = cfg.get("smoke") or {}
    cfg["models"] = sm.get("models", cfg["models"])
    cfg["generation"] = {**cfg["generation"], **(sm.get("generation") or {})}
    cfg["steering"] = {**cfg["steering"], **(sm.get("steering") or {})}
    # Must mirror step 2's load_cfg: the smoke probe block sets top_k, and the
    # selected-layers provenance guard compares the artifact's top_k against THIS value.
    if "probe" in sm:
        cfg["probe"] = {**cfg.get("probe", {}), **sm["probe"]}
    S.apply_smoke_paths(cfg)
    if sm.get("ood_root"):
        cfg["output"]["ood_root"] = sm["ood_root"]
    # Both the default grid and ood.reduced are capped the SAME way (<= 2 layers, the
    # smoke steering alphas), each resolved through resolve_ood_grid so an 'inherit' or
    # 'all' in either block is honoured before the cap rather than tripping int().
    smoke_alphas = (sm.get("steering") or {}).get("alphas")

    def _cap(reduced: bool) -> dict:
        lt, alphas = resolve_ood_grid(cfg, reduced)
        return {"layers_top": 2 if lt is None else min(lt, 2),
                "alphas": [float(a) for a in smoke_alphas if float(a) != 0]
                if smoke_alphas else alphas}

    ood = {**cfg["ood"], **_cap(False)}
    if isinstance(cfg["ood"].get("reduced"), dict):
        ood["reduced"] = {**cfg["ood"]["reduced"], **_cap(True)}
    cfg["ood"] = ood
    if args.limit is None:
        args.limit = int(sm.get("n_prompts", 8))
    log(f"  SMOKE: models={[m['name'] for m in cfg['models']]} limit={args.limit} "
        f"ood_root={cfg['output']['ood_root']}")


def verify_steer(paths, slug: str, log) -> bool:
    """The full-grid manifest hash, plus every shard record that claims that grid.

    FAILS on: a grid sha256 that does not recompute; a shard record of this grid whose
    batch plan differs from the manifest's (its conditions are not comparable with the
    baseline); a shard that halved batches on OOM (membership no longer static); a
    complete i/n shard set that does not cover the grid exactly once.
    Only LOGS: shard records of another grid (e.g. an earlier --reduced run -- its tags
    are reused by the parity grid, not invalidated) and conditions not yet complete,
    which is progress, not drift.
    """
    root = paths["steered"] / slug
    m = read_json(root / "manifest.json")
    tags = [S.BASELINE_TAG] + [
        S.condition_tag(l, nm, a)
        for l in m["layers"]["rows"] for nm in m["grid"]["directions"]
        for a in m["grid"]["alphas"]]
    got = sha256_lines(tags)
    ok = got == m["grid"]["sha256"]
    log(f"  steer/{slug}: committed {m['grid']['sha256'][:16]}  "
        f"recomputed {got[:16]}  {'MATCH' if ok else 'MISMATCH'}  "
        f"({m['grid'].get('variant', 'legacy')} grid, {len(tags)} conditions)")
    if m["grid"].get("n_conditions") not in (None, len(tags)):
        log(f"    !! manifest n_conditions={m['grid']['n_conditions']} but the grid "
            f"expands to {len(tags)} tags")
        ok = False

    plan = (m.get("batching") or {}).get("batch_plan_sha256")
    groups: dict[int, dict[int, dict]] = {}
    for f in sorted((root / "shards").glob("shard_*.json")):
        rec = read_json(f)
        if rec.get("grid_sha256") != m["grid"]["sha256"]:
            log(f"    {f.name}: records another grid "
                f"({str(rec.get('grid_sha256'))[:16]}), ignored for coverage")
            continue
        i, n = parse_shard(rec["shard"])
        groups.setdefault(n, {})[i] = rec
        if plan and rec.get("batch_plan_sha256") != plan:
            log(f"    !! {f.name}: batch plan {str(rec.get('batch_plan_sha256'))[:16]} != "
                f"manifest {plan[:16]}; its conditions are not comparable with the "
                f"baseline")
            ok = False
        if not (rec.get("batching") or {}).get("static_membership", True):
            log(f"    !! {f.name}: {rec['batching'].get('oom_halvings')} OOM halving(s) -- "
                f"batch membership was NOT static for this shard's conditions")
            ok = False
    for n, recs in sorted(groups.items()):
        if len(recs) < n:
            log(f"    {len(recs)}/{n} shard records present (missing "
                f"{sorted(set(range(1, n + 1)) - set(recs))}) -- incomplete or running")
            continue
        assigned = [t for i in sorted(recs) for t in recs[i]["conditions_assigned"]]
        cover = len(assigned) == len(tags) and set(assigned) == set(tags)
        ok = ok and cover
        log(f"    {n}/{n} shard records: {len(assigned)} assigned vs {len(tags)} in grid, "
            f"{'each exactly once' if cover else 'COVERAGE MISMATCH'}")

    n_prompts = m["grid"].get("n_prompts")
    if n_prompts:
        n_done = sum(1 for t in tags
                     if len(S.read_done_ids(root / t / "answers.jsonl")) >= int(n_prompts))
        log(f"    conditions complete on disk: {n_done}/{len(tags)}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Step 4: OOD direction transfer on Mozilla/sensitive-topic-disclaimer-eval")
    ap.add_argument("--config", default="configs/study.yaml")
    ap.add_argument("--stage", default="all", choices=STAGES)
    ap.add_argument("--models", default=None,
                    help="comma-separated slugs/hf_ids; default = every model in the config")
    ap.add_argument("--limit", type=int, default=None, help="first N OOD prompts")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="steering batch size (default: steering.batch_size)")
    ap.add_argument("--layers", default=None,
                    help="comma-separated layer rows, overriding the probe ranking")
    ap.add_argument("--reduced", action="store_true",
                    help="steer the cheap ood.reduced grid instead of the default "
                         "in-domain-parity grid")
    # Kept so older commands and docs do not break; parity IS the default now.
    ap.add_argument("--full-parity", action="store_true",
                    help="no-op alias: the in-domain-parity grid is now the default")
    ap.add_argument("--shard", default=None, metavar="i/n",
                    help="steer stage: run only conditions k with k %% n == i-1 (1-based, "
                         "step 2's rule; the baseline is k=0, so shard 1 generates it)")
    ap.add_argument("--from-local", default=None,
                    help="prepare from an offline copy (save_to_disk dir, or a "
                         "parquet/jsonl/csv file) instead of the hub")
    ap.add_argument("--token-budget", type=int, default=16384,
                    help="max padded tokens per classifier batch")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan (cells, generations, GPU-h) and exit")
    ap.add_argument("--verify", action="store_true",
                    help="recompute the committed hashes and exit non-zero on drift")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny model, few prompts, short generations, separate ood_root")
    ap.add_argument("--strict", action="store_true", help="abort on the first failure")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="skip the hook/capture identity gate (NOT for real runs)")
    ap.add_argument("--preflight-prompts", type=int, default=4)
    ap.add_argument("--skip-val-parity", action="store_true",
                    help="skip the C* val-parity gate (NOT for real runs)")
    ap.add_argument("--log-every", type=int, default=25)
    args = ap.parse_args()
    if args.full_parity and args.reduced:
        ap.error("--full-parity (the default grid) and --reduced contradict each other")
    # Validated before anything loads, so a malformed array index fails in a second,
    # not after a model load. --shard only partitions the steer stage; accepting it
    # elsewhere would suggest the other stages are sharded when they are not.
    parse_shard(args.shard)
    if args.shard and args.stage != "steer" and not args.verify:
        ap.error(f"--shard applies to --stage steer only (got --stage {args.stage})")

    cfg = G.load_config(G.repo_path(args.config))
    # steering.positions / output_format / save_activations / static_batch_membership and
    # directions.names describe LOCKED behaviour that is hardcoded in src/steer.py. Flipping
    # one in the yaml would otherwise change nothing, silently -- so assert instead.
    S.assert_locked_steering_config(cfg)
    log_root = G.repo_path(cfg["output"]["ood_root"]) / "logs"
    log = Tee(log_root / f"step4_{args.stage.replace('-', '_')}.log")

    log(f"\n{'=' * 78}")
    log(f"step4_ood_mozilla  stage={args.stage}  config={args.config}  "
        f"{datetime.now(timezone.utc).isoformat()}")
    for k, v in env_info().items():
        log(f"  {k:16s} {v}")
    if args.full_parity:
        log("  --full-parity is now the default")
    if args.smoke:
        apply_smoke(cfg, args, log)

    paths = ood_paths(cfg)
    specs = G.parse_models(cfg, args.models.split(",") if args.models else None)
    log(f"  models   {[s.slug for s in specs]}")
    log(f"  ood root {paths['root']}")

    if args.verify:
        # "nothing committed" and "committed and unchanged" must NOT both exit 0, or a
        # --verify used as a gate passes whether or not the stage ever ran. Steps 1-3
        # already exit 1 in that situation; match them.
        n_checked = 0
        ok = True
        if paths["prepare"].exists() and paths["prompts"].exists():
            ok = verify_prepare(paths, log)
            n_checked += 1
        else:
            log(f"  prepare: nothing committed at {paths['prepare']}")
        for spec in specs:
            man = paths["steered"] / spec.slug / "manifest.json"
            if not man.exists():
                log(f"  steer/{spec.slug}: nothing committed")
                continue
            n_checked += 1
            ok = verify_steer(paths, spec.slug, log) and ok
        if n_checked == 0:
            log("\n!! --verify found nothing to verify: no step-4 artifact is committed "
                "yet. Run the stages first; exiting non-zero so this cannot pass as a gate.")
            ok = False
        log.close()
        sys.exit(0 if ok else 1)

    stages = STAGE_ORDER if args.stage == "all" else (args.stage,)
    for st in stages:
        if st == "prepare":
            stage_prepare(cfg, args, paths, log)
        elif st == "generate":
            stage_generate(cfg, args, paths, specs, log)
        elif st == "score":
            stage_score(cfg, args, paths, specs, log)
        elif st == "steer":
            stage_steer(cfg, args, paths, specs, log)
        elif st == "score-steered":
            stage_score_steered(cfg, args, paths, specs, log)
        elif st == "metrics":
            stage_metrics(cfg, args, paths, specs, log)
    log("\ndone")
    log.close()


if __name__ == "__main__":
    main()
