"""Step 2 of plan.md -- the refusal/disclaimer directions and the causal steering sweep.

This is the study's core experiment. It extracts two directions per model per layer from
the 40% Direction split, selects the ten layers worth steering on the 20% Probing split,
proves the steering hook lands exactly where the activations were captured, runs the
alpha sweep on the 40% Steering split, scores every steered answer with C*, and reports
class-wise percentage-point change with a paired bootstrap CI.

STAGES (--stage), each independently runnable so the GPU and classifier stages can be
separate SLURM jobs under different conda envs (the DGX runs generation under `base`,
transformers 5.14.1, and the classifier under `refusal_clf`, transformers 5.17.0 -- one
process cannot do both):

  directions  CPU   balanced difference-of-means d_R / d_D at every steerable layer,
                    plus r_l, sigma_R_l, sigma_D_l, ||mu-mu||, split-half reliability,
                    cos(d_D, d_R).            -> data/study/directions/<slug>.{npz,json}
  probe       CPU   layer selection by mean(AUROC(R vs rest), AUROC(D vs rest)) from a
                    linear probe on the Probing split.  -> data/study/selected_layers.json
  preflight   GPU   steer.preflight_hook_matches_capture per model; HARD FAIL on drift.
                                              -> data/study/steer_preflight.json
  steer       GPU   the sweep: 10 layers x 2 directions x 10 non-zero alphas + ONE shared
                    baseline, per model.      -> data/steered/<slug>/<tag>/answers.jsonl
  score       GPU   C* over every steered answer, classifier loaded ONCE.
                                              -> data/steered/<slug>/predictions.parquet
  metrics     CPU   pp change + paired bootstrap CI + transition matrices + plots.
                                              -> results/reports/step2/
  all               every stage in order (laptop / smoke use).

WHAT THIS SUPERSEDES
--------------------
scripts/layer_probe.py selected layers by probe_bacc_3 x split-half reliability. The
user's locked criterion is mean(AUROC(R vs rest), AUROC(D vs rest)) from a LINEAR probe,
which is what the `probe` stage here implements; layer_probe.py's loaders and split-half
reliability machinery are reused (the reliability number is still reported, it is just no
longer part of the selection score). This stage rewrites data/study/selected_layers.json
and keeps a one-time backup of the pre-existing file.

MEASURED FACTS THIS SCRIPT OBEYS (violating any of them silently corrupts results)
---------------------------------------------------------------------------------
1. Saved activation row `l` is the OUTPUT of decoder module layers[l-1]; row 0 is
   inputs_embeds. src.steer owns the hook and the off-by-one; nothing here re-derives it.
2. Row 0 is byte-identical across prompts and row n_blocks is post-final-norm. Every loop
   over "all layers" here runs over 1..n_blocks-1 INCLUSIVE, via S.assert_steerable.
3. The capture point is `prompt_last` from the prefill, independent of max_new_tokens.
4. Batch-1 is required only where activations are captured. Nothing is captured in the
   steering stage, which is exactly why it may batch.
5. Greedy decoding gives no within-cell variance; every error bar here is the paired
   bootstrap over prompt ids.
6. 1,223 of the 1,750 prompts are in C*'s own training split, so every headline is
   reported twice: on the full Steering split and on the C*-test-clean subset, the
   pre-registered tiebreaker.

Usage:
  python -X utf8 scripts/step2_directions_steering.py --stage all --dry-run
  python -X utf8 scripts/step2_directions_steering.py --stage directions
  python -X utf8 scripts/step2_directions_steering.py --stage probe --verify
  python -X utf8 scripts/step2_directions_steering.py --stage steer --shard 3/8
  python -X utf8 scripts/step2_directions_steering.py --stage score --models qwen3-5-9b
  python -X utf8 scripts/step2_directions_steering.py --stage metrics
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import generate as G      # noqa: E402
from src import steer as S         # noqa: E402

DEFAULT_CONFIG = "configs/study.yaml"
STAGES = ("directions", "probe", "preflight", "steer", "score", "metrics", "all")

# The spec's planned budget, printed verbatim by --dry-run so the actual inventory can be
# read against it. 700 x 10 layers x 2 directions x 10 non-zero alphas + 700 baseline.
PLANNED_PROMPTS = 700
PLANNED_LAYERS = 10
PLANNED_DIRECTIONS = 2
PLANNED_NONZERO_ALPHAS = 10
PLANNED_ROSTER = 3
PLANNED_BATCH_SPEEDUP = 4.0      # measured multiple assumed for batch-8 on A100-40GB


# --------------------------------------------------------------------------- #
# logging / small utilities
# --------------------------------------------------------------------------- #
class Tee:
    """stdout + <out_dir>/<stage>.log. No emoji: the DGX log is read on Windows too."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.f = open(path, "a", encoding="utf-8", newline="\n")
        self("")
        self(f"===== {datetime.now(timezone.utc).isoformat(timespec='seconds')} "
             f"{' '.join(sys.argv)}")

    def __call__(self, *a) -> None:
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()

    def close(self) -> None:
        try:
            self.f.close()
        except Exception:
            pass


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj) -> str:
    """Hash of a canonical JSON rendering: key order and float repr are pinned, so the
    digest is reproducible across runs and machines."""
    return sha256_bytes(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                   ensure_ascii=True).encode("utf-8"))


def write_json(path, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=True), encoding="utf-8")


def rel_to_root(p) -> str:
    """Repo-relative, forward slashes -- the form every committed artifact records."""
    p = Path(p)
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def file_inventory(paths) -> list[dict]:
    """The input inventory every committed artifact carries: path, size, sha256."""
    inv = []
    for p in paths:
        p = Path(p)
        inv.append({"path": rel_to_root(p),
                    "exists": p.exists(),
                    "bytes": int(p.stat().st_size) if p.exists() else None,
                    "sha256": sha256_file(p) if p.is_file() else None})
    return inv


def fail(msg: str) -> "NoReturn":                      # noqa: F821
    """Fail loudly and early: a violated invariant is never worked around."""
    raise SystemExit(f"FATAL: {msg}")


# --------------------------------------------------------------------------- #
# config / roster / study inputs
# --------------------------------------------------------------------------- #
def load_cfg(path, smoke: bool, log) -> dict:
    cfg = G.load_config(G.repo_path(path))
    # steering.positions / output_format / save_activations / static_batch_membership and
    # directions.names describe LOCKED behaviour that is hardcoded in src/steer.py. Flipping
    # one in the yaml would otherwise change nothing, silently -- so assert instead.
    S.assert_locked_steering_config(cfg)
    if not smoke:
        return cfg
    sm = cfg.get("smoke") or {}
    if not sm:
        fail(f"--smoke requested but {path} has no `smoke:` block")
    # The smoke block mirrors generate.yaml: a tiny ungated model, 8 prompts, 24 new
    # tokens, 3 alphas, batch 2. It exists so the whole pipeline can be exercised on a
    # laptop GPU before five days of DGX time are queued.
    cfg = json.loads(json.dumps(cfg))                  # deep copy, config is plain data
    cfg["models"] = sm["models"]
    for key in ("generation", "steering"):
        if key in sm:
            cfg[key] = {**cfg.get(key, {}), **sm[key]}
    if "probe" in sm:
        cfg["probe"] = {**cfg.get("probe", {}), **sm["probe"]}
    S.apply_smoke_paths(cfg)   # step 1 wrote the _smoke tree; read from it, not production
    cfg.setdefault("_smoke", {})["n_prompts"] = int(sm.get("n_prompts", 8))
    log(f"  SMOKE: roots={ {k: cfg['output'][k] for k in S.SMOKE_PATH_KEYS if k in cfg['output']} }")
    log(f"  SMOKE: roster={[m['name'] for m in cfg['models']]} "
        f"n_prompts={cfg['_smoke']['n_prompts']} "
        f"alphas={cfg['steering']['alphas']} batch={cfg['steering'].get('batch_size')}")
    return cfg


@dataclass
class ModelCtx:
    """Everything the stages need about one generation model, read from disk once."""
    spec: G.ModelSpec
    slug: str
    root: Path
    n_blocks: int
    n_rows: int
    hidden_size: int
    manifest: dict
    # True only under --dry-run for a roster model step 1 has not generated yet: the
    # geometry fields are unknown placeholders and no real stage may ever see one.
    placeholder: bool = False

    @property
    def steerable(self) -> list[int]:
        """Fact 2: rows 1 .. n_blocks-1 inclusive."""
        return list(range(1, self.n_blocks))


def model_contexts(cfg, only, log, planning: bool = False) -> list[ModelCtx]:
    """Read each roster model's step-1 manifest.

    `planning=True` is passed ONLY by --dry-run, which exists to price the sweep before
    any GPU time is spent and therefore must work on a machine where step 1 has not run
    for part of the roster. In that mode a missing manifest degrades to a loudly-marked
    placeholder instead of exiting. Every real stage calls this with planning=False, so
    a missing manifest still fails early there.
    """
    out = []
    for spec in G.parse_models(cfg, only=only):
        root = G.repo_path(cfg["output"].get("root", cfg["output"]["generated_root"])) / spec.slug
        man_path = root / "manifest.json"
        if not man_path.exists():
            if not planning:
                fail(f"{spec.slug}: no {rel_to_root(man_path)}. Step 1 (generation + capture) "
                     f"must run for this model before step 2 touches it.")
            log(f"  {spec.slug:<24} NOTE: no {rel_to_root(man_path)} yet -- step 1 has not "
                f"generated this model. Counting it in the plan with the planned "
                f"{PLANNED_LAYERS}-layer grid; its real layer count is unknown until "
                f"step 1 runs.")
            out.append(ModelCtx(spec=spec, slug=spec.slug, root=root, n_blocks=0, n_rows=0,
                                hidden_size=0, manifest={}, placeholder=True))
            continue
        man = json.loads(man_path.read_text(encoding="utf-8"))
        m = man["model"]
        ctx = ModelCtx(spec=spec, slug=spec.slug, root=root,
                       n_blocks=int(m["n_blocks"]), n_rows=int(m["n_layer_rows"]),
                       hidden_size=int(m["hidden_size"]), manifest=man)
        if ctx.n_rows != ctx.n_blocks + 1:
            fail(f"{ctx.slug}: manifest says {ctx.n_rows} layer rows for {ctx.n_blocks} "
                 f"blocks; rows must be n_blocks+1 (row 0 = embeddings).")
        log(f"  {ctx.slug:<24} blocks={ctx.n_blocks} rows={ctx.n_rows} "
            f"H={ctx.hidden_size} steerable=1..{ctx.n_blocks - 1}")
        out.append(ctx)
    return out


def study_dir(cfg) -> Path:
    return G.repo_path(cfg["output"]["study_root"])


def steered_dir(cfg) -> Path:
    return G.repo_path(cfg["output"]["steered_root"])


def load_study_splits(cfg, log) -> dict:
    """The 3-way 40/20/40 partition written by step 1 (data/study/study_splits.json).

    Accepted shape: {"sha256": ..., "seed": ..., "splits": {name: [prompt_id, ...]}}.
    There is no held-out validation split (user decision 2026-09-22); the guard below
    refuses the superseded 35/15/40/10 partition that still carried one.
    """
    path = study_dir(cfg) / "study_splits.json"
    if not path.exists():
        fail(f"no {rel_to_root(path)}. Run step 1 (the split stage) first; step 2 never "
             f"invents a partition of its own.")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if "splits" not in doc:
        fail(f"{rel_to_root(path)} has no 'splits' mapping")
    splits = {k: sorted(int(v) for v in ids) for k, ids in doc["splits"].items()}
    for need in S.SPLIT_NAMES:
        if need not in splits:
            fail(f"{rel_to_root(path)} is missing the '{need}' split")
    overlap = set(splits["direction"]) & set(splits["steering"])
    if overlap:
        fail(f"direction and steering splits share {len(overlap)} prompt ids; the "
             f"directions would be estimated on the prompts they are evaluated on")
    # Structure alone does not prove provenance: the superseded majority-label partition
    # passes every check above. S.assert_study_splits_current checks the stratification
    # rule AND the roster.
    try:
        S.assert_study_splits_current(
            doc, [s.slug for s in G.parse_models(cfg)], path=rel_to_root(path),
            partition=str(cfg.get("study", {}).get("partition", "shared")),
            fractions=S.split_fractions(cfg))
    except (S.StaleArtifactError, ValueError) as e:
        fail(str(e))
    log(f"  study split sha256={doc.get('sha256', '?')[:16]} seed={doc.get('seed')} "
        + " ".join(f"{k}={len(v)}" for k, v in splits.items()))
    doc["splits"] = splits
    return doc


LABEL_CANDIDATES = ("longformer_label", "cstar_label", "label")
MODEL_CANDIDATES = ("gen_model", "model_slug", "slug", "model")
CSTAR_SPLIT_CANDIDATES = ("cstar_split", "split", "splits")


def load_labels(cfg, override, log) -> pd.DataFrame:
    """C* label per (gen_model, prompt_id), plus C*'s own train/val/test membership.

    Step 1 writes data/study/labels.parquet; the September pilot wrote the classifier's
    results/predictions/generated_predictions.parquet. Either is accepted, in that order,
    so this stage is not blocked on which one exists yet. C* is the Longformer run only,
    so the longformer_* columns are the ones read.
    """
    candidates = []
    if override:
        candidates.append(Path(override))
    if cfg["study"].get("labels_table"):
        candidates.append(G.repo_path(cfg["study"]["labels_table"]))
    candidates += [study_dir(cfg) / "labels.parquet",
                   study_dir(cfg) / "generated_predictions.parquet",
                   G.repo_path(cfg["study"]["classifier_repo"]) / "results" / "predictions"
                   / "generated_predictions.parquet"]
    src = next((p for p in candidates if Path(p).exists()), None)
    if src is None:
        fail("no C* label table found. Looked for: "
             + ", ".join(rel_to_root(p) for p in candidates))

    df = pd.read_parquet(src)
    lab = next((c for c in LABEL_CANDIDATES if c in df.columns), None)
    mod = next((c for c in MODEL_CANDIDATES if c in df.columns), None)
    if lab is None or mod is None:
        fail(f"{rel_to_root(src)} has columns {list(df.columns)}; need one of "
             f"{LABEL_CANDIDATES} and one of {MODEL_CANDIDATES}")
    cst = next((c for c in CSTAR_SPLIT_CANDIDATES if c in df.columns), None)

    out = pd.DataFrame({"prompt_id": df["prompt_id"].astype("int64"),
                        "gen_model": df[mod].astype(str),
                        "label": df[lab].astype(str)})
    # C*'s own split. A prompt that appeared in several processed parquets carries a
    # comma-joined value; 'test' then does not match, which is the conservative side of
    # the C*-test-clean subset.
    out["cstar_split"] = df[cst].astype(str) if cst else "unknown"
    labels3 = list(cfg["study"]["labels3"])
    bad = sorted(set(out["label"]) - set(labels3))
    if bad:
        fail(f"{rel_to_root(src)} carries labels {bad} outside {labels3}")
    log(f"  labels from {rel_to_root(src)}: {len(out)} rows, "
        f"models={sorted(out.gen_model.unique())}, "
        f"cstar test-clean={int((out.cstar_split == 'test').sum())}")
    return out


def load_stack(ctx: ModelCtx, prompt_ids, point, log) -> tuple[np.ndarray, list[int]]:
    """[n_prompts, n_layer_rows, H] float32 for the requested prompts.

    Reads whichever layout is on disk (stacked <pid>.npz or per_layer <pid>/) through
    G.load_hidden, so the layout decision stays a file-count decision and nothing here
    depends on it. Prompts with no activations are reported and dropped, never silently
    filled -- a missing prompt means step 1 did not finish for this model.
    """
    hid = ctx.root / "hidden"
    mats, kept, missing = [], [], []
    t0 = time.perf_counter()
    for i, pid in enumerate(prompt_ids):
        pid = int(pid)
        target = hid / str(pid) if (hid / str(pid)).is_dir() else hid / f"{pid}.npz"
        if not Path(target).exists():
            missing.append(pid)
            continue
        arr, layers = G.load_hidden(target, point, None)
        if arr.shape[0] != ctx.n_rows:
            fail(f"{ctx.slug} prompt {pid}: {arr.shape[0]} layer rows on disk, manifest "
                 f"says {ctx.n_rows}")
        mats.append(arr.astype(np.float32))
        kept.append(pid)
        if (i + 1) % 200 == 0:
            log(f"      {i + 1}/{len(prompt_ids)} prompts  {time.perf_counter() - t0:.0f}s")
    if not mats:
        fail(f"{ctx.slug}: no activations found under {rel_to_root(hid)}")
    if missing:
        log(f"    WARNING: {len(missing)} prompt(s) have no stored activations "
            f"(first few: {missing[:8]})")
    X = np.stack(mats)
    log(f"    loaded {X.shape} in {time.perf_counter() - t0:.0f}s")
    return X, kept


# --------------------------------------------------------------------------- #
# stage 1: directions
# --------------------------------------------------------------------------- #
def balanced_dim(Xpos: np.ndarray, Xneg: np.ndarray, target_n: int, draws: int,
                 seed: int) -> np.ndarray:
    """Class-balanced difference-of-means direction (the PDF's sampling approach).

    NORMAL (Xneg) is subsampled down to `target_n` = min(n_R, n_D) so the negative class
    cannot dominate the mean simply by being larger. For each of `draws` seeded draws the
    UNIT difference vector is taken; the unit vectors are averaged and the average is
    renormalised. Averaging unit vectors -- not raw differences -- is what stops one draw
    with a large norm from deciding the direction on its own.
    """
    if len(Xpos) < 2 or len(Xneg) < 2:
        raise ValueError(f"need >=2 rows per class, got pos={len(Xpos)} neg={len(Xneg)}")
    mu_pos = Xpos.mean(axis=0, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(Xneg)
    k = int(min(target_n, n))
    acc = np.zeros(Xpos.shape[1], dtype=np.float64)
    used = 0
    for _ in range(int(draws)):
        idx = rng.choice(n, size=k, replace=False) if k < n else np.arange(n)
        d = mu_pos - Xneg[idx].mean(axis=0, dtype=np.float64)
        nrm = float(np.linalg.norm(d))
        if nrm < 1e-12:
            continue
        acc += d / nrm
        used += 1
    if used == 0:
        raise ValueError("every balanced draw produced a zero difference vector")
    return S.unit(acc)


def split_half_reliability(X: np.ndarray, is_pos: np.ndarray, is_neg: np.ndarray,
                           target_n: int, n_halves: int, draws: int,
                           seed: int) -> tuple[float | None, float | None]:
    """Mean cosine between the direction estimated on two disjoint halves of the prompts.

    Reused from layer_probe.py, with the balanced estimator in place of the plain one so
    the number describes the direction actually being steered. Accuracy alone will
    happily rank a layer whose difference-in-means vector is half noise: measured on this
    data, reliability runs ~0.45 at layer 1 and ~0.95 from layer 20 up.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    cos = []
    for _ in range(int(n_halves)):
        perm = rng.permutation(n)
        for a, b in ((perm[: n // 2], perm[n // 2:]),):
            try:
                d1 = balanced_dim(X[a][is_pos[a]], X[a][is_neg[a]], target_n, draws,
                                  int(rng.integers(0, 2 ** 31 - 1)))
                d2 = balanced_dim(X[b][is_pos[b]], X[b][is_neg[b]], target_n, draws,
                                  int(rng.integers(0, 2 ** 31 - 1)))
            except ValueError:
                continue
            cos.append(float(np.dot(d1, d2)))
    if not cos:
        return None, None
    return float(np.mean(cos)), float(np.std(cos, ddof=1)) if len(cos) > 1 else 0.0


def direction_paths(cfg, slug: str) -> tuple[Path, Path]:
    d = study_dir(cfg) / "directions"
    return d / f"{slug}.npz", d / f"{slug}.json"


def directions_digest(arrays: dict) -> str:
    """sha256 over the float32 vector bytes in canonical key order.

    The .npz file itself is not hashed: a zip carries timestamps, so its bytes drift
    between identical runs. The arrays do not.
    """
    h = hashlib.sha256()
    for k in sorted(arrays):
        h.update(k.encode("utf-8"))
        h.update(np.asarray(arrays[k], dtype=np.float32).tobytes(order="C"))
    return h.hexdigest()


def stage_directions(cfg, args, ctxs, splits, labels, log) -> None:
    dcfg = cfg["directions"]
    seed = int(args.seed)
    draws = int(dcfg.get("balance_draws", 50))
    labels3 = list(cfg["study"]["labels3"])
    dir_ids = splits["splits"]["direction"]
    if args.limit:
        dir_ids = dir_ids[: args.limit]

    for ctx in ctxs:
        npz_path, json_path = direction_paths(cfg, ctx.slug)
        if args.verify:
            verify_directions(npz_path, json_path, log)
            continue
        if npz_path.exists() and json_path.exists() and not args.force:
            # Resume must not silently keep directions estimated on a DIFFERENT Direction
            # split. The artifact records the split it was built on; if that no longer
            # matches, every downstream sigma_l, layer score and pp is measured against
            # the wrong baseline, and nothing later would reveal it.
            prev = json.loads(json_path.read_text(encoding="utf-8")).get("study_split_sha256")
            cur = splits.get("sha256")
            if prev and cur and str(prev) != str(cur):
                fail(f"{rel_to_root(json_path)} was estimated on study split "
                     f"{str(prev)[:16]} but the loaded partition is {str(cur)[:16]}. "
                     f"The Direction split changed; re-run this stage with --force "
                     f"(and re-run --stage probe afterwards, since the layer scores "
                     f"were computed against the old directions).")
            log(f"\n=== {ctx.slug}: directions already written, skipping "
                f"(--force to redo) ===")
            continue

        log(f"\n=== {ctx.slug}: directions on the Direction split ===")
        X, kept = load_stack(ctx, dir_ids, "prompt_last", log)
        lab = (labels[labels.gen_model == ctx.slug].set_index("prompt_id")["label"]
               .reindex(kept))
        if lab.isna().any():
            fail(f"{ctx.slug}: {int(lab.isna().sum())} Direction-split prompts have no "
                 f"C* label; the class means cannot be estimated from them")
        lab = lab.to_numpy()
        is_N = lab == "NORMAL"
        is_D = lab == "DISCLAIMER"
        is_R = lab == "REFUSAL"
        n_N, n_D, n_R = int(is_N.sum()), int(is_D.sum()), int(is_R.sum())
        target_n = min(n_R, n_D)
        log(f"    class counts on this model's OWN C* label: NORMAL={n_N} "
            f"DISCLAIMER={n_D} REFUSAL={n_R}  -> NORMAL subsampled to {target_n}, "
            f"K={draws} draws")
        min_n = int(cfg["study"].get("min_n", 30))
        for name, n in (("NORMAL", n_N), ("DISCLAIMER", n_D), ("REFUSAL", n_R)):
            if n < min_n:
                log(f"    WARNING: {name} has n={n} < min_n={min_n} on the Direction "
                    f"split for {ctx.slug}; d_{name[0]} is reported but under-supported")
            if n < 2:
                fail(f"{ctx.slug}: {name} has n={n} on the Direction split; no direction "
                     f"can be estimated")

        arrays: dict[str, np.ndarray] = {}
        rows = []
        t0 = time.perf_counter()
        for l in ctx.steerable:                        # fact 2: 1 .. n_blocks-1
            S.assert_steerable(l, ctx.n_blocks)
            Xl = X[:, l, :].astype(np.float64)
            # One seed per (model, layer, direction) so a re-run of a single layer
            # reproduces bit-for-bit and adding a model never shifts another's draws.
            base_seed = seed + 1_000_003 * (l + 1) + int(G.sha1(ctx.slug)[:6], 16)
            dR = balanced_dim(Xl[is_R], Xl[is_N], target_n, draws, base_seed)
            dD = balanced_dim(Xl[is_D], Xl[is_N], target_n, draws, base_seed + 1)

            arrays[f"dR_L{l:03d}"] = dR.astype(np.float32)
            arrays[f"dD_L{l:03d}"] = dD.astype(np.float32)

            norms = np.linalg.norm(Xl, axis=1)
            sig_R = S.sigma_along(Xl, dR)
            sig_D = S.sigma_along(Xl, dD)
            sep_R = float(np.linalg.norm(Xl[is_R].mean(0) - Xl[is_N].mean(0)))
            sep_D = float(np.linalg.norm(Xl[is_D].mean(0) - Xl[is_N].mean(0)))
            rel_R, rel_R_sd = split_half_reliability(
                Xl, is_R, is_N, target_n, args.n_halves, args.reliability_draws,
                base_seed + 2)
            rel_D, rel_D_sd = split_half_reliability(
                Xl, is_D, is_N, target_n, args.n_halves, args.reliability_draws,
                base_seed + 3)

            row = {
                "layer": l,
                "relative_depth": round(l / ctx.n_blocks, 4),
                # r_l: the residual stream's own scale at this layer. alpha is not
                # comparable across layers without it.
                "r_l_median_norm": round(float(np.median(norms)), 4),
                "r_l_p10_norm": round(float(np.percentile(norms, 10)), 4),
                "r_l_p90_norm": round(float(np.percentile(norms, 90)), 4),
                # sigma_l: THE denominator of every downstream x-axis (alpha / sigma_l).
                "sigma_R": round(sig_R, 6),
                "sigma_D": round(sig_D, 6),
                "sep_R": round(sep_R, 6),
                "sep_D": round(sep_D, 6),
                "rel_R": None if rel_R is None else round(rel_R, 4),
                "rel_R_sd": None if rel_R_sd is None else round(rel_R_sd, 4),
                "rel_D": None if rel_D is None else round(rel_D, 4),
                "rel_D_sd": None if rel_D_sd is None else round(rel_D_sd, 4),
                "cos_dD_dR": round(float(np.dot(dD, dR)), 4),
            }
            if args.control == "random-unit":
                # Implemented but OFF by default: a random vector L2-matched to the found
                # direction (both are unit, so the match is exact), to show that any pp
                # change is not what an arbitrary perturbation of the same size buys.
                crng = np.random.default_rng(base_seed + 9)
                for k in range(int(args.control_draws)):
                    rv = S.unit(crng.normal(size=Xl.shape[1]))
                    arrays[f"rand{k}_L{l:03d}"] = rv.astype(np.float32)
                    row[f"sigma_rand{k}"] = round(S.sigma_along(Xl, rv), 6)
            rows.append(row)
        log(f"    {len(rows)} steerable layers in {time.perf_counter() - t0:.0f}s")

        arrays["layers"] = np.asarray(ctx.steerable, dtype=np.int32)
        digest = directions_digest(arrays)
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(npz_path, **arrays)

        summary = {
            "slug": ctx.slug,
            "hf_id": ctx.spec.hf_id,
            "seed": seed,
            "written_utc": utcnow(),
            "method": {
                "estimator": "balanced difference-of-means",
                "d_R": "unit(mu_R - mu_N)", "d_D": "unit(mu_D - mu_N)",
                "balance_target": f"NORMAL subsampled to min(n_R, n_D) = {target_n}",
                "balance_draws": draws,
                "averaging": "mean of UNIT vectors, renormalised",
                "estimated_on": "study split 'direction' (40%), this model's OWN C* label",
                "capture_point": "prompt_last",
                "steerable_rows": [1, ctx.n_blocks - 1],
                "excluded_rows": {"0": "embedding output, byte-identical across prompts",
                                  str(ctx.n_blocks): "post-final-norm, not a residual site"},
                "reliability": {"n_halves": args.n_halves,
                                "draws_per_half": args.reliability_draws},
                "control": {"kind": args.control, "draws": int(args.control_draws)
                            if args.control == "random-unit" else 0},
            },
            "n_prompts": len(kept),
            "class_counts": {"NORMAL": n_N, "DISCLAIMER": n_D, "REFUSAL": n_R},
            "min_n": min_n,
            "labels3": labels3,
            "n_blocks": ctx.n_blocks,
            "hidden_size": ctx.hidden_size,
            "prompt_ids": [int(p) for p in kept],
            "study_split_sha256": splits.get("sha256"),
            "inputs": file_inventory([study_dir(cfg) / "study_splits.json",
                                      ctx.root / "manifest.json"]),
            "npz": rel_to_root(npz_path),
            "npz_keys": sorted(arrays),
            "sha256": digest,
            "layers": {str(r["layer"]): r for r in rows},
        }
        write_json(json_path, summary)
        log(f"    wrote {rel_to_root(npz_path)} and {rel_to_root(json_path)} "
            f"sha256={digest[:16]}")

        log(f"\n  {'row':>4} {'depth':>6} {'||h||':>10} {'sigma_R':>9} {'sigma_D':>9} "
            f"{'sep_R':>8} {'sep_D':>8} {'rel_R':>6} {'rel_D':>6} {'cos':>6}")
        for r in rows:
            f = lambda v: f"{v:>6.3f}" if v is not None else "    --"
            log(f"  {r['layer']:>4} {r['relative_depth']:>6.2f} "
                f"{r['r_l_median_norm']:>10.2f} {r['sigma_R']:>9.3f} {r['sigma_D']:>9.3f} "
                f"{r['sep_R']:>8.3f} {r['sep_D']:>8.3f} {f(r['rel_R'])} {f(r['rel_D'])} "
                f"{f(r['cos_dD_dR'])}")


def verify_directions(npz_path: Path, json_path: Path, log) -> None:
    if not (npz_path.exists() and json_path.exists()):
        fail(f"--verify: {rel_to_root(npz_path)} or its summary is missing")
    doc = json.loads(json_path.read_text(encoding="utf-8"))
    with np.load(npz_path) as z:
        arrays = {k: z[k] for k in z.files}
    got = directions_digest(arrays)
    ok = got == doc.get("sha256")
    log(f"  {npz_path.name}: committed {str(doc.get('sha256'))[:16]} recomputed "
        f"{got[:16]}  {'MATCH' if ok else 'MISMATCH'}")
    if not ok:
        raise SystemExit(1)
    for k, v in arrays.items():
        if k == "layers":
            continue
        n = float(np.linalg.norm(v.astype(np.float64)))
        if abs(n - 1.0) > 1e-3:
            fail(f"--verify: {k} has norm {n:.6f}, directions must be unit vectors")


def load_directions(cfg, slug: str, log) -> tuple[dict, dict]:
    npz_path, json_path = direction_paths(cfg, slug)
    if not (npz_path.exists() and json_path.exists()):
        fail(f"{slug}: no directions. Run --stage directions first.")
    with np.load(npz_path) as z:
        arrays = {k: z[k].astype(np.float32) for k in z.files}
    doc = json.loads(json_path.read_text(encoding="utf-8"))
    if directions_digest(arrays) != doc.get("sha256"):
        fail(f"{slug}: directions sha256 drift between {rel_to_root(npz_path)} and its "
             f"summary. Re-run --stage directions --force or investigate.")
    return arrays, doc


# --------------------------------------------------------------------------- #
# stage 2: probe / layer selection
# --------------------------------------------------------------------------- #
def cv_auroc(X: np.ndarray, y: np.ndarray, seed: int, folds: int = 5) -> float | None:
    """Out-of-fold AUROC of a LINEAR probe (standardised + logistic regression).

    Linear on purpose: the intervention is a single additive direction, so a probe that
    needs a nonlinearity is reporting separability the steer cannot exploit. AUROC rather
    than balanced accuracy because it is threshold-free -- class priors differ sharply
    between models, and a thresholded score would rank layers partly on prior.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    u, c = np.unique(y, return_counts=True)
    if len(u) < 2 or c.min() < folds:
        return None
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, C=1.0))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=np.float64)
    for tr, te in skf.split(X, y):
        oof[te] = pipe.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, oof))


def contiguous_and_terminal(top: list[int], n_blocks: int) -> bool:
    """'The deepest ten' -- a selection that buys no depth story and must be flagged."""
    top = sorted(top)
    return (len(top) > 1 and top == list(range(top[0], top[0] + len(top)))
            and top[-1] == n_blocks - 1)


def stage_probe(cfg, args, ctxs, splits, labels, log) -> None:
    pcfg = cfg["probe"]
    top_k = int(args.top_k or pcfg.get("top_k", 10))
    folds = int(pcfg.get("folds", 5))
    dest = study_dir(cfg) / "selected_layers.json"
    report_path = study_dir(cfg) / "step2_probe.json"

    if args.verify:
        verify_probe(dest, top_k, log)
        return

    probe_ids = splits["splits"]["probing"]
    if args.limit:
        probe_ids = probe_ids[: args.limit]

    report = {"seed": int(args.seed), "written_utc": utcnow(),
              "criterion": pcfg.get("criterion", "mean_auroc_R_vs_rest_D_vs_rest"),
              "criterion_detail": "score = mean(AUROC(REFUSAL vs rest), "
                                  "AUROC(DISCLAIMER vs rest)) from a linear probe "
                                  "(standardised logistic regression, 5-fold CV, "
                                  "out-of-fold scores) on the 20% Probing split",
              "supersedes": "scripts/layer_probe.py (probe_bacc_3 x split-half reliability)",
              "folds": folds, "top_k": top_k,
              "study_split_sha256": splits.get("sha256"),
              "n_probing_requested": len(probe_ids),
              "models": {}}
    selected: dict[str, list[int]] = {}

    for ctx in ctxs:
        log(f"\n=== {ctx.slug}: linear probe on the Probing split ===")
        X, kept = load_stack(ctx, probe_ids, "prompt_last", log)
        lab = (labels[labels.gen_model == ctx.slug].set_index("prompt_id")["label"]
               .reindex(kept))
        if lab.isna().any():
            fail(f"{ctx.slug}: {int(lab.isna().sum())} Probing-split prompts have no C* label")
        lab = lab.to_numpy()
        counts = {c: int((lab == c).sum()) for c in cfg["study"]["labels3"]}
        log(f"    probing labels {counts}  n={len(kept)}")

        yR = (lab == "REFUSAL").astype(int)
        yD = (lab == "DISCLAIMER").astype(int)
        rows = []
        t0 = time.perf_counter()
        for l in range(ctx.n_rows):
            steerable = 1 <= l <= ctx.n_blocks - 1
            Xl = X[:, l, :]
            # Rows 0 and n_blocks are scored for completeness and never selected: row 0
            # is byte-identical across prompts (AUROC there is chance by construction)
            # and row n_blocks is post-final-norm, not a residual site a hook can write.
            aR = cv_auroc(Xl, yR, args.seed, folds)
            aD = cv_auroc(Xl, yD, args.seed, folds)
            score = (None if (aR is None or aD is None) else float(np.mean([aR, aD])))
            rows.append({"layer_row": l, "steerable": bool(steerable),
                         "relative_depth": round(l / ctx.n_blocks, 4),
                         "auroc_R_vs_rest": None if aR is None else round(aR, 4),
                         "auroc_D_vs_rest": None if aD is None else round(aD, 4),
                         "score": None if score is None else round(score, 4)})
        log(f"    {ctx.n_rows} rows probed in {time.perf_counter() - t0:.0f}s")

        cand = [r for r in rows if r["steerable"] and r["score"] is not None]
        if len(cand) < top_k:
            fail(f"{ctx.slug}: only {len(cand)} scorable steerable rows, need {top_k}")
        # Ties broken by the DEEPER row losing, so a tie never silently manufactures the
        # "deepest ten" pattern the warning below looks for.
        cand.sort(key=lambda r: (-r["score"], r["layer_row"]))
        top = sorted(r["layer_row"] for r in cand[:top_k])
        selected[ctx.slug] = top

        terminal = contiguous_and_terminal(top, ctx.n_blocks)
        report["models"][ctx.slug] = {
            "n_blocks": ctx.n_blocks, "n_layer_rows": ctx.n_rows,
            "hidden_size": ctx.hidden_size,
            "n_probing": len(kept), "probing_label_counts": counts,
            "layers": rows, "top_layers": top,
            "top_layers_relative_depth": [round(l / ctx.n_blocks, 3) for l in top],
            "top_layers_contiguous_and_terminal": terminal,
        }

        log(f"\n  {'row':>4} {'depth':>6} {'AUROC R':>8} {'AUROC D':>8} {'score':>7}")
        for r in rows:
            if not r["steerable"]:
                why = ("embedding row, byte-identical across prompts" if r["layer_row"] == 0
                       else "post-final-norm, not a residual site")
                log(f"  {r['layer_row']:>4}   -- excluded from selection ({why})")
                continue
            f = lambda v: f"{v:>8.4f}" if v is not None else "      --"
            mark = " <<" if r["layer_row"] in top else ""
            log(f"  {r['layer_row']:>4} {r['relative_depth']:>6.2f} "
                f"{f(r['auroc_R_vs_rest'])} {f(r['auroc_D_vs_rest'])} "
                f"{f(r['score'])}{mark}")
        log(f"\n  TOP {top_k}: {top}   relative depth "
            f"{[round(l / ctx.n_blocks, 2) for l in top]}")
        if terminal:
            log(f"  !! WARNING: {ctx.slug}'s selected layers are CONTIGUOUS and TERMINAL "
                f"({top[0]}..{top[-1]} = the deepest {top_k} steerable rows). That buys no "
                f"depth story -- every steered layer sits in the same late band. Reported, "
                f"NOT changed: the criterion is pre-registered.")

    payload = {k: selected[k] for k in sorted(selected)}
    digest = sha256_json(payload)
    report["selected_layers"] = payload
    report["selected_sha256"] = digest
    report["inputs"] = file_inventory([study_dir(cfg) / "study_splits.json"]
                                      + [c.root / "manifest.json" for c in ctxs])
    write_json(report_path, report)

    # layer_probe.py wrote this same path under the superseded criterion. Keep one
    # backup so the earlier selection is still inspectable, then overwrite.
    if dest.exists():
        try:
            old = json.loads(dest.read_text(encoding="utf-8"))
        except Exception:
            old = {}
        if "criterion" not in old:
            backup = dest.with_name("selected_layers_pre_step2.json")
            if not backup.exists():
                shutil.copy2(dest, backup)
                log(f"\n  kept layer_probe.py's selection at {rel_to_root(backup)}")

    write_json(dest, {
        "sha256": digest,
        "seed": int(args.seed),
        "written_utc": utcnow(),
        "criterion": report["criterion"],
        "criterion_detail": report["criterion_detail"],
        "supersedes": report["supersedes"],
        "top_k": top_k, "folds": folds,
        "study_split_sha256": splits.get("sha256"),
        "selected": payload,
        "relative_depth": {s: [round(l / next(c.n_blocks for c in ctxs if c.slug == s), 3)
                               for l in v] for s, v in payload.items()},
        "contiguous_and_terminal": {s: report["models"][s]["top_layers_contiguous_and_terminal"]
                                    for s in payload},
        "per_layer_table": {s: report["models"][s]["layers"] for s in payload},
        "inputs": report["inputs"],
        "full_report": rel_to_root(report_path),
    })
    log(f"\n  selected-layer sha256 = {digest[:16]}")
    log(f"  wrote {rel_to_root(dest)}")
    log(f"  wrote {rel_to_root(report_path)}")


def verify_probe(dest: Path, top_k: int, log) -> None:
    if not dest.exists():
        fail(f"--verify: {rel_to_root(dest)} does not exist")
    doc = json.loads(dest.read_text(encoding="utf-8"))
    got = sha256_json(doc["selected"])
    ok = got == doc.get("sha256")
    log(f"  selected_layers.json: committed {str(doc.get('sha256'))[:16]} recomputed "
        f"{got[:16]}  {'MATCH' if ok else 'MISMATCH'}")
    # Also recompute the top-k from the committed per-layer table: a hash over the
    # selection alone would not notice a selection that no longer follows its own scores.
    for slug, table in (doc.get("per_layer_table") or {}).items():
        cand = [r for r in table if r["steerable"] and r["score"] is not None]
        cand.sort(key=lambda r: (-r["score"], r["layer_row"]))
        recomputed = sorted(r["layer_row"] for r in cand[:top_k])
        if recomputed != sorted(doc["selected"][slug]):
            log(f"  {slug}: committed {doc['selected'][slug]} != top-{top_k} of the "
                f"committed score table {recomputed}  MISMATCH")
            ok = False
    if not ok:
        raise SystemExit(1)


def sigma_for(ddoc: dict, layer, dname: str):
    """sigma_l for one (layer, direction), tolerating either key spelling.

    The directions artifact keys sigma by CLASS ('sigma_R' / 'sigma_D') while a condition
    tag names the DIRECTION ('dR' / 'dD'). Looking up 'sigma_dR' therefore always missed,
    which silently blanked alpha/sigma_l -- the axis every plot and table in this study is
    required to use (locked decision D1). Try both spellings and fail loudly if neither
    is present rather than degrading to None.
    """
    rec = (ddoc.get("layers", {}) or {}).get(str(layer), {}) or {}
    for key in (f"sigma_{dname}", f"sigma_{str(dname).lstrip('dD') or dname}",
                f"sigma_{str(dname)[1:]}" if len(str(dname)) > 1 else f"sigma_{dname}"):
        if key in rec:
            return rec[key]
    return None


def load_selected_layers(cfg, slug: str, study_sha: str | None = None,
                         top_k: int | None = None) -> list[int]:
    dest = study_dir(cfg) / "selected_layers.json"
    if not dest.exists():
        fail(f"no {rel_to_root(dest)}. Run --stage probe first.")
    doc = json.loads(dest.read_text(encoding="utf-8"))
    pcfg = cfg.get("probe", {}) or {}
    # layer_probe.py's committed file parses fine here but was selected by a different
    # criterion on a different split -- steering on those layers would burn ~176 GPU-h
    # per model and be indistinguishable from a correct run.
    try:
        S.assert_selected_layers_current(
            doc,
            criterion=pcfg.get("criterion", "mean_auroc_R_vs_rest_D_vs_rest"),
            top_k=int(top_k if top_k is not None else pcfg.get("top_k", 10)),
            study_sha=study_sha, slugs=(slug,), path=rel_to_root(dest))
    except S.StaleArtifactError as e:
        fail(str(e))
    return sorted(int(l) for l in doc["selected"][slug])


# --------------------------------------------------------------------------- #
# stage 3: preflight
# --------------------------------------------------------------------------- #
def preflight_path(cfg) -> Path:
    return study_dir(cfg) / "steer_preflight.json"


def stage_preflight(cfg, args, ctxs, splits, log) -> None:
    dest = preflight_path(cfg)
    doc = json.loads(dest.read_text(encoding="utf-8")) if dest.exists() else {"models": {}}
    doc.setdefault("models", {})

    if args.verify:
        for ctx in ctxs:
            rec = doc["models"].get(ctx.slug)
            if not rec or not rec.get("passed"):
                log(f"  {ctx.slug}: NO PASSING PREFLIGHT")
                raise SystemExit(1)
            if int(rec["n_decoder_layers"]) != ctx.n_blocks:
                log(f"  {ctx.slug}: preflight recorded {rec['n_decoder_layers']} decoder "
                    f"layers, manifest says {ctx.n_blocks}  MISMATCH")
                raise SystemExit(1)
            log(f"  {ctx.slug}: passed, {rec['n_decoder_layers']} decoder layers, "
                f"worst layer {rec['worst_layer']['layer']} "
                f"min_cos={rec['worst_layer']['min_cos']:.6f}")
        return

    table = G.build_prompt_table(cfg)
    steer_ids = set(splits["splits"]["steering"])
    rows = [r for r in table.itertuples(index=False) if int(r.prompt_id) in steer_ids]

    for ctx in ctxs:
        if doc["models"].get(ctx.slug, {}).get("passed") and not args.force:
            log(f"\n=== {ctx.slug}: preflight already passed, skipping (--force to redo) ===")
            continue
        log(f"\n=== {ctx.slug}: preflight -- does the hook sit where the capture was? ===")
        runner = G.Runner(ctx.spec, cfg["generation"], cfg["capture"], log=log)
        try:
            res = S.preflight_hook_matches_capture(
                runner, rows, ctx.root, n_probe=args.n_probe,
                point=cfg["capture"]["points"][0], log=log)
        finally:
            del runner
            _empty_cuda_cache()
        res["checked_utc"] = utcnow()
        res["slug"] = ctx.slug
        res["hf_id"] = ctx.spec.hf_id
        res["manifest_n_blocks"] = ctx.n_blocks
        doc["models"][ctx.slug] = res
        log(f"  PASSED: {res['decoder_type']} with {res['n_decoder_layers']} layers, "
            f"steerable rows {res['steerable_rows'][0]}..{res['steerable_rows'][1]}")

    doc["seed"] = int(args.seed)
    doc["written_utc"] = utcnow()
    doc["study_split_sha256"] = splits.get("sha256")
    doc["inputs"] = file_inventory([study_dir(cfg) / "study_splits.json"]
                                   + [c.root / "manifest.json" for c in ctxs])
    doc["sha256"] = sha256_json({k: v.get("per_layer") for k, v in doc["models"].items()})
    write_json(dest, doc)
    log(f"\n  wrote {rel_to_root(dest)}")


def require_preflight(cfg, ctx: ModelCtx, args, log) -> dict:
    """steer refuses to run without a passing preflight for this model."""
    dest = preflight_path(cfg)
    if args.skip_preflight:
        log(f"  !! --skip-preflight: running WITHOUT the hook/capture gate for "
            f"{ctx.slug}. If the hook is off by one layer every number below is wrong.")
        return {"skipped": True}
    if not dest.exists():
        fail(f"no {rel_to_root(dest)}. Run --stage preflight (GPU, cheap) before --stage "
             f"steer, or pass --skip-preflight and accept that the gate did not run.")
    doc = json.loads(dest.read_text(encoding="utf-8"))
    rec = doc.get("models", {}).get(ctx.slug)
    if not rec or not rec.get("passed"):
        fail(f"{ctx.slug} has no passing entry in {rel_to_root(dest)}. Run "
             f"--stage preflight --models {ctx.slug}.")
    if int(rec.get("n_decoder_layers", -1)) != ctx.n_blocks:
        fail(f"{ctx.slug}: preflight saw {rec['n_decoder_layers']} decoder layers, the "
             f"manifest says {ctx.n_blocks}; the model on disk is not the model the gate "
             f"passed on")
    return rec


def _empty_cuda_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# stage 4: the steering sweep
# --------------------------------------------------------------------------- #
def nonzero_alphas(cfg) -> list[float]:
    """The grid minus 0: alpha=0 is the single shared baseline, not a per-cell run."""
    a0 = float(cfg["steering"].get("baseline_alpha", 0))
    return [float(a) for a in cfg["steering"]["alphas"] if float(a) != a0]


def direction_names(cfg, args) -> list[str]:
    names = list(cfg["steering"].get("directions", S.DIRECTION_NAMES))
    if args.control == "random-unit":
        names += [f"rand{k}" for k in range(int(args.control_draws))]
    return names


def build_conditions(cfg, args, layers: list[int]) -> list[tuple[str, int | None, str, float]]:
    """(tag, layer, direction, alpha), baseline FIRST.

    Baseline first so a --shard 1/N task produces it before anything needs it, and so a
    partial sweep always has something to subtract from.
    """
    out = [(S.BASELINE_TAG, None, "baseline", 0.0)]
    alphas = nonzero_alphas(cfg)
    if args.alphas:
        want = {float(x) for x in args.alphas.split(",") if x.strip()}
        alphas = [a for a in alphas if a in want]
    if args.layers:
        want_l = {int(x) for x in args.layers.split(",") if x.strip()}
        layers = [l for l in layers if l in want_l]
    for l in layers:
        for dn in direction_names(cfg, args):
            for a in alphas:
                out.append((S.condition_tag(l, dn, a), l, dn, float(a)))
    if args.conditions:
        want_t = {t.strip() for t in args.conditions.split(",") if t.strip()}
        out = [c for c in out if c[0] in want_t]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        if not 1 <= i <= n:
            fail(f"--shard {args.shard} is out of range")
        out = [c for k, c in enumerate(out) if k % n == (i - 1)]
    return out


def batch_plan_hash(rows, batches: list[list[int]], batch_size: int) -> str:
    """Identity of the batch plan: the prompt ids, in their batches, in order.

    Batched greedy decoding is not bitwise invariant to batch composition, so the
    baseline is only a valid subtrahend if every condition of a model saw EXACTLY this
    plan. The hash is what makes that checkable after the fact, across SLURM array tasks
    that never shared a process.
    """
    payload = {"batch_size": int(batch_size),
               "batches": [[int(rows[i].prompt_id) for i in b] for b in batches]}
    return sha256_json(payload)


def condition_dir(cfg, slug: str, tag: str) -> Path:
    return steered_dir(cfg) / slug / tag


def stage_steer(cfg, args, ctxs, splits, log) -> None:
    scfg = cfg["steering"]
    batch_size = int(args.batch_size or scfg.get("batch_size", 8))
    if batch_size < 1:
        fail("--batch-size must be >= 1")
    table = G.build_prompt_table(cfg)
    steer_ids = splits["splits"]["steering"]
    if args.limit:
        steer_ids = steer_ids[: args.limit]
    if cfg.get("_smoke"):
        steer_ids = steer_ids[: int(cfg["_smoke"]["n_prompts"])]
    id_set = set(int(p) for p in steer_ids)
    rows = [r for r in table.itertuples(index=False) if int(r.prompt_id) in id_set]
    if len(rows) != len(id_set):
        fail(f"prompt table covers {len(rows)} of the {len(id_set)} Steering-split ids")
    rows.sort(key=lambda r: int(r.prompt_id))
    if args.limit or cfg.get("_smoke"):
        log(f"  !! LIMITED RUN: {len(rows)} of the split's prompts. The batch plan, and "
            f"therefore every generated answer, differs from a full run. These outputs "
            f"must not be mixed with full-run outputs; the batch-plan-hash assert below "
            f"will refuse to do so.")

    if args.verify:
        for ctx in ctxs:
            verify_steer(cfg, ctx, log)
        return

    for ctx in ctxs:
        layers = load_selected_layers(cfg, ctx.slug)
        for l in layers:
            S.assert_steerable(l, ctx.n_blocks)
        conditions = build_conditions(cfg, args, layers)
        arrays, ddoc = load_directions(cfg, ctx.slug, log)
        pre = require_preflight(cfg, ctx, args, log)

        log(f"\n=== {ctx.slug}: steering sweep ===")
        log(f"  layers={layers}")
        log(f"  directions={direction_names(cfg, args)}  alphas={nonzero_alphas(cfg)}")
        log(f"  {len(conditions)} condition(s) x {len(rows)} prompts = "
            f"{len(conditions) * len(rows)} generations in this task")

        runner = G.Runner(ctx.spec, cfg["generation"], cfg["capture"], log=log)
        try:
            if int(runner.n_blocks) != ctx.n_blocks:
                fail(f"{ctx.slug}: loaded model has {runner.n_blocks} blocks, manifest "
                     f"says {ctx.n_blocks}")
            if int(runner.hidden_size) != ctx.hidden_size:
                fail(f"{ctx.slug}: loaded model hidden={runner.hidden_size}, manifest "
                     f"says {ctx.hidden_size}")
            gen = S.BatchedGenerator(runner, batch_size=batch_size,
                                     max_prompt_tokens=scfg.get("max_prompt_tokens"),
                                     log=log)
            # ONE plan per model, reused for every condition INCLUDING the baseline.
            batches = gen.plan_batches(rows)
            plan_hash = batch_plan_hash(rows, batches, gen.batch_size)
            log(f"  batch plan hash {plan_hash[:16]} ({len(batches)} batches of "
                f"{gen.batch_size})")
            assert_plan_consistent(cfg, ctx, plan_hash, log)

            done_gens, t_start = 0, time.perf_counter()
            for ci, (tag, layer, dname, alpha) in enumerate(conditions, 1):
                cdir = condition_dir(cfg, ctx.slug, tag)
                ans = cdir / "answers.jsonl"
                have = S.read_done_ids(ans)
                want = {int(r.prompt_id) for r in rows}
                if have >= want:
                    log(f"  [{ci}/{len(conditions)}] {tag}: complete ({len(have)}), skip")
                    continue
                if have:
                    # Static batch membership is the whole basis of the comparison, so a
                    # partial condition is redone in full rather than topped up with a
                    # smaller row set (which would be a different batch plan).
                    log(f"  [{ci}/{len(conditions)}] {tag}: partial ({len(have)}/"
                        f"{len(want)}); REDOING the whole condition -- topping up would "
                        f"rebatch the remaining rows and break static membership")
                    ans.unlink()

                hook_ctx, sigma = None, None
                if not S.is_baseline(tag):
                    key = f"{dname}_L{layer:03d}"
                    if key not in arrays:
                        fail(f"{ctx.slug}: direction {key} is not in the directions npz")
                    vec = arrays[key]
                    sigma = sigma_for(ddoc, layer, dname)
                    # Callable form: BatchedGenerator enters it per batch and asserts the
                    # hook fired exactly once (one prefill) each time.
                    hook_ctx = (lambda m=runner.model, l=layer, v=vec, a=alpha:
                                S.steering_hook(m, l, v, a, tok=runner.tok,
                                                n_blocks=ctx.n_blocks))

                extra = {"condition": tag, "layer": layer, "direction": dname,
                         "alpha": float(alpha)}
                log(f"  [{ci}/{len(conditions)}] {tag}  layer={layer} dir={dname} "
                    f"alpha={alpha}"
                    + (f"  alpha/sigma={alpha / sigma:.3f}" if sigma else ""))
                t0 = time.perf_counter()
                records = gen.generate(rows, batches, hook_ctx=hook_ctx, extra=extra)
                cdir.mkdir(parents=True, exist_ok=True)
                S.write_answers_jsonl(ans, records, mode="w")

                write_json(cdir / "meta.json", {
                    "condition": tag, "layer": layer, "direction": dname,
                    "alpha": float(alpha),
                    "sigma_l": sigma,
                    "alpha_over_sigma": (float(alpha) / sigma) if sigma else None,
                    "x_axis": scfg.get("x_axis", "alpha_over_sigma"),
                    "seed": int(args.seed),
                    "n_rows": len(rows),
                    "n_records": len(records),
                    "batch_plan_hash": plan_hash,
                    "batch_size_requested": gen.requested_batch_size,
                    "batch_size_effective": gen.batch_size,
                    "n_batches": len(batches),
                    "n_padded_repeats": gen.n_padded_repeats,
                    "n_oom_halvings": gen.n_oom_halvings,
                    "limited_run": bool(args.limit or cfg.get("_smoke")),
                    "generation": dict(runner.gen_kwargs),
                    "model": {"slug": ctx.slug, "hf_id": ctx.spec.hf_id,
                              "dtype": ctx.spec.dtype,
                              "attn_implementation": ctx.spec.attn_implementation,
                              "n_blocks": ctx.n_blocks,
                              "hidden_size": ctx.hidden_size,
                              "system_prompt": ctx.spec.system_prompt,
                              "chat_template_kwargs": ctx.spec.chat_template_kwargs},
                    "directions_sha256": ddoc.get("sha256"),
                    "study_split_sha256": splits.get("sha256"),
                    "preflight": {"skipped": bool(pre.get("skipped")),
                                  "worst_layer": pre.get("worst_layer")},
                    "answers_sha256": sha256_file(ans),
                    "seconds": round(time.perf_counter() - t0, 1),
                    "written_utc": utcnow(),
                })

                done_gens += len(records)
                elapsed = time.perf_counter() - t_start
                rate = done_gens / max(elapsed, 1e-9)
                remaining = (len(conditions) - ci) * len(rows)
                log(f"      {len(records)} answers in {time.perf_counter() - t0:.0f}s | "
                    f"cumulative {done_gens:,} gens, {elapsed / 3600:.2f} GPU-h, "
                    f"{rate:.2f} gen/s | projected {remaining / max(rate, 1e-9) / 3600:.1f} "
                    f"GPU-h left in this task")
        finally:
            del runner
            _empty_cuda_cache()


def assert_plan_consistent(cfg, ctx: ModelCtx, plan_hash: str, log) -> None:
    """Every condition of a model, baseline included, must share one batch plan."""
    root = steered_dir(cfg) / ctx.slug
    if not root.is_dir():
        return
    seen = {}
    for meta in sorted(root.glob("*/meta.json")):
        try:
            doc = json.loads(meta.read_text(encoding="utf-8"))
        except Exception:
            continue
        seen.setdefault(doc.get("batch_plan_hash"), []).append(meta.parent.name)
    other = {h: tags for h, tags in seen.items() if h and h != plan_hash}
    if other:
        fail(f"{ctx.slug}: existing conditions were generated under a DIFFERENT batch "
             f"plan ({ {h[:12]: tags[:4] for h, tags in other.items()} }) than this run's "
             f"{plan_hash[:12]}. Batched greedy decoding is not invariant to batch "
             f"composition, so the pp comparison against the baseline would be invalid. "
             f"Re-run the whole model at one --batch-size and one prompt set, or move the "
             f"old output aside.")
    if seen:
        log(f"  batch plan matches {sum(len(v) for v in seen.values())} existing "
            f"condition(s)")


def verify_steer(cfg, ctx: ModelCtx, log) -> None:
    root = steered_dir(cfg) / ctx.slug
    if not root.is_dir():
        log(f"  {ctx.slug}: nothing steered yet at {rel_to_root(root)}")
        raise SystemExit(1)
    ok, hashes = True, set()
    for meta_path in sorted(root.glob("*/meta.json")):
        doc = json.loads(meta_path.read_text(encoding="utf-8"))
        ans = meta_path.parent / "answers.jsonl"
        if not ans.exists():
            log(f"  {meta_path.parent.name}: meta.json without answers.jsonl  MISSING")
            ok = False
            continue
        got = sha256_file(ans)
        if got != doc.get("answers_sha256"):
            log(f"  {meta_path.parent.name}: answers sha256 {got[:16]} != committed "
                f"{str(doc.get('answers_sha256'))[:16]}  MISMATCH")
            ok = False
        n = len(S.read_done_ids(ans))
        if n != int(doc.get("n_rows", -1)):
            log(f"  {meta_path.parent.name}: {n} prompt ids, meta says "
                f"{doc.get('n_rows')}  MISMATCH")
            ok = False
        hashes.add(doc.get("batch_plan_hash"))
    if len(hashes) > 1:
        log(f"  {ctx.slug}: {len(hashes)} distinct batch plan hashes across conditions "
            f"-- the baseline comparison is INVALID")
        ok = False
    log(f"  {ctx.slug}: {len(list(root.glob('*/meta.json')))} condition(s), "
        f"{'OK' if ok else 'FAILED'}")
    if not ok:
        raise SystemExit(1)


# --------------------------------------------------------------------------- #
# stage 5: scoring with C*
# --------------------------------------------------------------------------- #
def import_classifier(cfg, log):
    """Load the classifier project's predict.py without letting its `src` package win.

    Both projects expose a top-level package named `src`. refusal_generation's is already
    imported (G, S above) and its module objects are held by reference, so the safe move
    is: pop `src*` out of sys.modules, let the classifier repo import its own, capture the
    module objects, then put refusal_generation's back. Nothing re-imports `src` after
    this point in the process, and both sets of functions keep working because they hold
    their own globals.
    """
    clf_root = G.repo_path(cfg["study"]["classifier_repo"]).resolve()
    pred_py = clf_root / "scripts" / "predict.py"
    if not pred_py.exists():
        fail(f"no {pred_py}; --stage score needs the classifier repo")

    saved = {k: v for k, v in sys.modules.items() if k == "src" or k.startswith("src.")}
    for k in list(saved):
        del sys.modules[k]
    sys.path.insert(0, str(clf_root))
    try:
        spec = importlib.util.spec_from_file_location("cstar_predict", pred_py)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["cstar_predict"] = mod
        spec.loader.exec_module(mod)
        M = sys.modules["src.metrics"]
        D = sys.modules["src.data"]
    finally:
        for k in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
            del sys.modules[k]
        sys.modules.update(saved)
        try:
            sys.path.remove(str(clf_root))
        except ValueError:
            pass
    log(f"  classifier modules from {rel_to_root(clf_root)} "
        f"(LABELS3={M.LABELS3}, MIN_N={M.MIN_N})")
    if list(M.LABELS3) != list(cfg["study"]["labels3"]):
        fail(f"study.yaml labels3={cfg['study']['labels3']} but the classifier's "
             f"M.LABELS3={list(M.LABELS3)}; the argmax->label mapping would be wrong")
    if int(M.MIN_N) != int(cfg["study"].get("min_n", 30)):
        fail(f"study.yaml min_n={cfg['study'].get('min_n')} but M.MIN_N={M.MIN_N}")
    return mod, M, D, clf_root


def parts_dir(cfg, slug: str) -> Path:
    return steered_dir(cfg) / slug / "predictions_parts"


def predictions_path(cfg, slug: str) -> Path:
    return steered_dir(cfg) / slug / "predictions.parquet"


def load_done_pairs(pdir: Path) -> set[tuple[str, int]]:
    """(condition_tag, prompt_id) already scored, for resume."""
    done: set[tuple[str, int]] = set()
    if not pdir.is_dir():
        return done
    for f in sorted(pdir.glob("part_*.parquet")):
        try:
            df = pd.read_parquet(f, columns=["condition_tag", "prompt_id"])
        except Exception:
            print(f"  warning: unreadable shard {f.name}, ignoring it for resume")
            continue
        done |= set(zip(df.condition_tag.astype(str), df.prompt_id.astype("int64")))
    return done


def stage_score(cfg, args, ctxs, log) -> None:
    import torch

    scoring = cfg["steering"].get("scoring", {})
    shard_rows = int(args.shard_rows or scoring.get("shard_rows", 20000))
    mod, M, D, clf_root = import_classifier(cfg, log)
    run_dir = G.repo_path(cfg["study"]["classifier_run_dir"]).resolve()
    if not run_dir.is_dir():
        fail(f"classifier run dir {run_dir} does not exist")
    if "longformer" not in run_dir.name.lower():
        fail(f"C* is the Longformer run ONLY (locked decision); {run_dir.name} is not it")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"\n=== C* scoring: {run_dir.name} on {device} ===")
    cfg_c, tok, model, amp = mod.load_classifier(run_dir, device, log)
    prefix = "longformer"
    if "longformer" not in str(cfg_c.get("candidate", "")).lower():
        fail(f"run dir candidate is {cfg_c.get('candidate')!r}, expected the Longformer C*")

    # Val-parity gate, ONCE per process: proves the checkpoint, tokenizer and truncation
    # were restored exactly. 200 conditions x 3 models would otherwise reload a 567 MiB
    # checkpoint per condition, which is hours of pure overhead.
    if scoring.get("val_parity_gate", True) and not args.skip_val_parity:
        val = pd.read_parquet(D.repo_path("data/processed") / "val.parquet")
        pv, _ = mod.predict(val["response_text"].tolist(), cfg_c, tok, model, device, amp,
                            args.token_budget)
        saved = np.load(run_dir / "val_probs.npy")[: len(val)]
        agree = float((pv.argmax(1) == saved.argmax(1)).mean())
        log(f"  val parity: agreement={agree:.4f} "
            f"max|dp|={float(np.abs(pv - saved).max()):.4f} n={len(val)}")
        if agree < getattr(mod, "MIN_VAL_AGREEMENT", 0.99):
            fail(f"val parity {agree:.4f} below the gate; the checkpoint or tokenisation "
                 f"was not restored correctly and every steered label would be suspect")
        del pv, saved

    for ctx in ctxs:
        root = steered_dir(cfg) / ctx.slug
        if not root.is_dir():
            log(f"\n  {ctx.slug}: nothing steered yet, skipping")
            continue
        pdir = parts_dir(cfg, ctx.slug)
        pdir.mkdir(parents=True, exist_ok=True)
        done = load_done_pairs(pdir)
        cond_dirs = sorted(p for p in root.iterdir()
                           if p.is_dir() and (p / "answers.jsonl").exists())
        log(f"\n=== {ctx.slug}: {len(cond_dirs)} condition(s), {len(done)} rows already "
            f"scored ===")

        buf: list[dict] = []
        n_written = len(done)
        part_ix = len(list(pdir.glob("part_*.parquet")))
        t0 = time.perf_counter()

        def flush(buf_rows) -> int:
            nonlocal part_ix
            if not buf_rows:
                return 0
            texts = [r["answer"] for r in buf_rows]
            probs, n_trunc = mod.predict(texts, cfg_c, tok, model, device, amp,
                                         args.token_budget)
            df = pd.DataFrame([{k: v for k, v in r.items() if k != "answer"}
                               for r in buf_rows])
            # Metrics are defined on the ARGMAX LABEL only: dynamic batch composition
            # shifts the probabilities by up to 0.023, so any probability-based statistic
            # would carry batching noise. The probs are stored for inspection, not used.
            df = pd.concat([df, mod.prob_columns(prefix, probs)], axis=1)
            df["classifier_truncated"] = False
            out = pdir / f"part_{part_ix:05d}.parquet"
            df.to_parquet(out, index=False)
            part_ix += 1
            log(f"    shard {out.name}: {len(df)} rows "
                f"(classifier truncated {n_trunc}) "
                f"{time.perf_counter() - t0:.0f}s elapsed")
            return len(df)

        for cdir in cond_dirs:
            tag = cdir.name
            try:
                parsed = S.parse_condition_tag(tag)
            except ValueError:
                log(f"    skipping {tag}: not a condition tag")
                continue
            df = pd.read_json(cdir / "answers.jsonl", lines=True,
                              dtype={"prompt_id": "int64"})
            if "answer" not in df.columns:
                fail(f"{cdir / 'answers.jsonl'} has no 'answer' column")
            df["answer"] = df["answer"].fillna("").astype(str)
            for r in df.itertuples(index=False):
                pid = int(r.prompt_id)
                if (tag, pid) in done:
                    continue
                buf.append({
                    "prompt_id": pid,
                    "condition_tag": tag,
                    "layer": parsed["layer"],
                    "direction": parsed["direction"],
                    "alpha": float(parsed["alpha"]),
                    "answer": r.answer,
                    "n_generated_tokens": int(getattr(r, "n_generated_tokens", 0) or 0),
                    "truncated": bool(getattr(r, "truncated", False)),
                    "empty_answer": not r.answer.strip(),
                })
                if len(buf) >= shard_rows:
                    n_written += flush(buf)
                    buf = []
        n_written += flush(buf)

        consolidate_predictions(cfg, ctx, log)
        log(f"  {ctx.slug}: {n_written} scored rows in "
            f"{time.perf_counter() - t0:.0f}s")

    del model
    _empty_cuda_cache()


def consolidate_predictions(cfg, ctx: ModelCtx, log) -> None:
    """Concatenate the shards into the one committed parquet, plus a manifest.

    Shards are written per 20k rows so a crash five hours in loses at most one shard;
    the single parquet is what the metrics stage and any human reader open.
    """
    pdir = parts_dir(cfg, ctx.slug)
    parts = sorted(pdir.glob("part_*.parquet"))
    if not parts:
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    before = len(df)
    df = df.drop_duplicates(subset=["condition_tag", "prompt_id"], keep="last")
    if len(df) != before:
        log(f"    dropped {before - len(df)} duplicate (condition, prompt) rows across "
            f"shards (a re-run of a partial shard)")
    df = df.sort_values(["condition_tag", "prompt_id"]).reset_index(drop=True)
    dest = predictions_path(cfg, ctx.slug)
    df.to_parquet(dest, index=False)
    write_json(dest.with_name("predictions_manifest.json"), {
        "slug": ctx.slug,
        "written_utc": utcnow(),
        "n_rows": int(len(df)),
        "n_conditions": int(df.condition_tag.nunique()),
        "n_prompts": int(df.prompt_id.nunique()),
        "classifier_run_dir": cfg["study"]["classifier_run_dir"],
        "metric_basis": cfg["steering"].get("scoring", {}).get("metric_basis",
                                                               "argmax_label"),
        "shards": [p.name for p in parts],
        "columns": list(df.columns),
        "label_counts": {k: int(v) for k, v in
                         df[f"longformer_label"].value_counts().items()},
        "sha256": sha256_file(dest),
    })
    log(f"    wrote {rel_to_root(dest)} ({len(df)} rows, "
        f"{df.condition_tag.nunique()} conditions)")


# --------------------------------------------------------------------------- #
# stage 6: metrics
# --------------------------------------------------------------------------- #
def make_boot_index(n: int, B: int, seed: int) -> np.ndarray:
    """ONE [B, n] resample index, built once and reused by every condition.

    Two reasons, and the second is the important one:
      * cost -- a fresh np.random.choice per condition is ~4 CPU-hours per model over
        200 conditions x 3 statistics; one shared index is minutes.
      * correctness -- the bootstrap is PAIRED over prompt ids. Resampling prompts (not
        answers) and applying the same resample to the baseline and the steered condition
        is what makes the CI a CI on the DIFFERENCE. Reusing one index across conditions
        additionally makes the condition-to-condition comparison share its noise, so the
        curves in a plot move together rather than jittering independently.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(int(B), int(n)), dtype=np.int32)


def boot_ci(sample: np.ndarray, ci: float) -> tuple[float, float]:
    lo, hi = np.percentile(sample, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
    return float(lo), float(hi)


def stage_metrics(cfg, args, ctxs, splits, labels, log) -> None:
    import matplotlib
    matplotlib.use("Agg")                     # headless: SLURM has no display
    import matplotlib.pyplot as plt

    bcfg = cfg["study"]["bootstrap"]
    B = int(args.bootstrap or bcfg.get("B", 10000))
    ci = float(bcfg.get("ci", 0.95))
    labels3 = list(cfg["study"]["labels3"])
    min_n = int(cfg["study"].get("min_n", 30))
    # A smoke run must never overwrite the real report directory.
    out_dir = G.repo_path("results/reports/step2_smoke" if cfg.get("_smoke")
                          else "results/reports/step2")
    out_dir.mkdir(parents=True, exist_ok=True)

    cstar_split = (labels.drop_duplicates("prompt_id").set_index("prompt_id")["cstar_split"])
    clean_ids = set(int(p) for p in cstar_split[cstar_split == "test"].index)

    pp_rows: list[dict] = []
    # Two further views of the SAME per-prompt label arrays the transition counts below
    # are built from -- not a second accounting. tr_rows is the per-prompt before/after
    # record on the full split; tt_rows is the long-format 3x3 behind the bar graphs.
    tr_rows: list[dict] = []
    tt_rows: list[dict] = []
    report = {
        "seed": int(args.seed), "written_utc": utcnow(),
        "bootstrap": {"B": B, "ci": ci, "method": bcfg.get("method", "percentile"),
                      "paired_over": "prompt_id", "shared_index": True},
        "metric_basis": cfg["steering"].get("scoring", {}).get("metric_basis",
                                                               "argmax_label"),
        "labels3": labels3, "min_n": min_n,
        "subsets": {
            "full": "the whole 40% Steering split",
            "cstar_test_clean": "Steering-split prompts whose C* split is 'test'. "
                                "1,223 of the 1,750 prompts are in C*'s own training "
                                "split, so this subset is the PRE-REGISTERED TIEBREAKER: "
                                "where full and clean disagree, clean wins.",
        },
        "models": {},
    }

    for ctx in ctxs:
        pred_path = predictions_path(cfg, ctx.slug)
        if not pred_path.exists():
            log(f"\n  {ctx.slug}: no {rel_to_root(pred_path)}, skipping "
                f"(run --stage score)")
            continue
        pred = pd.read_parquet(pred_path)
        pred["prompt_id"] = pred["prompt_id"].astype("int64")
        lab_col = "longformer_label"
        if lab_col not in pred.columns:
            fail(f"{rel_to_root(pred_path)} has no {lab_col} column")

        base = pred[pred.condition_tag == S.BASELINE_TAG]
        if base.empty:
            fail(f"{ctx.slug}: no {S.BASELINE_TAG} rows in {rel_to_root(pred_path)}; "
                 f"there is nothing to subtract")
        base = base.drop_duplicates("prompt_id").set_index("prompt_id")

        _, ddoc = load_directions(cfg, ctx.slug, log)
        conds = [t for t in sorted(pred.condition_tag.unique()) if not S.is_baseline(t)]
        log(f"\n=== {ctx.slug}: {len(conds)} steered condition(s), baseline n={len(base)} ===")

        mrep = {"n_conditions": len(conds), "baseline_n": int(len(base)), "subsets": {}}
        for subset, ids in (("full", sorted(base.index)),
                            ("cstar_test_clean", sorted(set(base.index) & clean_ids))):
            ids = np.asarray(ids, dtype=np.int64)
            n = len(ids)
            if n < min_n:
                log(f"  subset {subset}: n={n} < min_n={min_n}; reported as counts only")
            bsub = base.loc[ids]
            b_lab = bsub[lab_col].to_numpy()
            b_len = bsub["n_generated_tokens"].to_numpy(dtype=np.float64)
            b_trunc = bsub["truncated"].to_numpy(dtype=np.float64)
            b_onehot = {c: (b_lab == c).astype(np.float64) for c in labels3}
            base_pct = {c: float(100 * b_onehot[c].mean()) for c in labels3}

            # ONE index per (model, subset) -- see make_boot_index.
            idx = make_boot_index(n, B, int(args.seed) + 7919 * (subset == "full"))
            log(f"  subset {subset}: n={n}  baseline "
                + " ".join(f"{c}={base_pct[c]:.1f}%" for c in labels3))

            srep = {"n": int(n), "baseline_pct": base_pct,
                    "baseline_median_tokens": float(np.median(b_len)),
                    "baseline_truncation_rate": float(b_trunc.mean()),
                    "bootstrap_index_seed": int(args.seed) + 7919 * (subset == "full"),
                    "conditions": {}}
            t0 = time.perf_counter()
            for tag in conds:
                sub = pred[pred.condition_tag == tag].drop_duplicates("prompt_id")
                sub = sub.set_index("prompt_id")
                miss = [int(p) for p in ids if p not in sub.index]
                if miss:
                    log(f"    {tag}: {len(miss)} prompt(s) unscored, skipping "
                        f"(re-run --stage score)")
                    continue
                s = sub.loc[ids]
                s_lab = s[lab_col].to_numpy()
                meta = S.parse_condition_tag(tag)
                layer, dname, alpha = meta["layer"], meta["direction"], meta["alpha"]
                sigma = sigma_for(ddoc, layer, dname)
                aos = (float(alpha) / sigma) if sigma else None

                crec = {"layer": layer, "direction": dname, "alpha": float(alpha),
                        "sigma_l": sigma, "alpha_over_sigma": aos, "n": int(n),
                        "classwise": {}}
                for c in labels3:
                    s_one = (s_lab == c).astype(np.float64)
                    d = s_one - b_onehot[c]                # paired per-prompt difference
                    pp = float(100 * d.mean())
                    lo, hi = boot_ci(100 * d[idx].mean(axis=1), ci)
                    crec["classwise"][c] = {
                        "baseline_pct": base_pct[c],
                        "steered_pct": float(100 * s_one.mean()),
                        "pp": pp, "ci_lo": lo, "ci_hi": hi,
                        "significant": bool(lo > 0 or hi < 0),
                        "evidence": "ok" if n >= min_n else "insufficient_evidence",
                    }
                    pp_rows.append({"model": ctx.slug, "subset": subset,
                                    "condition_tag": tag, "layer": layer,
                                    "direction": dname, "alpha": float(alpha),
                                    "sigma_l": sigma, "alpha_over_sigma": aos,
                                    "class": c, "n": int(n),
                                    "baseline_pct": base_pct[c],
                                    "steered_pct": float(100 * s_one.mean()),
                                    "pp": pp, "ci_lo": lo, "ci_hi": hi})
                total_pp = sum(crec["classwise"][c]["pp"] for c in labels3)
                if abs(total_pp) > 1e-6:
                    fail(f"{ctx.slug} {tag} {subset}: class-wise pp sums to {total_pp}, "
                         f"must be 0 -- the label sets are not aligned per prompt")

                # Length and truncation move WITH the steer, and a steered answer that
                # runs into the 1024-token cap is scored on a fragment. That is the one
                # real mechanism by which a DISCLAIMER could be cut into a REFUSAL, so it
                # is reported next to every pp rather than in a footnote.
                s_len = s["n_generated_tokens"].to_numpy(dtype=np.float64)
                s_trunc = s["truncated"].to_numpy(dtype=np.float64)
                dt = s_trunc - b_trunc
                tlo, thi = boot_ci(100 * dt[idx].mean(axis=1), ci)
                if args.boot_length:
                    dl_boot = (np.median(s_len[idx], axis=1)
                               - np.median(b_len[idx], axis=1))
                    llo, lhi = boot_ci(dl_boot, ci)
                else:
                    llo = lhi = None
                crec["delta_truncation_rate_pp"] = {
                    "pp": float(100 * dt.mean()), "ci_lo": tlo, "ci_hi": thi,
                    "baseline_rate": float(b_trunc.mean()),
                    "steered_rate": float(s_trunc.mean())}
                crec["delta_median_tokens"] = {
                    "delta": float(np.median(s_len) - np.median(b_len)),
                    "ci_lo": llo, "ci_hi": lhi,
                    "baseline_median": float(np.median(b_len)),
                    "steered_median": float(np.median(s_len))}
                crec["empty_answer_rate"] = float(
                    s["empty_answer"].to_numpy(dtype=np.float64).mean()
                    if "empty_answer" in s.columns else 0.0)

                # Transition matrix: where each baseline class went. Rates only where the
                # baseline row has >= MIN_N support; below that the cell is a count.
                T, counts = {}, {}
                for i_c in labels3:
                    m = b_onehot[i_c].astype(bool)
                    denom = int(m.sum())
                    counts[i_c] = {j: int(((s_lab == j) & m).sum()) for j in labels3}
                    if denom >= min_n:
                        T[i_c] = {j: counts[i_c][j] / denom for j in labels3}
                    else:
                        T[i_c] = None          # reported as counts, flagged below
                crec["transition"] = {
                    "rates": T, "counts": counts,
                    "baseline_row_n": {c: int(b_onehot[c].sum()) for c in labels3},
                    "rows_below_min_n": [c for c in labels3
                                         if int(b_onehot[c].sum()) < min_n],
                    "note": f"T[i][j] = count(i at baseline -> j when steered) / "
                            f"count(i at baseline); rows with n < {min_n} carry counts "
                            f"only (null rates) per the project's min-n rule",
                }
                # rate_pct is x100 and stays populated below min_n: the below_min_n flag
                # carries the rule, so the table never has to be re-joined to find a count.
                aos_f = float("nan") if aos is None else float(aos)
                for i_c in labels3:
                    denom = int(b_onehot[i_c].sum())
                    for j in labels3:
                        tt_rows.append({
                            "model": ctx.slug, "subset": subset, "condition_tag": tag,
                            "layer": int(layer), "direction": str(dname),
                            "alpha": float(alpha), "alpha_over_sigma": aos_f,
                            "from_class": i_c, "to_class": j,
                            "count": counts[i_c][j], "denom": denom,
                            "rate_pct": (100.0 * counts[i_c][j] / denom) if denom
                                        else float("nan"),
                            "below_min_n": bool(denom < min_n)})
                if subset == "full":
                    # The clean subset is a filter on cstar_split, so the per-prompt
                    # record is written once, over the whole Steering split.
                    csp = cstar_split.reindex(ids).fillna("unknown").astype(str).to_numpy()
                    for p, b_l, s_l, cs in zip(ids, b_lab, s_lab, csp):
                        tr_rows.append({
                            "prompt_id": int(p), "model": ctx.slug, "layer": int(layer),
                            "direction": str(dname), "alpha": float(alpha),
                            "alpha_over_sigma": aos_f, "before": str(b_l),
                            "after": str(s_l), "changed": bool(b_l != s_l),
                            "cstar_split": str(cs)})
                srep["conditions"][tag] = crec
            log(f"    {len(srep['conditions'])} condition(s) bootstrapped "
                f"(B={B}) in {time.perf_counter() - t0:.0f}s")
            mrep["subsets"][subset] = srep

        report["models"][ctx.slug] = mrep

    if not report["models"]:
        fail("no model had predictions to report on; run --stage score first")

    df = pd.DataFrame(pp_rows)
    df.to_parquet(out_dir / "pp_table.parquet", index=False)
    report["pp_table"] = rel_to_root(out_dir / "pp_table.parquet")
    report["inputs"] = file_inventory(
        [predictions_path(cfg, c.slug) for c in ctxs]
        + [study_dir(cfg) / "selected_layers.json", study_dir(cfg) / "study_splits.json"])
    report["figures"] = make_plots(cfg, report, df, out_dir, plt, log)

    tr = pd.DataFrame(tr_rows, columns=TRANSITION_PP_COLUMNS)
    tr = tr.astype({"prompt_id": "int64", "layer": "int64", "alpha": "float64",
                    "alpha_over_sigma": "float64", "changed": "bool"})
    if len(tr) and not (tr["changed"] == (tr["before"] != tr["after"])).all():
        fail("transitions_per_prompt: changed != (before != after) on some row")
    tr.to_parquet(out_dir / "transitions_per_prompt.parquet", index=False)
    tt = pd.DataFrame(tt_rows, columns=TRANSITION_TABLE_COLUMNS)
    tt = tt.astype({"layer": "int64", "alpha": "float64", "alpha_over_sigma": "float64",
                    "count": "int64", "denom": "int64", "rate_pct": "float64",
                    "below_min_n": "bool"})
    tt.to_parquet(out_dir / "transitions_table.parquet", index=False)
    tt.to_csv(out_dir / "transitions_table.csv", index=False)
    # Kept out of report["figures"] on purpose: metrics.json keeps its exact prior shape.
    bar_figs = make_transition_bars(cfg, tt, out_dir, plt, log)
    # Main-paper figure: the top-N of the steered layers by peak target-class pp. Its
    # own JSON, so metrics.json keeps its shape; every table above still has all layers.
    ranking = rank_main_figure_layers(cfg, df, log)
    write_json(out_dir / "main_figure_layers.json", ranking)
    main_figs = make_main_transition_bars(cfg, tt, ranking, out_dir, plt, log)

    report["sha256"] = sha256_json({m: {s: {t: v["classwise"]
                                            for t, v in sub["conditions"].items()}
                                        for s, sub in mr["subsets"].items()}
                                    for m, mr in report["models"].items()})
    write_json(out_dir / "metrics.json", report)
    write_summary_md(cfg, report, df, out_dir / "summary.md", log, bar_figs=bar_figs,
                     main_figs=main_figs, ranking=ranking)
    log(f"\n  wrote {rel_to_root(out_dir / 'metrics.json')} "
        f"sha256={report['sha256'][:16]}")
    log(f"  wrote {rel_to_root(out_dir / 'pp_table.parquet')} ({len(df)} rows)")
    log(f"  wrote {rel_to_root(out_dir / 'transitions_per_prompt.parquet')} "
        f"({len(tr)} rows, {int(tr['changed'].sum())} changed)")
    log(f"  wrote {rel_to_root(out_dir / 'transitions_table.parquet')} + .csv "
        f"({len(tt)} rows)")
    log(f"  wrote {len(bar_figs)} transition bar figure(s) (transitions_bar_*.png)")
    log(f"  wrote {len(main_figs)} main-paper figure(s) (transitions_bar_main_*.png) + "
        f"{rel_to_root(out_dir / 'main_figure_layers.json')}")
    log(f"  wrote {rel_to_root(out_dir / 'summary.md')}")


def target_class(direction: str) -> str:
    """The class a direction is supposed to push toward."""
    return {"dR": "REFUSAL", "dD": "DISCLAIMER"}.get(direction, "REFUSAL")


def make_plots(cfg, report, df: pd.DataFrame, out_dir: Path, plt, log) -> list[str]:
    """pp vs alpha/sigma_l (layers as series) and a transition heatmap for the best layer.

    Matplotlib only, no seaborn. The x-axis is alpha/sigma_l everywhere: a layer whose
    residual stream is 10x larger is otherwise made to look 10x less steerable.
    """
    labels3 = list(cfg["study"]["labels3"])
    figs = []
    for model, mdf in df.groupby("model"):
        for subset, sdf in mdf.groupby("subset"):
            for dname, ddf in sdf.groupby("direction"):
                if ddf.empty:
                    continue
                fig, axes = plt.subplots(1, len(labels3), figsize=(5 * len(labels3), 4),
                                         sharex=True, sharey=True)
                axes = np.atleast_1d(axes)
                layers = sorted(ddf.layer.dropna().unique())
                cmap = plt.get_cmap("viridis")
                for ax, c in zip(axes, labels3):
                    for k, l in enumerate(layers):
                        cur = ddf[(ddf["class"] == c) & (ddf.layer == l)].copy()
                        x = cur["alpha_over_sigma"]
                        if x.isna().all():
                            x = cur["alpha"]
                        cur = cur.assign(_x=x).sort_values("_x")
                        col = cmap(k / max(1, len(layers) - 1))
                        ax.plot(cur["_x"], cur["pp"], marker="o", ms=3, lw=1.2,
                                color=col, label=f"L{int(l)}")
                        ax.fill_between(cur["_x"], cur["ci_lo"], cur["ci_hi"],
                                        color=col, alpha=0.12, linewidth=0)
                    ax.axhline(0, color="0.4", lw=0.8)
                    ax.axvline(0, color="0.4", lw=0.8)
                    ax.set_title(c)
                    ax.set_xlabel(r"$\alpha/\sigma_l$")
                axes[0].set_ylabel("percentage-point change vs baseline")
                axes[-1].legend(fontsize=6, ncol=2, title="layer")
                fig.suptitle(f"{model} | {dname} | {subset} "
                             f"(95% CI, paired bootstrap over prompt ids)")
                fig.tight_layout()
                name = f"pp_{model}_{dname}_{subset}.png"
                fig.savefig(out_dir / name, dpi=150)
                plt.close(fig)
                figs.append(name)

    # Transition heatmap for the best layer of each (model, direction): the cell with the
    # largest pp on the direction's target class, preferring the pre-registered subset.
    for model, mrep in report["models"].items():
        for dname in sorted(set(df[df.model == model].direction.dropna())):
            pick = None
            for subset in ("cstar_test_clean", "full"):
                sub = mrep["subsets"].get(subset, {}).get("conditions", {})
                cands = [(v["classwise"][target_class(dname)]["pp"], t, v)
                         for t, v in sub.items() if v["direction"] == dname]
                if cands:
                    pick = (subset, *max(cands, key=lambda z: z[0]))
                    break
            if pick is None:
                continue
            subset, _, tag, rec = pick
            T, counts = rec["transition"]["rates"], rec["transition"]["counts"]
            labels3 = list(cfg["study"]["labels3"])
            Mx = np.array([[(T[i][j] if T[i] else np.nan) for j in labels3]
                           for i in labels3], dtype=float)
            fig, ax = plt.subplots(figsize=(5.2, 4.4))
            im = ax.imshow(Mx, cmap="magma", vmin=0, vmax=1)
            ax.set_xticks(range(len(labels3)), labels3, rotation=30, ha="right")
            ax.set_yticks(range(len(labels3)), labels3)
            ax.set_xlabel("steered label")
            ax.set_ylabel("baseline label")
            for i, ri in enumerate(labels3):
                for j, rj in enumerate(labels3):
                    txt = (f"{Mx[i, j]:.2f}\nn={counts[ri][rj]}" if T[ri]
                           else f"n={counts[ri][rj]}\n(<min_n)")
                    ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                            color="white" if (T[ri] and Mx[i, j] < 0.6) else "black")
            fig.colorbar(im, ax=ax, label="P(steered | baseline)")
            ax.set_title(f"{model} | {tag} | {subset}", fontsize=9)
            fig.tight_layout()
            name = f"transition_{model}_{tag}_{subset}.png"
            fig.savefig(out_dir / name, dpi=150)
            plt.close(fig)
            figs.append(name)
    log(f"  wrote {len(figs)} figure(s) to {rel_to_root(out_dir)}")
    return figs


TRANSITION_PP_COLUMNS = ["prompt_id", "model", "layer", "direction", "alpha",
                         "alpha_over_sigma", "before", "after", "changed", "cstar_split"]
TRANSITION_TABLE_COLUMNS = ["model", "subset", "condition_tag", "layer", "direction",
                            "alpha", "alpha_over_sigma", "from_class", "to_class", "count",
                            "denom", "rate_pct", "below_min_n"]
# Okabe-Ito: colourblind-safe, and one FIXED colour per transition in every figure, so a
# reader comparing two models or two layers never has to re-read the legend.
TRANSITION_COLOURS = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#0072B2", "#D55E00"]


def off_diagonal_transitions(labels3: list[str]) -> list[tuple[str, str, str]]:
    """(from, to, 'N->D'-style short name) for the six off-diagonal cells, in labels3 order."""
    short = {c: str(c)[:1].upper() for c in labels3}
    if len(set(short.values())) != len(labels3):
        fail(f"labels3={labels3} do not have distinct initials; the transition short "
             f"names (N->D, ...) would collide")
    return [(i, j, f"{short[i]}->{short[j]}") for i in labels3 for j in labels3 if i != j]


def _draw_transition_panel(ax, gl: pd.DataFrame, trans, min_n: int) -> None:
    """Grouped bars of the six off-diagonal transition rates for ONE layer.

    `gl` is the transitions_table slice for one (model, direction, subset, layer). x is
    the non-zero alpha grid with alpha/sigma_l under each tick (sigma differs per layer,
    so the ticks are per panel). A from_class row below min_n is still drawn -- hatched
    and faded -- because hiding it would make an empty baseline class look like "no
    movement"; the hatch says the height rests on a count, not a rate.
    """
    alphas = sorted(a for a in gl.alpha.unique() if a != 0)
    xs = np.arange(len(alphas))
    width = 0.8 / len(trans)
    for t, (i_c, j_c, _name) in enumerate(trans):
        cell = (gl[(gl.from_class == i_c) & (gl.to_class == j_c)]
                .set_index("alpha"))
        for x, a in zip(xs, alphas):
            if a not in cell.index:
                continue
            r = cell.loc[a]
            h = 0.0 if pd.isna(r.rate_pct) else float(r.rate_pct)
            low = bool(r.below_min_n)
            ax.bar(x - 0.4 + (t + 0.5) * width, h, width,
                   color=TRANSITION_COLOURS[t], alpha=0.35 if low else 1.0,
                   hatch="///" if low else None,
                   edgecolor="0.3" if low else TRANSITION_COLOURS[t],
                   linewidth=0.4)
    ticks = []
    for a in alphas:
        aos = gl[gl.alpha == a].alpha_over_sigma.iloc[0]
        ticks.append(f"{a:+g}\n({aos:.2f})" if pd.notna(aos) else f"{a:+g}\n(n/a)")
    ax.set_xticks(xs, ticks, fontsize=6)
    ax.grid(axis="y", color="0.9", lw=0.6)
    ax.set_axisbelow(True)


def _transition_legend(fig, trans, min_n: int) -> None:
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=TRANSITION_COLOURS[t], label=name)
               for t, (_i, _j, name) in enumerate(trans)]
    handles.append(Patch(facecolor="white", edgecolor="0.3", hatch="///",
                         label=f"hatched: baseline n < {min_n} (count, not a rate)"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=7,
               frameon=False)


def make_transition_bars(cfg, tt: pd.DataFrame, out_dir: Path, plt, log) -> list[str]:
    """Grouped bars of the six off-diagonal transition rates, one panel per layer.

    One figure per (model, direction, subset), covering EVERY steered layer (all 10).
    These are the appendix figures; the main-paper figure is make_main_transition_bars.
    """
    labels3 = list(cfg["study"]["labels3"])
    min_n = int(cfg["study"].get("min_n", 30))
    trans = off_diagonal_transitions(labels3)
    figs = []
    if tt.empty:
        return figs
    for (model, dname, subset), g in tt.groupby(["model", "direction", "subset"],
                                                sort=True):
        layers = sorted(int(l) for l in g.layer.unique())
        ncol = min(5, len(layers))
        nrow = int(np.ceil(len(layers) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.0 * nrow + 0.9),
                                 sharey=True, squeeze=False)
        for k, ax in enumerate(axes.flat):
            if k >= len(layers):
                ax.axis("off")
                continue
            l = layers[k]
            _draw_transition_panel(ax, g[g.layer == l], trans, min_n)
            ax.set_title(f"L{l}", fontsize=9)
            if k % ncol == 0:
                ax.set_ylabel("transition rate (%)")
            if k // ncol == nrow - 1:
                ax.set_xlabel(r"$\alpha$  ($\alpha/\sigma_l$)", fontsize=8)
        _transition_legend(fig, trans, min_n)
        fig.suptitle(f"{model} | {dname} | {subset} | 100 x count(i -> j) / "
                     f"count(i at baseline)", fontsize=10)
        fig.tight_layout(rect=(0, 0.06, 1, 0.97))
        name = f"transitions_bar_{model}_{dname}_{subset}.png"
        fig.savefig(out_dir / name, dpi=150)
        plt.close(fig)
        figs.append(name)
    return figs


def rank_main_figure_layers(cfg, df: pd.DataFrame, log) -> dict:
    """Top-N of the steered layers per (model, direction) by PEAK target-class pp.

    Peak pp = the largest target-class pp over the non-zero alphas at that layer (d_D ->
    DISCLAIMER, d_R -> REFUSAL), on steering.main_figure.rank_subset (the full Steering
    split). Ties break toward the shallower layer. This chooses FIGURE PANELS only: every
    metric, table and appendix figure already covers all steered layers. The same
    ranking is reused for the clean-subset figure, so both show the same layers.
    """
    mcfg = (cfg.get("steering") or {}).get("main_figure") or {}
    top_n = int(mcfg.get("top_n_layers", 5))
    rank_subset = str(mcfg.get("rank_subset", "full"))
    out = {"top_n_layers": top_n, "rank_subset": rank_subset,
           "criterion": "max over non-zero alphas of the target-class pp "
                        "(dD -> DISCLAIMER, dR -> REFUSAL); ties -> shallower layer",
           "note": "Figure panels only. The layers are chosen on the same prompts they "
                   "are plotted on, so their pp is optimistic: caption them as 'top "
                   f"{top_n} by peak pp', not as typical layers.",
           "models": {}}
    if df.empty:
        return out
    for (model, dname), g in df[df.subset == rank_subset].groupby(
            ["model", "direction"], sort=True):
        tc = target_class(dname)
        cur = g[g["class"] == tc]
        if cur.empty:
            continue
        peak = cur.loc[cur.groupby("layer").pp.idxmax()]      # one row per layer
        peak = peak.sort_values(["pp", "layer"], ascending=[False, True])
        rows = []
        for k, r in enumerate(peak.itertuples(index=False)):
            aos = r.alpha_over_sigma
            rows.append({"rank": k + 1, "layer": int(r.layer), "peak_pp": float(r.pp),
                         "ci_lo": float(r.ci_lo), "ci_hi": float(r.ci_hi),
                         "alpha_at_peak": float(r.alpha),
                         "alpha_over_sigma_at_peak": None if aos is None or pd.isna(aos)
                                                     else float(aos),
                         "condition_tag": str(r.condition_tag)})
        chosen = [x["layer"] for x in rows[:top_n]]
        out["models"].setdefault(str(model), {})[str(dname)] = {
            "target_class": tc, "n_layers_ranked": len(rows), "layers": chosen,
            "ranking": rows}
        if len(rows) < top_n:
            log(f"  !! main figure {model} {dname}: only {len(rows)} steered layer(s) "
                f"scored, fewer than top_n_layers={top_n}")
        log(f"  main figure {model} {dname}: top {top_n} by peak {tc} pp on "
            f"{rank_subset} -> " + ", ".join(f"L{x['layer']} {x['peak_pp']:+.1f}"
                                             for x in rows[:top_n]))
    return out


def make_main_transition_bars(cfg, tt: pd.DataFrame, ranking: dict, out_dir: Path,
                              plt, log) -> list[str]:
    """The main-paper transition bar graph: rows d_R / d_D x that direction's top-N layers.

    One figure per (model, subset). Panels are in rank order (#1 = largest peak pp), each
    drawn exactly as in the appendix figures, and y is shared across the whole figure so
    d_R and d_D rates compare directly.
    """
    labels3 = list(cfg["study"]["labels3"])
    min_n = int(cfg["study"].get("min_n", 30))
    trans = off_diagonal_transitions(labels3)
    top_n = int(ranking["top_n_layers"])
    figs = []
    if tt.empty:
        return figs
    for model, per_dir in ranking["models"].items():
        dnames = [d for d in S.DIRECTION_NAMES if d in per_dir]
        if not dnames:
            continue
        for subset in sorted(tt[tt.model == model].subset.unique()):
            g_m = tt[(tt.model == model) & (tt.subset == subset)]
            fig, axes = plt.subplots(len(dnames), top_n,
                                     figsize=(3.6 * top_n, 3.0 * len(dnames) + 1.0),
                                     sharey=True, squeeze=False)
            for r_i, dname in enumerate(dnames):
                rec = per_dir[dname]
                by_layer = {x["layer"]: x for x in rec["ranking"]}
                for k in range(top_n):
                    ax = axes[r_i, k]
                    gl = (g_m[(g_m.direction == dname) & (g_m.layer == rec["layers"][k])]
                          if k < len(rec["layers"]) else g_m.iloc[:0])
                    if gl.empty:
                        ax.axis("off")
                        continue
                    l = rec["layers"][k]
                    _draw_transition_panel(ax, gl, trans, min_n)
                    ax.set_title(f"{dname} L{l}  #{k + 1}  peak "
                                 f"{rec['target_class'][:1]} "
                                 f"{by_layer[l]['peak_pp']:+.1f}pp "
                                 f"({ranking['rank_subset']})", fontsize=8)
                    if k == 0:
                        ax.set_ylabel(f"{dname}\ntransition rate (%)")
                    if r_i == len(dnames) - 1:
                        ax.set_xlabel(r"$\alpha$  ($\alpha/\sigma_l$)", fontsize=8)
            _transition_legend(fig, trans, min_n)
            fig.suptitle(f"{model} | {subset} | top {top_n} steered layers per direction "
                         f"by peak target-class pp | 100 x count(i -> j) / "
                         f"count(i at baseline)", fontsize=10)
            fig.tight_layout(rect=(0, 0.06, 1, 0.96))
            name = f"transitions_bar_main_{model}_{subset}.png"
            fig.savefig(out_dir / name, dpi=150)
            plt.close(fig)
            figs.append(name)
    return figs


def write_summary_md(cfg, report, df: pd.DataFrame, dest: Path, log,
                     bar_figs: list[str] | None = None,
                     main_figs: list[str] | None = None,
                     ranking: dict | None = None) -> None:
    labels3 = list(cfg["study"]["labels3"])
    L = []
    L.append("# Step 2 -- refusal/disclaimer directions and causal steering")
    L.append("")
    L.append(f"Generated {report['written_utc']} | seed {report['seed']} | "
             f"metrics.json sha256 `{report.get('sha256', '')[:16]}`")
    L.append("")
    L.append("All effects are percentage-point (pp) changes in the C* class share "
             "relative to the shared alpha=0 baseline, on the ARGMAX LABEL only. "
             f"CIs are {int(report['bootstrap']['ci'] * 100)}% percentile intervals from "
             f"a paired bootstrap over prompt ids "
             f"(B={report['bootstrap']['B']}, one shared resample index).")
    L.append("")
    L.append("**Dual reporting.** Every headline appears twice: on the full Steering "
             "split and on the C*-test-clean subset. 1,223 of the 1,750 prompts are in "
             "C*'s own training split, so the clean subset is the PRE-REGISTERED "
             "TIEBREAKER -- where the two disagree, the clean number stands.")
    L.append("")
    for model, mrep in report["models"].items():
        L.append(f"## {model}")
        L.append("")
        for subset, srep in mrep["subsets"].items():
            L.append(f"### {subset} (n={srep['n']})")
            L.append("")
            L.append("Baseline: " + ", ".join(
                f"{c} {srep['baseline_pct'][c]:.1f}%" for c in labels3)
                + f" | median {srep['baseline_median_tokens']:.0f} tokens"
                f" | truncated {100 * srep['baseline_truncation_rate']:.1f}%")
            L.append("")
            sub = df[(df.model == model) & (df.subset == subset)]
            if sub.empty:
                L.append("_no scored conditions_")
                L.append("")
                continue
            L.append("| direction | best cell | alpha | alpha/sigma | target class | "
                     "pp | 95% CI | d trunc (pp) | d median tokens |")
            L.append("|---|---|---|---|---|---|---|---|---|")
            for dname in sorted(sub.direction.dropna().unique()):
                tc = target_class(dname)
                cur = sub[(sub.direction == dname) & (sub["class"] == tc)]
                if cur.empty:
                    continue
                best = cur.loc[cur.pp.idxmax()]
                rec = srep["conditions"][best.condition_tag]
                aos = "n/a" if best.alpha_over_sigma is None or pd.isna(
                    best.alpha_over_sigma) else f"{best.alpha_over_sigma:.2f}"
                L.append(
                    f"| {dname} | `{best.condition_tag}` (L{int(best.layer)}) | "
                    f"{best.alpha:g} | {aos} | {tc} | {best.pp:+.1f} | "
                    f"[{best.ci_lo:+.1f}, {best.ci_hi:+.1f}] | "
                    f"{rec['delta_truncation_rate_pp']['pp']:+.1f} | "
                    f"{rec['delta_median_tokens']['delta']:+.0f} |")
            L.append("")
            flagged = sorted({t for t, v in srep["conditions"].items()
                              if v["transition"]["rows_below_min_n"]})
            if flagged:
                L.append(f"_Transition rows below min_n={report['min_n']} (reported as "
                         f"counts, not rates) in {len(flagged)} condition(s), e.g. "
                         f"`{flagged[0]}`: "
                         f"{srep['conditions'][flagged[0]]['transition']['rows_below_min_n']}._")
                L.append("")
    if ranking and ranking.get("models"):
        n_top = ranking["top_n_layers"]
        L.append(f"## Main-paper transition figure: top {n_top} layers by peak pp")
        L.append("")
        L.append(f"Ranked per direction on the `{ranking['rank_subset']}` subset by "
                 f"{ranking['criterion']}. Figure panels only -- every table here covers "
                 f"all steered layers. {ranking['note']}")
        L.append("")
        L.append("| model | direction | rank | layer | peak pp | 95% CI | alpha | "
                 "alpha/sigma |")
        L.append("|---|---|---|---|---|---|---|---|")
        for model, per_dir in ranking["models"].items():
            for dname, rec in per_dir.items():
                for x in rec["ranking"][:n_top]:
                    aos = x["alpha_over_sigma_at_peak"]
                    L.append(f"| {model} | {dname} ({rec['target_class']}) | {x['rank']} | "
                             f"L{x['layer']} | {x['peak_pp']:+.1f} | "
                             f"[{x['ci_lo']:+.1f}, {x['ci_hi']:+.1f}] | "
                             f"{x['alpha_at_peak']:g} | "
                             f"{'n/a' if aos is None else f'{aos:.2f}'} |")
        L.append("")
    L.append("## Figures")
    L.append("")
    for f in main_figs or []:
        L.append(f"- `{f}` -- MAIN PAPER: rows dR / dD x the top "
                 f"{(ranking or {}).get('top_n_layers', '?')} layers by peak pp")
    for f in report.get("figures", []):
        L.append(f"- `{f}`")
    for f in bar_figs or []:
        L.append(f"- `{f}` -- off-diagonal transition rates (%), one panel per steered "
                 f"layer (all of them; appendix)")
    L.append("")
    L.append("## Artifacts")
    L.append("")
    L.append("- `metrics.json` -- every cell, every CI, every transition matrix")
    L.append("- `pp_table.parquet` -- long-format pp table (one row per "
             "model x subset x condition x class)")
    L.append("- `main_figure_layers.json` -- the full per-direction layer ranking by peak "
             "pp behind the main-paper figure (all steered layers, not just the top N)")
    L.append("- `transitions_per_prompt.parquet` -- C* label before (shared alpha=0 "
             "baseline) and after, one row per model x steered condition x prompt on the "
             "full Steering split; filter `cstar_split == 'test'` for the clean subset")
    L.append("- `transitions_table.parquet` / `.csv` -- long-format 3x3 transition table "
             "(one row per model x subset x condition x from x to, diagonal included); "
             f"rate_pct = 100 x count / denom, flagged below_min_n when denom < "
             f"{report['min_n']}")
    dest.write_text("\n".join(L) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# --dry-run
# --------------------------------------------------------------------------- #
def dry_run(cfg, args, ctxs, splits, log) -> None:
    scfg = cfg["steering"]
    batch_size = int(args.batch_size or scfg.get("batch_size", 8))
    alphas = nonzero_alphas(cfg)
    dnames = direction_names(cfg, args)

    planned_per_model = (PLANNED_PROMPTS * PLANNED_LAYERS * PLANNED_DIRECTIONS
                         * PLANNED_NONZERO_ALPHAS) + PLANNED_PROMPTS
    planned_total = planned_per_model * PLANNED_ROSTER
    log("\n" + "=" * 78)
    log("PLANNED BUDGET (the locked grid, 3-model roster)")
    log(f"  {PLANNED_PROMPTS} steering prompts x {PLANNED_LAYERS} layers x "
        f"{PLANNED_DIRECTIONS} directions x {PLANNED_NONZERO_ALPHAS} non-zero alphas")
    log(f"  = {PLANNED_PROMPTS * PLANNED_LAYERS * PLANNED_DIRECTIONS * PLANNED_NONZERO_ALPHAS:,}"
        f" + {PLANNED_PROMPTS} baseline = {planned_per_model:,} per model")
    log(f"  x {PLANNED_ROSTER} models = {planned_total:,} generations")
    log(f"  {S.estimate_gpu_hours(planned_total):,.0f} GPU-h at batch-1 / "
        f"{S.estimate_gpu_hours(planned_total, batch_speedup=PLANNED_BATCH_SPEEDUP):,.0f} "
        f"GPU-h at batch-8 "
        f"({S.estimate_gpu_hours(planned_total, batch_speedup=PLANNED_BATCH_SPEEDUP) / 4 / 24:.1f}"
        f" days on 4x A100-40GB)")

    n_steer = len(splits["splits"]["steering"])
    if args.limit:
        n_steer = min(n_steer, args.limit)
    if cfg.get("_smoke"):
        n_steer = min(n_steer, int(cfg["_smoke"]["n_prompts"]))
    log("")
    log(f"ACTUAL INVENTORY (this config, this split, --batch-size {batch_size})")
    log(f"  steering split n={n_steer}  alphas={alphas}  directions={dnames}")
    grand = 0
    for ctx in ctxs:
        try:
            layers = load_selected_layers(cfg, ctx.slug)
        except SystemExit:
            layers = []
            log(f"  {ctx.slug}: no selected layers yet (run --stage probe); "
                f"assuming {PLANNED_LAYERS}")
        n_layers = len(layers) or PLANNED_LAYERS
        conds = build_conditions(cfg, args, layers or list(range(1, n_layers + 1)))
        n_gen = len(conds) * n_steer
        grand += n_gen
        mark = "  [NOT YET GENERATED BY STEP 1]" if ctx.placeholder else ""
        log(f"  {ctx.slug:<24} layers={layers or '(assumed %d)' % PLANNED_LAYERS}{mark}")
        log(f"  {'':<24} {len(conds)} conditions "
            f"(1 baseline + {n_layers} x {len(dnames)} x {len(alphas)}) "
            f"x {n_steer} prompts = {n_gen:,} generations")
        log(f"  {'':<24} {S.estimate_gpu_hours(n_gen):,.1f} GPU-h at batch-1 / "
            f"{S.estimate_gpu_hours(n_gen, batch_speedup=float(batch_size) / 2):,.1f} "
            f"GPU-h at the chosen batch size (assuming a {batch_size / 2:.1f}x speedup)")
    log(f"  TOTAL across {len(ctxs)} configured model(s): {grand:,} generations, "
        f"{S.estimate_gpu_hours(grand):,.0f} GPU-h at batch-1")
    if len(ctxs) < PLANNED_ROSTER:
        log(f"  NOTE: the roster in this config has {len(ctxs)} model(s); the third is "
            f"still a TODO(roster) placeholder in study.yaml. The planned budget above "
            f"is the 3-model number.")
    if args.shard:
        log(f"  --shard {args.shard} selects {len(build_conditions(cfg, args, list(range(1, 11))))} "
            f"condition(s) per model in this task")
    log("=" * 78 + "\n")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Step 2: directions, layer selection, steering sweep, C* scoring, pp metrics")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--stage", default="all", choices=STAGES)
    ap.add_argument("--models", default=None,
                    help="comma-separated slugs; default is the whole configured roster")
    ap.add_argument("--seed", type=int, default=None, help="default: study.seed")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap the prompts per split (smoke runs only)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the cell inventory and the GPU-h estimate, then exit")
    ap.add_argument("--verify", action="store_true",
                    help="recompute this stage's hash and exit non-zero on drift")
    ap.add_argument("--force", action="store_true", help="redo completed work")
    ap.add_argument("--smoke", action="store_true", help="use the config's smoke block")
    # directions
    ap.add_argument("--n-halves", type=int, default=50,
                    help="split-half reliability draws (directions stage)")
    ap.add_argument("--reliability-draws", type=int, default=5,
                    help="balanced draws inside each reliability half; the full K=50 is "
                         "used for the committed direction itself")
    ap.add_argument("--control", default="off", choices=("off", "random-unit"),
                    help="random-direction control; OFF by default and generating nothing")
    ap.add_argument("--control-draws", type=int, default=5)
    # probe
    ap.add_argument("--top-k", type=int, default=0, help="default: probe.top_k (10)")
    # preflight
    ap.add_argument("--n-probe", type=int, default=4,
                    help="prompts checked by the hook/capture gate")
    # steer
    ap.add_argument("--batch-size", type=int, default=0, help="default: steering.batch_size")
    ap.add_argument("--conditions", default=None, help="comma-separated condition tags")
    ap.add_argument("--layers", default=None, help="comma-separated layer rows")
    ap.add_argument("--alphas", default=None, help="comma-separated alphas")
    ap.add_argument("--shard", default=None, metavar="i/n",
                    help="split the condition list across SLURM array tasks")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="run the sweep WITHOUT the hook/capture gate (not recommended)")
    # score
    ap.add_argument("--labels", default=None, help="override the C* label table path")
    ap.add_argument("--token-budget", type=int, default=16384)
    ap.add_argument("--shard-rows", type=int, default=0, help="default: scoring.shard_rows")
    ap.add_argument("--skip-val-parity", action="store_true")
    # metrics
    ap.add_argument("--bootstrap", type=int, default=0, help="default: study.bootstrap.B")
    ap.add_argument("--no-boot-length", dest="boot_length", action="store_false",
                    help="skip the CI on delta-median-length (the one slow bootstrap)")
    ap.set_defaults(boot_length=True)
    args = ap.parse_args()

    stages = [s for s in STAGES if s != "all"] if args.stage == "all" else [args.stage]
    # A smoke run must never overwrite the real report directory. cfg is not loaded yet
    # here (the log file needs a directory first), so key off the flag.
    out_dir = G.repo_path("results/reports/step2_smoke" if args.smoke
                          else "results/reports/step2")
    log = Tee(out_dir / f"step2_{args.stage}.log")
    try:
        cfg = load_cfg(args.config, args.smoke, log)
        if args.seed is None:
            args.seed = int(cfg["study"]["seed"])
        if args.control == "off" and (cfg["steering"].get("control") or {}).get("enabled"):
            args.control = str(cfg["steering"]["control"].get("kind", "random-unit"))
        only = [s.strip() for s in args.models.split(",")] if args.models else None

        log(f"config={args.config} stage={args.stage} seed={args.seed} "
            f"models={only or 'all'} limit={args.limit or 0} control={args.control}")
        ctxs = model_contexts(cfg, only, log, planning=args.dry_run)
        splits = load_study_splits(cfg, log)

        if args.dry_run:
            dry_run(cfg, args, ctxs, splits, log)
            return

        labels = None
        if any(s in stages for s in ("directions", "probe", "metrics")) and not (
                args.verify and "probe" in stages and len(stages) == 1):
            labels = load_labels(cfg, args.labels, log)

        for stage in stages:
            log(f"\n########## stage: {stage} ##########")
            if stage == "directions":
                stage_directions(cfg, args, ctxs, splits, labels, log)
            elif stage == "probe":
                stage_probe(cfg, args, ctxs, splits, labels, log)
            elif stage == "preflight":
                stage_preflight(cfg, args, ctxs, splits, log)
            elif stage == "steer":
                stage_steer(cfg, args, ctxs, splits, log)
            elif stage == "score":
                stage_score(cfg, args, ctxs, log)
            elif stage == "metrics":
                stage_metrics(cfg, args, ctxs, splits, labels, log)
        log("\ndone")
    finally:
        log.close()


if __name__ == "__main__":
    main()
