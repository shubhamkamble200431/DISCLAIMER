"""Step 3 of plan.md -- layer-wise direction geometry: is d_D the same axis as d_R?

The PDF asks for: "construct layer-wise disclaimer and refusal directions (for all
layers), measure their cosine similarity S_i = cos(dD_i, dR_i), plot S_i vs layers".
This script is the whole of that, plus the controls the number is worthless without. CPU only, no GPU, no generation: it runs entirely on the `prompt_last`
activations step 1 already wrote, and finishes in minutes.

WHAT IS COMPUTED, PER MODEL, PER LAYER ROW 1..n_blocks-1

  S            cos(d_D, d_R) with both directions estimated on the FULL 40% Direction
               split -- the same vectors step 2 steers with. No cross-fitting (user
               decision 2026-09-22): the full direction set is used for all geometry.
               d_R = unit(mu_R - mu_N) and d_D = unit(mu_D - mu_N) share mu_N, so
               sampling noise in mu_N enters both with the same sign and pushes S UP a
               little (measured earlier at ~+0.055 at L4, ~+0.02 from L12 up). The null
               band below is built with the SAME shared-mu_N estimator, so it carries
               the same inflation and the comparison against it stays fair.

  null band    the SAME full-split estimator with the class labels permuted within the
               Direction split, --n-null draws, 95 percent percentile band, plus a
               one-sided permutation p-value P(S_null >= S). A layer whose S falls
               inside the band has not shown that d_D and d_R are aligned at all -- in
               the early layers they usually have not. Those layers are marked on every
               plot and printed by name.

  cos(d_R, d_len), cos(d_D, d_len)
               d_len is a pure LENGTH axis built from NORMAL answers only: the prompts
               whose answers hit the generation cap minus the shortest 30 percent.
               Answer length is a first-order axis at `prompt_last` (the capture point
               is the last prompt token, where the model has already decided roughly how
               much it is about to say), and mid-layer d_D is substantially that axis.
               A reviewer will ask; the paper needs the number. Full Direction split.

  ||mu_D - mu_N|| / ||mu_R - mu_N||
               how far the disclaimer centroid sits from normal relative to refusal.
               A high cosine with a small norm ratio is a different story from a high
               cosine with a ratio near 1.

  cstar_cos_centroid_DR
               refusal_classifier src/metrics.py::label_geometry's cos(centroid_D,
               centroid_R). This is a DIFFERENT statistic: raw centroids, not centroid
               DIFFERENCES from NORMAL. On residual-stream activations it is dominated
               by the shared mean vector and runs near 1.0 at every layer. It is
               reported for continuity with the classifier-side report and must never be
               quoted as S_i.

MEASURED FACTS THIS SCRIPT OBEYS (violating any of them silently corrupts the figure)
  * Saved row l is the OUTPUT of decoder block l-1. Row 0 is inputs_embeds and is
    byte-identical across prompts at this capture point, so mu_R - mu_N is the zero
    vector there and the cosine is 0/0 -> NaN. Row n_blocks is post-final-norm, not a
    residual site and not steerable. Rows 0 and n_blocks are asserted out and skipped
    with an explicit log line, never quietly dropped.
  * Directions use the SAME balanced difference-of-means as step 2 (subsample NORMAL to
    min(n_R, n_D), K seeded draws, average the UNIT vectors, renormalise), with K and
    the seed read from configs/study.yaml, so step 2's steering vectors and step 3's
    geometry are the same objects.
  * Labels are each model's OWN C* label. 1,223 of the 1,750 prompts are in C*'s own
    training split, so S is additionally reported on the cstar_split == 'test' subset,
    under the project min-n rule (M.MIN_N = 30).

OUTPUTS (results/reports/step3/)
  geometry.json      per-layer records + input inventory + seed + sha256
  geometry.parquet   the same records, flat
  step3_<slug>.png/.pdf, step3_combined.png/.pdf
  summary.md         what the curve means for the central hypothesis, in words
  compute.log / plot.log

Usage:
  python -X utf8 scripts/step3_geometry.py --stage all
  python -X utf8 scripts/step3_geometry.py --stage compute --models qwen3-5-9b
  python -X utf8 scripts/step3_geometry.py --stage plot
  python -X utf8 scripts/step3_geometry.py --dry-run
  python -X utf8 scripts/step3_geometry.py --verify
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import generate as G  # noqa: E402
from src import steer as S  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs" / "study.yaml"
STUDY_DIR = ROOT / "data" / "study"
OUT_DIR = ROOT / "results" / "reports" / "step3"
POINT = "prompt_last"                     # fact 3: the only capture point in this study
LABELS3 = ["NORMAL", "DISCLAIMER", "REFUSAL"]
FALLBACK_MIN_N = 30                       # used only if src.metrics cannot be imported

# Candidate label-table column names, in resolution order. C* is the Longformer run
# (locked decision), so `longformer_label` wins over any generic `label` column.
LABEL_COLS = ["longformer_label", "cstar_label", "label", "pred_label"]


class Tee:
    """stdout + <out_dir>/<stage>.log, same shape as the classifier's Tee."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()


# --------------------------------------------------------------------------- #
# seeding
# --------------------------------------------------------------------------- #
def seed_for(base: int, *parts) -> int:
    """Deterministic per-(model, layer, purpose) seed.

    Derived from a hash rather than `base + l` so that adding a model, reordering the
    roster or skipping a layer cannot change any other cell's numbers -- a re-run with
    --models qwen3-5-9b must reproduce the full-roster row for that model exactly.
    """
    h = hashlib.blake2b(("|".join([str(base)] + [str(p) for p in parts])).encode(),
                        digest_size=8)
    return int.from_bytes(h.digest(), "big") % (2 ** 32)


# --------------------------------------------------------------------------- #
# optional import of the classifier's metrics module
# --------------------------------------------------------------------------- #
def load_cstar_metrics(repo: Path, log):
    """Load refusal_classifier/src/metrics.py under its own module name.

    It cannot be imported as `src.metrics`: this process already has the GENERATION
    repo's `src` package bound, and the two repos both call their package `src`. The
    module is self-contained (json / pathlib / numpy; sklearn is imported lazily inside
    the functions it is needed in), so loading it by file path is safe in the generation
    env even though the classifier env has a different transformers pin.
    """
    path = Path(repo) / "src" / "metrics.py"
    if not path.exists():
        log(f"  WARNING metrics module not found at {path.as_posix()}; "
            f"label_geometry columns will be null and min_n falls back to "
            f"{FALLBACK_MIN_N}")
        return None
    try:
        spec = importlib.util.spec_from_file_location("cstar_metrics", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as e:                                   # noqa: BLE001
        log(f"  WARNING could not import {path.as_posix()}: {e!r}; "
            f"label_geometry columns will be null")
        return None
    if list(mod.LABELS3) != LABELS3:
        raise RuntimeError(f"metrics.LABELS3 {mod.LABELS3} != study labels {LABELS3}; "
                           f"the label order is not shared, refusing to continue")
    log(f"  loaded C* metrics from {path.as_posix()}  MIN_N={mod.MIN_N}")
    return mod


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #
def read_manifest(model_root: Path) -> dict:
    p = model_root / "manifest.json"
    if not p.exists():
        raise FileNotFoundError(
            f"no manifest at {p.as_posix()} -- step 1 has not run for this model. "
            f"This script never invents activations.")
    return json.loads(p.read_text(encoding="utf-8"))


def load_stack(model_root: Path, prompt_ids, n_rows: int, log) -> tuple[np.ndarray, list[int]]:
    """[n_kept, n_rows, H] float32 for one model, over the requested prompts.

    Reads through G.load_hidden, so both the `stacked` (<pid>.npz, the study layout) and
    the legacy `per_layer` (<pid>/*.npy) trees work unchanged. Prompts with no
    activations on disk are skipped and counted, never zero-filled.
    """
    hid = model_root / "hidden"
    if not hid.exists():
        raise FileNotFoundError(f"no hidden/ dir at {hid.as_posix()}")
    mats, kept, missing = [], [], 0
    t0 = time.perf_counter()
    for i, pid in enumerate(prompt_ids):
        folder = hid / str(pid)
        path = folder if folder.is_dir() else hid / f"{pid}.npz"
        if not path.exists():
            missing += 1
            continue
        arr, layers = G.load_hidden(path, POINT)
        if arr.shape[0] != n_rows:
            raise RuntimeError(
                f"prompt {pid}: {arr.shape[0]} layer rows on disk, manifest says "
                f"{n_rows}. Mixed runs in one tree -- refusing to average across them.")
        if int(layers[0]) != 0 or int(layers[-1]) != n_rows - 1:
            raise RuntimeError(f"prompt {pid}: layer index {layers[0]}..{layers[-1]} is "
                               f"not a full 0..{n_rows - 1} range")
        mats.append(arr.astype(np.float32, copy=False))
        kept.append(int(pid))
        if (i + 1) % 200 == 0:
            log(f"      {i + 1}/{len(prompt_ids)}  {time.perf_counter() - t0:.0f}s")
    if not mats:
        raise RuntimeError(f"no activations found under {hid.as_posix()}")
    X = np.stack(mats)
    log(f"      loaded {len(kept)} prompts, missing {missing}, "
        f"{X.nbytes / 1e6:.0f} MB, {time.perf_counter() - t0:.0f}s")
    return X, kept


def resolve_label_table(cfg: dict, slug: str, override: str | None, log) -> pd.DataFrame:
    """prompt_id -> C* label (+ answer length + C*'s own split), for ONE model.

    Resolution order: --labels, then step 1's per-model study table, then the committed
    classifier predictions parquet. Whichever is used is logged, because a silent switch
    of label source between step 2 and step 3 would make the two steps' directions
    incomparable without anything in the output saying so.
    """
    clf_repo = G.repo_path(cfg["study"].get("classifier_repo", "../refusal_classifier"))
    cands = []
    if override:
        cands.append(Path(override))
    cands += [
        STUDY_DIR / "labels" / f"{slug}.parquet",
        STUDY_DIR / "step1_labels.parquet",
        STUDY_DIR / "labels.parquet",
        clf_repo / "results" / "predictions" / "generated_predictions.parquet",
    ]
    src = next((p for p in cands if p.exists()), None)
    if src is None:
        raise FileNotFoundError(
            "no label table found. Tried:\n  "
            + "\n  ".join(p.as_posix() for p in cands)
            + "\nRun step 1's scoring stage, or pass --labels PATH.")

    df = pd.read_parquet(src)
    if "gen_model" in df.columns:
        df = df[df.gen_model.astype(str) == slug]
        if df.empty:
            raise RuntimeError(f"{src.as_posix()} has no rows with gen_model == '{slug}' "
                               f"(has {sorted(set(pd.read_parquet(src).gen_model))})")
    col = next((c for c in LABEL_COLS if c in df.columns), None)
    if col is None:
        raise RuntimeError(f"{src.as_posix()} has none of {LABEL_COLS}; "
                           f"columns are {list(df.columns)}")
    out = pd.DataFrame({"prompt_id": df["prompt_id"].astype(np.int64),
                        "label": df[col].astype(str)})
    for extra, dest in (("n_generated_tokens", "n_generated_tokens"),
                        ("hit_max_new_tokens", "hit_max_new_tokens"),
                        ("split", "cstar_split")):
        out[dest] = df[extra].values if extra in df.columns else None
    bad = sorted(set(out.label) - set(LABELS3))
    if bad:
        raise RuntimeError(f"unexpected labels {bad} in {src.as_posix()}; "
                           f"expected exactly {LABELS3}")
    out = out.drop_duplicates(subset=["prompt_id"]).set_index("prompt_id")
    log(f"  labels: {src.as_posix()}  column='{col}'  n={len(out)}")
    return out


def fill_lengths_from_answers(model_root: Path, tbl: pd.DataFrame, log) -> pd.DataFrame:
    """Backfill n_generated_tokens / hit_max_new_tokens from answers/<pid>.json.

    Only needed when the label table is a bare (prompt_id, label) file. The per-prompt
    JSON is written by G.write_prompt, so both keys are always present there.
    """
    if tbl["n_generated_tokens"].notna().all() and tbl["hit_max_new_tokens"].notna().all():
        return tbl
    ans = model_root / "answers"
    if not ans.exists():
        log(f"  WARNING no answers/ dir at {ans.as_posix()}; d_len cannot be built")
        return tbl
    n, tok, cap = 0, {}, {}
    for pid in tbl.index:
        p = ans / f"{pid}.json"
        if not p.exists():
            continue
        rec = json.loads(p.read_text(encoding="utf-8"))
        tok[pid] = rec.get("n_generated_tokens")
        cap[pid] = rec.get("hit_max_new_tokens")
        n += 1
    tbl["n_generated_tokens"] = tbl.index.map(tok)
    tbl["hit_max_new_tokens"] = tbl.index.map(cap)
    log(f"  backfilled answer lengths from {n} answers/*.json")
    return tbl


def load_selected_layers(slug: str, log) -> list[int] | None:
    p = STUDY_DIR / "selected_layers.json"
    if not p.exists():
        log(f"  selected_layers.json absent -- plots will carry no layer marks")
        return None
    sel = json.loads(p.read_text(encoding="utf-8")).get("selected", {}).get(slug)
    if not sel:
        log(f"  selected_layers.json has no entry for '{slug}' -- no layer marks")
        return None
    log(f"  selected layers for {slug}: {sorted(int(x) for x in sel)}")
    return sorted(int(x) for x in sel)


def load_step2_directions(path: Path, hidden: int, log) -> dict[tuple[str, int], np.ndarray] | None:
    """Step 2's stored directions, if they exist, as {(name, layer): unit vector}.

    Two layouts are accepted, because step 2 owns that file and this script must not
    dictate its internals:
      A  keys 'dR' / 'dD' as [n_layers, H] plus an int 'layers' index array
      B  per-layer keys 'dR_L007' / 'dD_7'
    Anything else is reported and treated as absent.
    """
    if not path.exists():
        log(f"  no step-2 directions at {path.as_posix()} -- recomputing from activations")
        return None
    out: dict[tuple[str, int], np.ndarray] = {}
    with np.load(path) as z:
        keys = list(z.keys())
        if "layers" in keys and all(n in keys for n in S.DIRECTION_NAMES):
            layers = [int(v) for v in z["layers"]]
            for name in S.DIRECTION_NAMES:
                a = np.asarray(z[name])
                if a.ndim != 2 or a.shape[0] != len(layers):
                    log(f"  WARNING {path.name}: '{name}' has shape {a.shape}, expected "
                        f"[{len(layers)}, H] -- ignoring the file")
                    return None
                for i, l in enumerate(layers):
                    out[(name, l)] = a[i]
        else:
            import re
            pat = re.compile(r"^(" + "|".join(S.DIRECTION_NAMES) + r")_L?(\d+)$")
            for k in keys:
                m = pat.match(k)
                if m:
                    out[(m.group(1), int(m.group(2)))] = np.asarray(z[k]).ravel()
    if not out:
        log(f"  WARNING {path.as_posix()} matched no known layout (keys={keys[:8]}) "
            f"-- recomputing from activations")
        return None
    widths = {v.size for v in out.values()}
    if widths != {hidden}:
        log(f"  WARNING {path.name}: direction width {widths} != hidden {hidden} "
            f"-- ignoring the file")
        return None
    log(f"  step-2 directions: {path.as_posix()}  "
        f"{len(out)} vectors over {len({l for _, l in out})} layers")
    return out


# --------------------------------------------------------------------------- #
# estimators
# --------------------------------------------------------------------------- #
def balanced_dim(Xpos: np.ndarray, Xneg: np.ndarray, n_sub: int, draws: int,
                 rng: np.random.Generator) -> tuple[np.ndarray, float] | tuple[None, None]:
    """The study's balanced difference-of-means, shared with step 2.

    Subsample the NEGATIVE class (NORMAL) down to n_sub = min(n_R, n_D), take the UNIT
    difference vector, repeat over `draws` seeded subsamples, average the unit vectors
    and renormalise. Averaging UNIT vectors rather than raw differences is what stops one
    draw with a large norm from dominating; renormalising afterwards is what makes the
    result a direction again.

    Returns (unit direction, mean raw ||mu_pos - mu_neg||). The raw norm is the number
    the ||mu_D - mu_N|| / ||mu_R - mu_N|| ratio is built from, and it is lost by
    normalising, so it is carried out here.
    """
    if len(Xpos) < 2 or len(Xneg) < 2:
        return None, None
    n_sub = int(min(n_sub, len(Xneg)))
    if n_sub < 2:
        return None, None
    mu_pos = Xpos.mean(axis=0, dtype=np.float64)
    full = n_sub >= len(Xneg)
    k = 1 if full else int(draws)          # no subsampling to do -> one deterministic draw
    acc, norms = np.zeros(Xpos.shape[1], dtype=np.float64), []
    for _ in range(k):
        idx = (slice(None) if full
               else rng.choice(len(Xneg), size=n_sub, replace=False))
        diff = mu_pos - Xneg[idx].mean(axis=0, dtype=np.float64)
        n = float(np.linalg.norm(diff))
        if not np.isfinite(n) or n < 1e-12:
            continue                        # row 0 lands here: identical across prompts
        acc += diff / n
        norms.append(n)
    if not norms or float(np.linalg.norm(acc)) < 1e-12:
        return None, None
    return S.unit(acc), float(np.mean(norms))


def dim_pair(X: np.ndarray, lab: np.ndarray, draws: int, rng: np.random.Generator):
    """(d_R, d_D, ||mu_R-mu_N||, ||mu_D-mu_N||) on one pool of rows."""
    XN = X[lab == "NORMAL"]
    XR, XD = X[lab == "REFUSAL"], X[lab == "DISCLAIMER"]
    n_sub = min(len(XR), len(XD))          # balance_target: min_RD
    dR, nR = balanced_dim(XR, XN, n_sub, draws, rng)
    dD, nD = balanced_dim(XD, XN, n_sub, draws, rng)
    return dR, dD, nR, nD


def length_direction(X: np.ndarray, short_mask: np.ndarray, cap_mask: np.ndarray,
                     draws: int, rng: np.random.Generator):
    """d_len = unit(mu_cap - mu_short) over NORMAL answers only, balanced the same way.

    The two groups are 'hit the generation cap' and 'shortest 30 percent', both taken
    from NORMAL answers so that refusal/disclaimer content cannot leak into what is
    meant to be a pure length axis.
    """
    Xc, Xs = X[cap_mask], X[short_mask]
    return balanced_dim(Xc, Xs, min(len(Xc), len(Xs)), draws, rng)


def cos(a, b) -> float | None:
    if a is None or b is None:
        return None
    return float(np.dot(np.asarray(a, np.float64), np.asarray(b, np.float64)))


def null_band_layer(Xl: np.ndarray, lab: np.ndarray, n_null: int, draws: int,
                    rng: np.random.Generator, observed: float | None = None) -> dict:
    """The full-split estimator under labels permuted WITHIN the Direction split.

    Class counts are preserved (it is a permutation, not a resample), so the null keeps
    the real n_R / n_D imbalance and the real balanced-subsample size. Under it d_R and
    d_D carry no class signal and share only mu_N -- so the band sits slightly ABOVE zero
    by exactly the shared-mu_N inflation the observed S also carries, which is what makes
    S-vs-band the right comparison. `observed` adds the one-sided permutation p-value
    P(S_null >= S), with the +1 correction so it is never 0.
    """
    vals = []
    for _ in range(int(n_null)):
        perm = rng.permutation(lab)
        dR, dD, _, _ = dim_pair(Xl, perm, draws, rng)
        c = cos(dR, dD)
        if c is not None:
            vals.append(c)
    if not vals:
        return {"mean": None, "lo": None, "hi": None, "n_draws": 0, "p_perm": None}
    a = np.asarray(vals, dtype=np.float64)
    p = (None if observed is None
         else float((1 + int((a >= observed).sum())) / (1 + len(a))))
    return {"mean": float(a.mean()), "lo": float(np.percentile(a, 2.5)),
            "hi": float(np.percentile(a, 97.5)), "n_draws": int(len(a)), "p_perm": p}


def within(v, lo, hi) -> bool | None:
    if None in (v, lo, hi):
        return None
    return bool(lo <= v <= hi)


# --------------------------------------------------------------------------- #
# per-model computation
# --------------------------------------------------------------------------- #
def compute_model(slug: str, cfg: dict, args, metrics_mod, log) -> dict:
    root, _, _ = G.model_dirs(cfg["output"]["root"], slug)
    man = read_manifest(root)
    n_rows = int(man["model"]["n_layer_rows"])
    n_blocks = int(man["model"]["n_blocks"])
    hidden = int(man["model"]["hidden_size"])
    min_n = int(getattr(metrics_mod, "MIN_N", cfg["study"].get("min_n", FALLBACK_MIN_N)))
    draws = int(args.balance_draws)
    null_draws = int(args.null_balance_draws)

    log(f"\n=== {slug}  blocks={n_blocks} rows={n_rows} H={hidden} ===")
    log(f"  layer rows analysed: 1..{n_blocks - 1} inclusive")
    log(f"  row 0 SKIPPED: inputs_embeds, identical across prompts at '{POINT}' -- "
        f"mu_R - mu_N is the zero vector there and the cosine would be NaN")
    if n_rows - 1 >= n_blocks:
        log(f"  row {n_blocks} SKIPPED: post-final-norm, not a residual site and not "
            f"steerable (rows {n_blocks}..{n_rows - 1} are excluded)")

    splits = load_study_splits(cfg)
    dir_ids = [int(p) for p in splits["splits"]["direction"]]
    log(f"  study split sha256={splits['sha256'][:16]}  direction n={len(dir_ids)}")

    tbl = resolve_label_table(cfg, slug, args.labels, log)
    tbl = fill_lengths_from_answers(root, tbl, log)

    ids = [p for p in dir_ids if p in tbl.index]
    if len(ids) < len(dir_ids):
        log(f"  WARNING {len(dir_ids) - len(ids)} Direction-split prompts have no label")
    if args.limit:
        ids = ids[: args.limit]
        log(f"  --limit {args.limit}: SMOKE RUN, numbers are not publishable")

    log("  loading Direction-split activations")
    X, kept = load_stack(root, ids, n_rows, log)
    meta = tbl.loc[kept]
    lab = meta["label"].to_numpy().astype(str)
    counts = {c: int((lab == c).sum()) for c in LABELS3}
    n_sub = min(counts["REFUSAL"], counts["DISCLAIMER"])
    log(f"  direction-split labels {counts}  balanced NORMAL subsample n={n_sub} "
        f"(K={draws} draws)")
    if counts["REFUSAL"] < min_n or counts["DISCLAIMER"] < min_n:
        log(f"  WARNING class support below MIN_N={min_n}: every S for this model is "
            f"marked insufficient_evidence")

    # ---- the length axis, NORMAL answers only -------------------------------- #
    short_mask = cap_mask = None
    len_note = ""
    ntok = pd.to_numeric(meta["n_generated_tokens"], errors="coerce").to_numpy(dtype=float)
    cap_flag = meta["hit_max_new_tokens"]
    is_normal = lab == "NORMAL"
    if np.isnan(ntok).all() or cap_flag.isna().all():
        len_note = "no answer-length columns available"
    else:
        cap = np.asarray([bool(v) if v is not None and v == v else False
                          for v in cap_flag.to_numpy()])
        cap_mask = is_normal & cap
        with np.errstate(invalid="ignore"):
            thresh = np.nanpercentile(ntok[is_normal], 30) if is_normal.any() else np.nan
        short_mask = is_normal & (ntok <= thresh) & ~cap_mask   # disjoint by construction
        n_short, n_cap = int(short_mask.sum()), int(cap_mask.sum())
        log(f"  d_len groups (NORMAL only): shortest 30% n={n_short} "
            f"(<= {thresh:.0f} tokens) vs generation-cap n={n_cap}")
        if n_cap < min_n or n_short < min_n:
            len_note = (f"insufficient_evidence: short n={n_short}, cap n={n_cap}, "
                        f"MIN_N={min_n}")
            log(f"  WARNING {len_note} -- the length cosines are reported as null, not "
                f"guessed. Length is still a first-order axis here; the claim simply "
                f"cannot be supported on this model's data.")
            short_mask = cap_mask = None

    # ---- C*-test-clean subset ------------------------------------------------ #
    cstar_split = meta["cstar_split"]
    clean = (cstar_split.astype(str).to_numpy() == "test") if cstar_split.notna().any() \
        else np.zeros(len(kept), dtype=bool)
    clean_counts = {c: int(((lab == c) & clean).sum()) for c in LABELS3}
    clean_ok = (clean_counts["REFUSAL"] >= min_n and clean_counts["DISCLAIMER"] >= min_n)
    log(f"  C*-test-clean subset: n={int(clean.sum())} {clean_counts} "
        f"-> {'reported' if clean_ok else 'insufficient_evidence'} "
        f"(1,223 of 1,750 prompts are in C*'s own training split)")

    # ---- step-2 directions --------------------------------------------------- #
    stored = load_step2_directions(
        G.repo_path(cfg["output"]["study_root"]) / "directions" / f"{slug}.npz", hidden, log)
    need = set(range(1, n_blocks))
    if stored is not None:
        have = {l for _, l in stored} & need
        if have != need:
            log(f"  step-2 directions cover {len(have)}/{len(need)} analysed layers "
                f"(missing {sorted(need - have)[:8]}...) -- recomputing ALL layers "
                f"rather than mixing two estimates in one curve")
            stored = None
    source = "step2_npz" if stored is not None else "recomputed_from_activations"
    log(f"  directions source: {source}")

    selected = load_selected_layers(slug, log) or []

    # ---- per-layer loop ------------------------------------------------------ #
    rows, drift = [], []
    t0 = time.perf_counter()
    for l in range(n_rows):
        steerable = 1 <= l <= n_blocks - 1
        if not steerable:
            rows.append({"model": slug, "layer_row": l, "analysed": False,
                         "reason": ("row 0: inputs_embeds, identical across prompts"
                                    if l == 0 else "post-final-norm, not a residual site")})
            continue
        Xl = X[:, l, :]
        rng = np.random.default_rng(seed_for(args.seed, slug, l, "naive"))
        dR_fit, dD_fit, normR, normD = dim_pair(Xl, lab, draws, rng)

        dR, dD = dR_fit, dD_fit
        agree_R = agree_D = None
        if stored is not None:
            dR, dD = S.unit(stored[("dR", l)]), S.unit(stored[("dD", l)])
            agree_R, agree_D = cos(dR, dR_fit), cos(dD, dD_fit)
            worst = min([a for a in (agree_R, agree_D) if a is not None], default=1.0)
            if worst < args.agree_min:
                drift.append((l, agree_R, agree_D))

        # Full Direction split throughout -- no cross-fitting (user decision 2026-09-22).
        S_l = cos(dR, dD)
        null = null_band_layer(
            Xl, lab, args.n_null, null_draws,
            np.random.default_rng(seed_for(args.seed, slug, l, "null")), observed=S_l)

        cos_R_len = cos_D_len = None
        if short_mask is not None:
            d_len, _ = length_direction(
                Xl, short_mask, cap_mask, draws,
                np.random.default_rng(seed_for(args.seed, slug, l, "dlen")))
            cos_R_len, cos_D_len = cos(dR, d_len), cos(dD, d_len)

        S_clean = None
        if clean.any():
            rngc = np.random.default_rng(seed_for(args.seed, slug, l, "clean"))
            cR, cD, _, _ = dim_pair(Xl[clean], lab[clean], draws, rngc)
            S_clean = cos(cR, cD)

        geo_cos = None
        if metrics_mod is not None:
            y = np.array([LABELS3.index(v) for v in lab])
            g = metrics_mod.label_geometry(Xl.astype(np.float64), y)
            pair = g["pairs"].get("DISCLAIMER|REFUSAL", {})
            geo_cos = pair.get("cosine")

        rec = {
            "model": slug, "layer_row": l, "analysed": True,
            "relative_depth": round(l / n_blocks, 4),
            "selected": bool(l in selected),
            "n_direction": len(kept),
            "n_NORMAL": counts["NORMAL"], "n_DISCLAIMER": counts["DISCLAIMER"],
            "n_REFUSAL": counts["REFUSAL"], "n_balanced_subsample": int(n_sub),
            "directions_source": source,
            "agree_dR_vs_recomputed": agree_R, "agree_dD_vs_recomputed": agree_D,
            "S": S_l,
            "null_mean": null["mean"], "null_lo": null["lo"], "null_hi": null["hi"],
            "null_n_draws": null["n_draws"], "p_perm": null["p_perm"],
            "within_null": within(S_l, null["lo"], null["hi"]),
            "cos_R_len": cos_R_len, "cos_D_len": cos_D_len,
            "norm_R": normR, "norm_D": normD,
            "norm_ratio_D_over_R": (None if not normR else (normD / normR if normD else None)),
            "sigma_dR": S.sigma_along(Xl, dR) if dR is not None else None,
            "sigma_dD": S.sigma_along(Xl, dD) if dD is not None else None,
            "cstar_cos_centroid_DR": geo_cos,
            "S_cstar_test_clean": S_clean,
            "cstar_test_clean_evidence": "ok" if clean_ok else "insufficient_evidence",
            "evidence": ("ok" if (counts["REFUSAL"] >= min_n and counts["DISCLAIMER"] >= min_n)
                         else "insufficient_evidence"),
        }
        rows.append(rec)
        if l % 4 == 0 or l == n_blocks - 1:
            p_txt = "--" if null["p_perm"] is None else f"{null['p_perm']:.4f}"
            log(f"    L{l:03d}  S={fmt(S_l)}"
                f"  null=[{fmt(null['lo'])},{fmt(null['hi'])}]  p_perm={p_txt}"
                f"  {'WITHIN NULL' if rec['within_null'] else ''}"
                f"  ({time.perf_counter() - t0:.0f}s)")

    if drift:
        msg = ("step-2 directions disagree with the recomputed balanced difference-of-means "
               f"below --agree-min={args.agree_min} at layers "
               + ", ".join(f"L{l}(dR={fmt(a)},dD={fmt(b)})" for l, a, b in drift[:10]))
        if args.allow_direction_drift:
            log(f"  WARNING {msg} -- continuing because --allow-direction-drift was given")
        else:
            raise RuntimeError(
                msg + ". Steps 2 and 3 are then not describing the same vectors. Rerun "
                      "step 2, or pass --allow-direction-drift if the difference is "
                      "understood and intended.")

    ok = [r for r in rows if r.get("analysed")]
    marked = [r["layer_row"] for r in ok if r.get("within_null")]
    log(f"\n  layers whose S falls INSIDE the null band "
        f"(no evidence of a shared axis): {marked if marked else 'none'}")
    sel_marked = sorted(set(marked) & set(selected))
    if sel_marked:
        log(f"  WARNING {len(sel_marked)} of the {len(selected)} SELECTED steering "
            f"layers are in that list: {sel_marked}")

    return {
        "model": slug,
        "hf_id": man["model"].get("hf_id"),
        "n_blocks": n_blocks, "n_layer_rows": n_rows, "hidden_size": hidden,
        "capture_point": POINT,
        "analysed_rows": [1, n_blocks - 1],
        "excluded_rows": {"0": "inputs_embeds, identical across prompts -> 0/0 cosine",
                          str(n_blocks): "post-final-norm, not a residual site"},
        "n_direction_prompts": len(kept),
        "direction_label_counts": counts,
        "balanced_subsample_n": int(n_sub),
        "balance_draws": draws,
        "min_n": min_n,
        "selected_layers": selected,
        "directions_source": source,
        "length_axis": {"available": short_mask is not None, "note": len_note,
                        "n_short": int(short_mask.sum()) if short_mask is not None else 0,
                        "n_cap": int(cap_mask.sum()) if cap_mask is not None else 0},
        "cstar_test_clean": {"n": int(clean.sum()), "counts": clean_counts,
                             "evidence": "ok" if clean_ok else "insufficient_evidence"},
        "layers_within_null": marked,
        "selected_layers_within_null": sel_marked,
        "layers": rows,
    }


def fmt(v) -> str:
    return "  --  " if v is None else f"{v:+.3f}"


# --------------------------------------------------------------------------- #
# hashing / artifacts
# --------------------------------------------------------------------------- #
def canonical_payload(models: list[dict]) -> str:
    """The bytes the sha256 is taken over: every analysed layer's numbers, rounded.

    Rounded to 6 dp so that a float repr difference between numpy builds cannot make a
    reproduced run look like drift, while any real change to an estimate still does.
    """
    out = []
    for m in models:
        for r in m["layers"]:
            if not r.get("analysed"):
                continue
            out.append({k: (round(v, 6) if isinstance(v, float) else v)
                        for k, v in sorted(r.items())})
    return json.dumps(out, sort_keys=True, ensure_ascii=False)


def sha256_of(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def write_artifacts(report: dict, out_dir: Path, log) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "geometry.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    flat = pd.DataFrame([r for m in report["models"] for r in m["layers"]
                         if r.get("analysed")])
    flat.to_parquet(out_dir / "geometry.parquet", index=False)
    log(f"\nsha256 = {report['sha256']}")
    log(f"wrote {(out_dir / 'geometry.json').as_posix()}")
    log(f"wrote {(out_dir / 'geometry.parquet').as_posix()}  rows={len(flat)}")


def verify(out_dir: Path, log) -> int:
    """Recompute the committed hash and cross-check the parquet against the json."""
    jp, pp = out_dir / "geometry.json", out_dir / "geometry.parquet"
    if not jp.exists():
        log(f"MISSING {jp.as_posix()} -- nothing to verify")
        return 1
    rep = json.loads(jp.read_text(encoding="utf-8"))
    digest = sha256_of(canonical_payload(rep["models"]))
    ok = digest == rep.get("sha256")
    log(f"committed {str(rep.get('sha256'))[:16]}  recomputed {digest[:16]}  "
        f"{'MATCH' if ok else 'MISMATCH'}")
    if pp.exists():
        n_json = sum(1 for m in rep["models"] for r in m["layers"] if r.get("analysed"))
        n_pq = len(pd.read_parquet(pp))
        same = n_json == n_pq
        log(f"parquet rows {n_pq} vs json analysed rows {n_json}  "
            f"{'MATCH' if same else 'MISMATCH'}")
        ok = ok and same
    else:
        log(f"MISSING {pp.as_posix()}")
        ok = False
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
def _series(layers: list[dict], key: str):
    xs = [r["layer_row"] for r in layers if r.get("analysed")]
    ys = [r.get(key) for r in layers if r.get("analysed")]
    return np.array(xs, dtype=float), np.array([np.nan if v is None else v for v in ys],
                                               dtype=float)


def _broken_line(ax, x, y, mask_break, **kw):
    """Markers plus straight segments, with the line BROKEN wherever mask_break is set.

    Deliberately not a smoothed or interpolating curve: the early layers are where the
    estimate is least stable, and a spline through them would invent a rising trend that
    the null band says is not there. Breaking the line at within-null layers makes
    the gap in the evidence visible instead of paving over it.
    """
    yy = y.copy()
    yy[mask_break] = np.nan
    ax.plot(x, yy, **kw)
    kw2 = {k: v for k, v in kw.items() if k in ("color", "zorder")}
    ax.plot(x[~np.isnan(y)], y[~np.isnan(y)], linestyle="none", marker="o", ms=3.2,
            **kw2)
    if mask_break.any():
        ax.plot(x[mask_break & ~np.isnan(y)], y[mask_break & ~np.isnan(y)],
                linestyle="none", marker="x", ms=5.5, mew=1.2, **kw2)


def plot_model(m: dict, out_dir: Path, log) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = m["layers"]
    x, s = _series(L, "S")
    _, nlo = _series(L, "null_lo")
    _, nhi = _series(L, "null_hi")
    _, rlen = _series(L, "cos_R_len")
    _, dlen = _series(L, "cos_D_len")
    _, ratio = _series(L, "norm_ratio_D_over_R")
    ov = np.array([bool(r.get("within_null")) for r in L if r.get("analysed")])
    sel = m.get("selected_layers") or []

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 9.5), sharex=True)
    fig.suptitle(f"Step 3 -- layer-wise direction geometry: {m['model']}\n"
                 f"capture point '{m['capture_point']}', rows 1..{m['n_blocks'] - 1}, "
                 f"Direction split n={m['n_direction_prompts']} "
                 f"{m['direction_label_counts']}", fontsize=10)

    def mark(ax):
        for l in sel:
            ax.axvline(l, color="0.82", lw=0.9, zorder=0)
        ax.grid(True, alpha=0.25, lw=0.5)

    ax = axes[0]
    mark(ax)
    ax.fill_between(x, nlo, nhi, color="0.6", alpha=0.35, lw=0,
                    label="null band (permuted labels, same estimator, 95%)")
    _broken_line(ax, x, s, ov, color="tab:blue", lw=1.4,
                 label="S_i (full Direction split)")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_ylabel("S_i = cos(d_D, d_R)")
    ax.legend(fontsize=7.5, loc="best")
    ax.set_title("x marks a layer whose S falls inside the null band; the line is "
                 "broken there. Grey verticals: the 10 selected steering layers.",
                 fontsize=8)

    ax = axes[1]
    mark(ax)
    if np.isfinite(rlen).any() or np.isfinite(dlen).any():
        _broken_line(ax, x, rlen, np.zeros_like(ov, dtype=bool), color="tab:green",
                     lw=1.2, label="cos(d_R, d_len)")
        _broken_line(ax, x, dlen, np.zeros_like(ov, dtype=bool), color="tab:purple",
                     lw=1.2, label="cos(d_D, d_len)")
        ax.legend(fontsize=7.5, loc="best")
    else:
        ax.text(0.5, 0.5, f"length axis unavailable: {m['length_axis']['note']}",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_ylabel("cos with the length axis")

    ax = axes[2]
    mark(ax)
    _broken_line(ax, x, ratio, np.zeros_like(ov, dtype=bool), color="tab:brown", lw=1.2)
    ax.axhline(1.0, color="k", lw=0.6, linestyle=":")
    ax.set_ylabel("||mu_D-mu_N|| / ||mu_R-mu_N||")
    ax.set_xlabel("layer row (row l = output of decoder block l-1)")

    fig.tight_layout(rect=(0, 0, 1, 0.965))
    for ext in ("png", "pdf"):
        p = out_dir / f"step3_{m['model']}.{ext}"
        fig.savefig(p, dpi=170 if ext == "png" else None)
        log(f"wrote {p.as_posix()}")
    plt.close(fig)


def plot_combined(models: list[dict], out_dir: Path, log) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    fig.suptitle("Step 3 -- direction geometry across the roster "
                 "(x = relative depth l / n_blocks, so models of different depth line up)",
                 fontsize=10)
    panels = [("S", "S_i = cos(d_D, d_R), full Direction split"),
              ("cos_R_len", "cos(d_R, d_len)"),
              ("cos_D_len", "cos(d_D, d_len)"),
              ("norm_ratio_D_over_R", "||mu_D-mu_N|| / ||mu_R-mu_N||")]
    for ax, (key, title) in zip(axes.ravel(), panels):
        for m in models:
            L = m["layers"]
            depth = np.array([r["relative_depth"] for r in L if r.get("analysed")])
            _, y = _series(L, key)
            ov = np.array([bool(r.get("within_null")) for r in L if r.get("analysed")])
            yy = y.copy()
            if key == "S":
                yy[ov] = np.nan            # never connect a line across untested layers
            ax.plot(depth, yy, lw=1.3, label=m["model"])
            ax.plot(depth[~np.isnan(y)], y[~np.isnan(y)], linestyle="none", marker="o",
                    ms=2.6, color=ax.lines[-1].get_color())
            for l in (m.get("selected_layers") or []):
                ax.axvline(l / m["n_blocks"], color="0.9", lw=0.7, zorder=0)
        ax.axhline(0.0, color="k", lw=0.6)
        if key == "norm_ratio_D_over_R":
            ax.axhline(1.0, color="k", lw=0.6, linestyle=":")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("relative depth")
        ax.grid(True, alpha=0.25, lw=0.5)
        ax.legend(fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "pdf"):
        p = out_dir / f"step3_combined.{ext}"
        fig.savefig(p, dpi=170 if ext == "png" else None)
        log(f"wrote {p.as_posix()}")
    plt.close(fig)


def write_summary(report: dict, out_dir: Path, log) -> None:
    """summary.md -- what the curve means, in words, from the numbers actually computed."""
    lines = ["# Step 3 -- layer-wise refusal / disclaimer direction geometry", "",
             f"seed {report['seed']} | sha256 `{report['sha256'][:16]}...` | "
             f"balance draws K={report['args']['balance_draws']}, "
             f"null draws={report['args']['n_null']} | full Direction split, "
             f"no cross-fitting",
             "",
             "`S_i = cos(d_D, d_R)` at layer row `i`, both directions built by the same "
             "balanced difference-of-means used for steering in step 2. Row `l` is the "
             "output of decoder block `l-1`; rows 0 and `n_blocks` are excluded (row 0 is "
             "identical across prompts at this capture point, so the direction is 0/0; row "
             "`n_blocks` is post-final-norm and is not a residual site).", ""]

    for m in report["models"]:
        ok = [r for r in m["layers"] if r.get("analysed") and r.get("S") is not None]
        lines += [f"## {m['model']}", "",
                  f"- Direction split: n={m['n_direction_prompts']}, labels "
                  f"{m['direction_label_counts']}, balanced NORMAL subsample "
                  f"n={m['balanced_subsample_n']} (per-model class counts differ between "
                  f"models by design -- one shared prompt partition, each model's own C* "
                  f"label).",
                  f"- Directions: {m['directions_source']}.",
                  f"- Evidence: {ok[0]['evidence'] if ok else 'n/a'} "
                  f"(MIN_N={m['min_n']}).",
                  f"- C*-test-clean subset: n={m['cstar_test_clean']['n']} "
                  f"{m['cstar_test_clean']['counts']} -> "
                  f"{m['cstar_test_clean']['evidence']}."]
        if not ok:
            lines += ["", "No layer produced an estimate.", ""]
            continue
        peak = max(ok, key=lambda r: r["S"])
        sel = [r for r in ok if r["selected"]]
        selmean = float(np.mean([r["S"] for r in sel])) if sel else None
        dlen = [r["cos_D_len"] for r in sel if r.get("cos_D_len") is not None]
        rlen = [r["cos_R_len"] for r in sel if r.get("cos_R_len") is not None]
        ratio = [r["norm_ratio_D_over_R"] for r in sel if r.get("norm_ratio_D_over_R")]
        p_peak = peak.get("p_perm")
        lines += [
            f"- Peak S_i = {peak['S']:+.3f} at row {peak['layer_row']} "
            f"(relative depth {peak['relative_depth']:.2f}); null band there "
            f"[{fmt(peak['null_lo'])}, {fmt(peak['null_hi'])}], permutation p = "
            + ("n/a." if p_peak is None else f"{p_peak:.4f}."),
            f"- Mean S_i over the {len(sel)} selected steering layers: "
            + (f"{selmean:+.3f}." if selmean is not None else "no selected layers on file."),
            f"- Layers whose S falls inside the null band: "
            f"{m['layers_within_null'] if m['layers_within_null'] else 'none'}"
            + (f"; of the selected steering layers, "
               f"{m['selected_layers_within_null']} do."
               if m['selected_layers_within_null'] else "."),
        ]
        if dlen and rlen:
            lines.append(f"- Length axis at the selected layers: mean cos(d_D, d_len) = "
                         f"{np.mean(dlen):+.3f}, mean cos(d_R, d_len) = "
                         f"{np.mean(rlen):+.3f}.")
        else:
            lines.append(f"- Length axis: {m['length_axis']['note'] or 'not computed'}.")
        if ratio:
            lines.append(f"- ||mu_D-mu_N|| / ||mu_R-mu_N|| at the selected layers: mean "
                         f"{np.mean(ratio):.3f}.")
        lines.append("")

    lines += [
        "## What this means for the central hypothesis", "",
        "The study claims that disclaiming and refusing are separately steerable "
        "behaviours. `S_i` is the number a reviewer will press on, because it measures "
        "how much of `d_D` is simply `d_R`. Read it this way:", "",
        "- **A high `S_i` at the steered layers does not by itself sink the claim, but it "
        "does move the burden of proof onto step 2.** If `d_D` and `d_R` are substantially "
        "aligned where the steering happens, then any pp change produced by pushing along "
        "`d_D` could have been produced by pushing along `d_R`. What separates the two is "
        "then not the geometry but the *behavioural* result: steering along `d_D` must "
        "move DISCLAIMER without moving REFUSAL by the same amount, and the alpha/sigma_l "
        "dose-response curves for the two directions must differ in shape, not only in "
        "scale. Quote this file next to that figure, not instead of it.",
        "- **`S_i` is computed on the full Direction split (no cross-fitting).** `d_R` "
        "and `d_D` share `mu_N`, which nudges `S_i` up slightly; the null band is built "
        "with the same shared-`mu_N` estimator, so it carries the same inflation. Quote "
        "`S_i` together with its null band, never alone.",
        "- **Do not read the early layers.** Where `S_i` falls inside the null band the "
        "data has not shown any shared axis, and it has equally not shown the absence of "
        "one. Those layers are marked with an x and the curve is broken through them on "
        "purpose.",
        "- **Check the length axis before claiming a `d_D` effect is about disclaiming.** "
        "The capture point is the last prompt token, where the model has already settled "
        "roughly how long its answer will be, and a disclaimer is a long answer. A "
        "`cos(d_D, d_len)` that is large at the steered layers means part of any measured "
        "effect is a length effect wearing a disclaimer's clothes.",
        "- **`cstar_cos_centroid_DR` is not `S_i`.** The classifier-side "
        "`metrics.label_geometry` reports the cosine between the raw class *centroids*, "
        "which on residual-stream activations is dominated by the shared mean vector and "
        "sits near 1.0 at every layer. It is in the table for continuity with the "
        "classifier report. `S_i` is the cosine between centroid *differences from "
        "NORMAL*. Conflating them would turn a real finding into a trivial one.",
        "- **Labels are C*'s, and C* was trained on 1,223 of these 1,750 prompts.** The "
        "`*_cstar_test_clean` columns repeat `S` on the held-out subset; where they are "
        "marked `insufficient_evidence` the subset is simply too small for the min-n rule "
        f"(MIN_N={report['models'][0]['min_n'] if report['models'] else 30}) and no number "
        "should be quoted from them.", ""]
    p = out_dir / "summary.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    log(f"wrote {p.as_posix()}")


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #
def resolve_models(cfg: dict, args, log) -> list[str]:
    if args.models:
        slugs = [s.strip() for s in args.models.split(",") if s.strip()]
    else:
        slugs = [sp.slug for sp in G.parse_models(cfg)]
    root = G.repo_path(cfg["output"]["root"])
    present = [s for s in slugs if (root / s / "manifest.json").exists()]
    for s in slugs:
        if s not in present:
            log(f"  SKIP {s}: no {(root / s / 'manifest.json').as_posix()} -- step 1 has "
                f"not run for it. Nothing is estimated for a model with no activations.")
    if not present:
        raise SystemExit(
            f"none of {slugs} have activations under {root.as_posix()}. Run step 1 first, "
            f"or pass --models with a slug that exists.")
    return present


def load_study_splits(cfg: dict) -> dict:
    """study_splits.json, with the provenance guard.

    The superseded majority-label partition (sha 7fffb02a, roster gemma-4-e2b /
    qwen3-5-4b / qwen3-5-9b) is still on disk and parses cleanly, so structure checks are
    not enough -- S.assert_study_splits_current checks the stratification rule and the
    roster. Geometry read off the wrong Direction split is wrong in a way no plot shows.
    """
    path = STUDY_DIR / "study_splits.json"
    if not path.exists():
        raise SystemExit(
            f"!! no {path.as_posix()}. Run step 1's split stage first; step 3 never "
            f"invents a partition of its own.")
    doc = json.loads(path.read_text(encoding="utf-8"))
    try:
        S.assert_study_splits_current(
            doc, [s.slug for s in G.parse_models(cfg)], path=path.as_posix(),
            partition=str((cfg.get("study") or {}).get("partition", "shared")),
            fractions=S.split_fractions(cfg))
    except (S.StaleArtifactError, ValueError) as e:
        raise SystemExit(f"!! {e}")
    return doc


def dry_run(cfg: dict, args, log) -> None:
    slugs = resolve_models(cfg, args, log)
    splits = load_study_splits(cfg)
    n_dir = len(splits["splits"]["direction"])
    log(f"\nDRY RUN -- nothing is written")
    log(f"  models            {slugs}")
    log(f"  direction prompts {n_dir}" + (f" (capped to {args.limit})" if args.limit else ""))
    total_layers = 0
    for s in slugs:
        man = read_manifest(G.repo_path(cfg["output"]["root"]) / s)
        nb = int(man["model"]["n_blocks"])
        total_layers += nb - 1
        log(f"  {s}: rows 1..{nb - 1} analysed ({nb - 1} layers), "
            f"rows 0 and {nb} excluded, H={man['model']['hidden_size']}")
    fits = total_layers * 2 * (args.balance_draws + args.n_null * args.null_balance_draws)
    log(f"  balanced difference-of-means fits: ~{fits:,}")
    log(f"  GPU hours: 0.0 -- CPU only, no model is loaded and nothing is generated")
    log(f"  expected wall time: a few minutes per model")
    log(f"  outputs would be written under {OUT_DIR.as_posix()}")


def stage_compute(cfg: dict, args, log) -> dict:
    slugs = resolve_models(cfg, args, log)
    metrics_mod = load_cstar_metrics(
        G.repo_path(cfg["study"].get("classifier_repo", "../refusal_classifier")), log)
    models = [compute_model(s, cfg, args, metrics_mod, log) for s in slugs]
    splits = load_study_splits(cfg)
    report = {
        "stage": "step3_geometry",
        "seed": args.seed,
        "config": str(Path(args.config).as_posix()),
        "study_split_sha256": splits["sha256"],
        "capture_point": POINT,
        "labels3": LABELS3,
        "definitions": {
            "S_i": "cos(unit(mu_D - mu_N), unit(mu_R - mu_N)) at layer row i, both "
                   "directions on the FULL Direction split (no cross-fitting)",
            "null": "S_i with class labels permuted within the Direction split (same "
                    "shared-mu_N estimator); p_perm = (1 + #{null >= S_i}) / (1 + n)",
            "d_len": "unit(mu_cap - mu_short) over NORMAL answers only: hit the "
                     "generation cap minus the shortest 30 percent",
            "cstar_cos_centroid_DR": "refusal_classifier metrics.label_geometry's "
                                     "cos(centroid_D, centroid_R) -- raw centroids, NOT "
                                     "centroid differences; a different statistic from S_i",
        },
        "inputs": {
            "study_splits": (STUDY_DIR / "study_splits.json").as_posix(),
            "selected_layers": (STUDY_DIR / "selected_layers.json").as_posix(),
            "labels": args.labels or "auto-resolved per model (see the log)",
            "activations_root": G.repo_path(cfg["output"]["root"]).as_posix(),
        },
        "args": {"balance_draws": args.balance_draws,
                 "null_balance_draws": args.null_balance_draws,
                 "n_null": args.n_null,
                 "limit": args.limit, "agree_min": args.agree_min},
        "smoke": bool(args.limit),
        "models": models,
    }
    report["sha256"] = sha256_of(canonical_payload(models))
    write_artifacts(report, OUT_DIR, log)
    return report


def stage_plot(args, log) -> None:
    p = OUT_DIR / "geometry.json"
    if not p.exists():
        raise SystemExit(f"{p.as_posix()} not found -- run --stage compute first. "
                         f"This script never plots numbers it has not computed.")
    report = json.loads(p.read_text(encoding="utf-8"))
    log(f"plotting from {p.as_posix()}  sha256={report['sha256'][:16]}")
    for m in report["models"]:
        plot_model(m, OUT_DIR, log)
    if len(report["models"]) > 1:
        plot_combined(report["models"], OUT_DIR, log)
    else:
        log("only one model on file -- skipping the combined figure")
    write_summary(report, OUT_DIR, log)


def main() -> None:
    # STUDY_DIR/OUT_DIR are module globals so the whole file can use them, but they must
    # follow the CONFIG (and the smoke suffix), not a hardcoded data/study. Declared up
    # front: --out-dir's default reads OUT_DIR below, and Python forbids a `global` that
    # follows a use of the name in the same function.
    global STUDY_DIR, OUT_DIR

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--stage", choices=["compute", "plot", "all"], default="all")
    ap.add_argument("--models", default=None,
                    help="comma-separated slugs; default is the config roster")
    ap.add_argument("--labels", default=None,
                    help="parquet of prompt_id -> C* label; default auto-resolves")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--seed", type=int, default=None, help="default: study.seed")
    ap.add_argument("--balance-draws", type=int, default=None,
                    help="K for the balanced difference-of-means; default: "
                         "directions.balance_draws")
    ap.add_argument("--null-balance-draws", type=int, default=10,
                    help="K inside each null draw. The null band's width is dominated by "
                         "the label permutation, not by subsample averaging, so 10 keeps "
                         "200 draws x ~32 layers inside a minute")
    ap.add_argument("--n-null", type=int, default=200,
                    help="label permutations per layer (the spec's floor is 200)")
    ap.add_argument("--agree-min", type=float, default=0.99,
                    help="minimum cos(step-2 direction, recomputed direction) before this "
                         "script refuses to continue")
    ap.add_argument("--allow-direction-drift", action="store_true",
                    help="downgrade the --agree-min failure to a warning")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the Direction-split prompts (smoke runs only)")
    ap.add_argument("--smoke", action="store_true",
                    help="read the _smoke tree written by steps 1-2 (config.smoke block)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="recompute the committed sha256 and exit non-zero on drift")
    args = ap.parse_args()

    # Resolve the smoke redirect BEFORE the mkdir, or running --smoke still creates the
    # production report directory as a side effect. `default_out` reads the module-level
    # value, which is what --out-dir defaulted to.
    default_out = str(OUT_DIR)
    OUT_DIR = Path(args.out_dir)
    if args.smoke and args.out_dir == default_out:
        OUT_DIR = ROOT / "results" / "reports" / "step3_smoke"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.verify:
        sys.exit(verify(OUT_DIR, Tee(OUT_DIR / "verify.log")))

    cfg = G.load_config(args.config)
    # steering.positions / output_format / save_activations / static_batch_membership and
    # directions.names describe LOCKED behaviour that is hardcoded in src/steer.py. Flipping
    # one in the yaml would otherwise change nothing, silently -- so assert instead.
    S.assert_locked_steering_config(cfg)
    if args.smoke:
        sm = cfg.get("smoke") or {}
        cfg["models"] = sm.get("models", cfg["models"])
        S.apply_smoke_paths(cfg)
        if args.limit is None:
            args.limit = int(sm.get("n_prompts", 8))
    STUDY_DIR = G.repo_path(cfg["output"]["study_root"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.seed is None:
        args.seed = int(cfg["study"]["seed"])
    if args.balance_draws is None:
        args.balance_draws = int(cfg["directions"]["balance_draws"])
    if args.n_null < 200:
        raise SystemExit("--n-null below 200 makes the null band too coarse to compare "
                         "an interval against; the spec's floor is 200")

    stage = "dry-run" if args.dry_run else args.stage
    log = Tee(OUT_DIR / f"{stage}.log")
    log(f"\n{'=' * 78}\nstep3_geometry  stage={stage}  seed={args.seed}  "
        f"K={args.balance_draws}  null={args.n_null}\n{'=' * 78}")

    if args.dry_run:
        dry_run(cfg, args, log)
        return
    if args.stage in ("compute", "all"):
        stage_compute(cfg, args, log)
    if args.stage in ("plot", "all"):
        stage_plot(args, log)


if __name__ == "__main__":
    main()
