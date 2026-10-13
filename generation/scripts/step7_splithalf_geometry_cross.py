"""Step 7 -- split-half geometry: median cross cos(d_D, d_R) with a 95% split-half interval.

Step 3's single full-split S = cos(d_D, d_R) is inflated by the mu_N that d_D and d_R
share. This step replaces it with the split-half CROSS estimate, which removes that:

  1. per model, per layer in the union of the top-10 layers by AUROC(REFUSAL vs rest)
     and the top-10 by AUROC(DISCLAIMER vs rest), both read from step 2's probe table
     (selected_layers.json -> per_layer_table; same tie-break as step 2: higher AUROC
     first, then the shallower row)
  2. the 40% Direction split is cut 50/50 into halves A and B, stratified by the model's
     own C* label so each half holds half the REFUSAL and half the DISCLAIMER prompts
  3. in each half: d_D(A), d_R(A), d_D(B), d_R(B) with step 2's balanced
     difference-of-means (NORMAL subsampled to min(n_R, n_D) of that half, K draws,
     mean of unit vectors, renormalised) -- the function is imported from step 2
  4.   cross = 1/2 [cos(d_D(A), d_R(B)) + cos(d_R(A), d_D(B))]
     cross pairs a direction from A with one from B, so the two never share a mu_N
     estimate; that removes step 3's shared-mu_N inflation. No reliability correction.
  5. repeated over --n-splits (200) seeds
  6. per layer: median(cross) = the dot, 2.5th / 97.5th percentiles = the 95% interval.

--layer-set selected keeps the old behaviour (step 2's top-10 by mean AUROC); --layer-set
all runs every steerable row; --layers overrides all of them.

CPU only. Reads the `prompt_last` activations step 1 wrote and step 2's
selected_layers.json; writes nothing outside results/reports/step7_cross[_smoke]/.

OUTPUTS (results/reports/step7_cross/<slug>/)
  splits.parquet    one row per (seed, layer): cross + half sizes
  summary.csv/.parquet  one row per layer: median_cross, p2_5, p97_5
  step7_<slug>.png/.pdf   dot-whisker plot
  step7.json        provenance: split sha, label counts, layers (and which top-10 list
                    each came from), K, n_splits, seed
and step7_combined.png/.pdf over every model with a summary (--stage plot).

Usage:
  python -X utf8 scripts/step7_splithalf_geometry.py                    # all models
  python -X utf8 scripts/step7_splithalf_geometry.py --models olmo3-7b
  python -X utf8 scripts/step7_splithalf_geometry.py --stage plot
  python -X utf8 scripts/step7_splithalf_geometry.py --smoke --models smoke-olmo2-1b
"""
from __future__ import annotations

import argparse
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

DEFAULT_CONFIG = "configs/study.yaml"
LABELS3 = ("NORMAL", "DISCLAIMER", "REFUSAL")


def import_step2():
    """Step 2's loaders and estimator, imported rather than copied, so the halves use
    exactly the direction estimator that produced the steering vectors."""
    path = ROOT / "scripts" / "step2_directions_steering.py"
    spec = importlib.util.spec_from_file_location("step2_directions_steering", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


S2 = import_step2()


def sub_seed(*parts: int) -> int:
    return int(np.random.SeedSequence([int(p) for p in parts]).generate_state(1)[0])


def stratified_halves(lab: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Boolean mask, True = half A. Each class is permuted and cut in two; an odd class
    gives its extra prompt to A or B at random so neither half is systematically larger."""
    in_a = np.zeros(len(lab), dtype=bool)
    for c in LABELS3:
        idx = np.flatnonzero(lab == c)
        if len(idx) == 0:
            continue
        idx = rng.permutation(idx)
        k = len(idx) // 2 + (int(rng.integers(2)) if len(idx) % 2 else 0)
        in_a[idx[:k]] = True
    return in_a


def half_directions(Xl: np.ndarray, lab: np.ndarray, draws: int, seed: int):
    """(d_D, d_R) on one half, step 2's estimator, NORMAL balanced to min(n_R, n_D)."""
    XN, XD, XR = Xl[lab == "NORMAL"], Xl[lab == "DISCLAIMER"], Xl[lab == "REFUSAL"]
    target_n = min(len(XR), len(XD))
    dD = S2.balanced_dim(XD, XN, target_n, draws, seed)
    dR = S2.balanced_dim(XR, XN, target_n, draws, seed + 1)
    return dD, dR


def top_by(table: list[dict], key: str, k: int) -> list[int]:
    """Top-k steerable rows by one AUROC column, step 2's tie-break (shallower wins)."""
    cand = [r for r in table if r["steerable"] and r.get(key) is not None]
    if len(cand) < k:
        S2.fail(f"only {len(cand)} steerable rows with {key}, need {k}")
    cand.sort(key=lambda r: (-r[key], r["layer_row"]))
    return sorted(int(r["layer_row"]) for r in cand[:k])


def resolve_layers(cfg, ctx, args, splits) -> tuple[list[int], dict]:
    """(layers, provenance). The staleness guard on selected_layers.json runs first in
    every mode, so a retired probe table can never feed the union."""
    selected = S2.load_selected_layers(cfg, ctx.slug, study_sha=splits.get("sha256"))
    if args.layers:
        layers = sorted(int(x) for x in args.layers.split(","))
        return layers, {"layer_set": "manual"}
    if args.layer_set == "selected":
        return selected, {"layer_set": "selected (step 2 top-k by mean AUROC)"}
    if args.layer_set == "all":
        return list(range(1, ctx.n_blocks)), {"layer_set": "all steerable rows"}
    doc = json.loads((S2.study_dir(cfg) / "selected_layers.json").read_text(encoding="utf-8"))
    table = (doc.get("per_layer_table") or {}).get(ctx.slug)
    if not table:
        S2.fail(f"{ctx.slug}: selected_layers.json has no per_layer_table entry")
    k = int(args.top_k or (cfg.get("probe") or {}).get("top_k", 10))
    top_R = top_by(table, "auroc_R_vs_rest", k)
    top_D = top_by(table, "auroc_D_vs_rest", k)
    layers = sorted(set(top_R) | set(top_D))
    return layers, {"layer_set": f"union of top-{k} AUROC(R vs rest) and top-{k} "
                                 f"AUROC(D vs rest)",
                    "top_k": k, "top_R": top_R, "top_D": top_D}


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for layer, g in df.groupby("layer", sort=True):
        c = g["cross"].to_numpy(dtype=float)
        rows.append({
            "layer": int(layer),
            "median_cross": float(np.median(c)),
            "p2_5": float(np.percentile(c, 2.5)),
            "p97_5": float(np.percentile(c, 97.5)),
        })
    return pd.DataFrame(rows)


def compute_model(cfg, ctx, args, splits, labels, out_root: Path, log) -> None:
    seed = int(args.seed)
    draws = int(args.balance_draws or cfg["directions"].get("balance_draws", 50))
    layers, layer_prov = resolve_layers(cfg, ctx, args, splits)
    for l in layers:
        if not 1 <= l <= ctx.n_blocks - 1:
            S2.fail(f"{ctx.slug}: layer row {l} is outside the steerable 1..{ctx.n_blocks - 1}")

    dir_ids = splits["splits"]["direction"]
    log(f"\n=== {ctx.slug}: split-half cross, {len(dir_ids)} Direction-split prompts, "
        f"{len(layers)} layers {layers}, {args.n_splits} splits, K={draws} ===")
    if "top_R" in layer_prov:
        log(f"  top-{layer_prov['top_k']} AUROC R: {layer_prov['top_R']}")
        log(f"  top-{layer_prov['top_k']} AUROC D: {layer_prov['top_D']}")
    X, kept = S2.load_stack(ctx, dir_ids, "prompt_last", log)
    X = X[:, layers, :]                                     # [n, len(layers), H]
    lab = (labels[labels.gen_model == ctx.slug].set_index("prompt_id")["label"]
           .reindex(kept))
    if lab.isna().any():
        S2.fail(f"{ctx.slug}: {int(lab.isna().sum())} Direction-split prompts have no C* label")
    lab = lab.to_numpy().astype(str)
    counts = {c: int((lab == c).sum()) for c in LABELS3}
    log(f"  labels (own C*): {counts}  -> ~{counts['REFUSAL'] // 2} R / "
        f"{counts['DISCLAIMER'] // 2} D per half")
    for c, n in counts.items():
        if n < 4:
            S2.fail(f"{ctx.slug}: {c} has n={n}; each half needs >= 2 rows per class")

    model_seed = sub_seed(seed, int(G.sha1(ctx.slug)[:8], 16))
    Xd = [X[:, j, :].astype(np.float64) for j in range(len(layers))]
    recs = []
    t0 = time.perf_counter()
    for s in range(int(args.n_splits)):
        rng = np.random.default_rng(sub_seed(model_seed, s))
        in_a = stratified_halves(lab, rng)
        la, lb = lab[in_a], lab[~in_a]
        for j, l in enumerate(layers):
            Xl = Xd[j]
            dDa, dRa = half_directions(Xl[in_a], la, draws, sub_seed(model_seed, s, l, 0))
            dDb, dRb = half_directions(Xl[~in_a], lb, draws, sub_seed(model_seed, s, l, 1))
            cross = 0.5 * (float(np.dot(dDa, dRb)) + float(np.dot(dRa, dDb)))
            recs.append({
                "seed_index": s, "layer": int(l), "cross": cross,
                "n_A": int(in_a.sum()), "n_B": int((~in_a).sum()),
                "n_R_A": int((la == "REFUSAL").sum()), "n_D_A": int((la == "DISCLAIMER").sum()),
                "n_R_B": int((lb == "REFUSAL").sum()), "n_D_B": int((lb == "DISCLAIMER").sum()),
            })
        if (s + 1) % 25 == 0:
            log(f"    {s + 1}/{args.n_splits} splits  {time.perf_counter() - t0:.0f}s")

    df = pd.DataFrame(recs)
    summ = summarise(df)
    out = out_root / ctx.slug
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "splits.parquet", index=False)
    summ.to_parquet(out / "summary.parquet", index=False)
    summ.to_csv(out / "summary.csv", index=False)
    S2.write_json(out / "step7.json", {
        "slug": ctx.slug, "hf_id": ctx.spec.hf_id, "written_utc": S2.utcnow(),
        "seed": seed, "n_splits": int(args.n_splits), "balance_draws": draws,
        "layers": layers, **layer_prov,
        "n_prompts": len(kept), "label_counts": counts,
        "study_split_sha256": splits.get("sha256"),
        "method": {
            "halves": "50/50 of the Direction split, stratified by own C* label",
            "estimator": "step 2 balanced difference-of-means, per half",
            "cross": "0.5*[cos(d_D(A), d_R(B)) + cos(d_R(A), d_D(B))]",
            "interval": "2.5th / 97.5th percentile of cross over splits",
        },
        "summary": summ.to_dict(orient="records"),
    })
    log(f"  {len(df)} (split, layer) rows in {time.perf_counter() - t0:.0f}s -> "
        f"{S2.rel_to_root(out)}")
    log(f"\n  {'layer':>5} {'median':>8} {'p2.5':>8} {'p97.5':>8}")
    for r in summ.itertuples():
        log(f"  {r.layer:>5} {r.median_cross:>8.3f} {r.p2_5:>8.3f} {r.p97_5:>8.3f}")
    plot_model(ctx.slug, summ, df, out, log)


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _dot_whisker(ax, summ: pd.DataFrame, label=None, color="C0"):
    x = np.arange(len(summ))
    med = summ["median_cross"].to_numpy()
    lo = med - summ["p2_5"].to_numpy()
    hi = summ["p97_5"].to_numpy() - med
    ax.errorbar(x, med, yerr=[lo, hi], fmt="o", color=color, capsize=4, ms=6,
                lw=1.4, label=label)


def plot_model(slug: str, summ: pd.DataFrame, df: pd.DataFrame, out: Path, log) -> None:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(max(6.0, 0.55 * len(summ) + 2), 4.2))
    _dot_whisker(ax, summ, label="cross: median, 95% split-half interval")
    x = np.arange(len(summ))
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.set_xticks(x, [f"L{l}" for l in summ["layer"]], fontsize=8)
    ax.set_xlabel("layer row (top-10 AUROC R  ∪  top-10 AUROC D)")
    ax.set_ylabel("cross cos(d_D, d_R)")
    n = int(df["seed_index"].nunique())
    ax.set_title(f"{slug}: split-half cross cos(d_D, d_R), {n} splits")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"step7_{slug}.{ext}", dpi=200)
    plt.close(fig)
    log(f"  plot -> {S2.rel_to_root(out / f'step7_{slug}.png')}")


def plot_combined(out_root: Path, log) -> None:
    """One panel per model that has a summary; layers differ per model, so no shared x."""
    found = sorted(p for p in out_root.glob("*/summary.parquet"))
    if not found:
        log(f"  no */summary.parquet under {S2.rel_to_root(out_root)}; nothing to plot")
        return
    plt = _pyplot()
    widths = [max(4.0, 0.4 * len(pd.read_parquet(p)) + 1.5) for p in found]
    fig, axes = plt.subplots(1, len(found), figsize=(sum(widths), 4.0), sharey=True,
                             squeeze=False, gridspec_kw={"width_ratios": widths})
    for ax, p in zip(axes[0], found):
        summ = pd.read_parquet(p)
        _dot_whisker(ax, summ)
        ax.axhline(0.0, color="0.6", lw=0.8)
        ax.set_xticks(np.arange(len(summ)), [f"L{l}" for l in summ["layer"]], fontsize=7,
                      rotation=90)
        ax.set_title(p.parent.name)
    axes[0][0].set_ylabel("cross cos(d_D, d_R)\nmedian + 95% split-half interval")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_root / f"step7_combined.{ext}", dpi=200)
    plt.close(fig)
    log(f"  combined plot -> {S2.rel_to_root(out_root / 'step7_combined.png')} "
        f"({len(found)} model(s))")


def _selfcheck() -> None:
    """Cheap invariants, run on every invocation."""
    lab = np.array(["NORMAL"] * 11 + ["DISCLAIMER"] * 5 + ["REFUSAL"] * 4)
    m = stratified_halves(lab, np.random.default_rng(0))
    for c, n in (("NORMAL", 11), ("DISCLAIMER", 5), ("REFUSAL", 4)):
        a = int(m[lab == c].sum())
        assert a in (n // 2, n // 2 + 1), (c, a)
    t = [{"layer_row": 0, "steerable": False, "auroc_R_vs_rest": 0.99, "auroc_D_vs_rest": 0.99},
         {"layer_row": 1, "steerable": True, "auroc_R_vs_rest": 0.9, "auroc_D_vs_rest": 0.6},
         {"layer_row": 2, "steerable": True, "auroc_R_vs_rest": 0.8, "auroc_D_vs_rest": 0.7},
         {"layer_row": 3, "steerable": True, "auroc_R_vs_rest": 0.8, "auroc_D_vs_rest": 0.9}]
    assert top_by(t, "auroc_R_vs_rest", 2) == [1, 2]       # tie at 0.8 -> shallower row
    assert top_by(t, "auroc_D_vs_rest", 2) == [2, 3]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--stage", choices=("compute", "plot", "all"), default="all")
    ap.add_argument("--models", default=None, help="comma-separated slugs (default: roster)")
    ap.add_argument("--layer-set", choices=("union", "selected", "all"), default="union",
                    help="union = top-k AUROC(R) | top-k AUROC(D) (default); selected = "
                         "step 2's top-k by mean AUROC; all = every steerable row")
    ap.add_argument("--top-k", type=int, default=0, help="default: probe.top_k (10)")
    ap.add_argument("--layers", default=None,
                    help="comma-separated layer rows; overrides --layer-set")
    ap.add_argument("--n-splits", type=int, default=200)
    ap.add_argument("--balance-draws", type=int, default=0,
                    help="K per half; default: directions.balance_draws (50)")
    ap.add_argument("--seed", type=int, default=None, help="default: study.seed")
    ap.add_argument("--labels", default=None, help="override the C* label table path")
    ap.add_argument("--smoke", action="store_true", help="use the config's smoke block")
    args = ap.parse_args()

    _selfcheck()
    out_root = G.repo_path("results/reports/step7_cross_smoke" if args.smoke
                           else "results/reports/step7_cross")
    log = S2.Tee(out_root / f"step7_{args.stage}.log")
    try:
        if args.stage in ("compute", "all"):
            cfg = S2.load_cfg(args.config, args.smoke, log)
            if args.seed is None:
                args.seed = int(cfg["study"]["seed"])
            only = [s.strip() for s in args.models.split(",")] if args.models else None
            ctxs = S2.model_contexts(cfg, only, log)
            splits = S2.load_study_splits(cfg, log)
            labels = S2.load_labels(cfg, args.labels, log)
            for ctx in ctxs:
                compute_model(cfg, ctx, args, splits, labels, out_root, log)
        if args.stage in ("plot", "all"):
            plot_combined(out_root, log)
        log("\ndone")
    finally:
        log.close()


if __name__ == "__main__":
    main()
