"""Step 7 -- split-half geometry: the "95% empirical split-half interval" of cos(d_D, d_R).

Step 3's single full-split S = cos(d_D, d_R) is attenuated by estimation noise in both
directions and inflated by the mu_N they share. This step replaces it with a split-half
estimate that corrects for the first and removes the second (UPDATES.pdf, Step 7):

  1. per model, per selected layer (top-10 from step 2's probe stage)
  2. the 40% Direction split is cut 50/50 into halves A and B, stratified by the model's
     own C* label so each half holds half the REFUSAL and half the DISCLAIMER prompts
  3. in each half: d_D(A), d_R(A), d_D(B), d_R(B) with step 2's balanced
     difference-of-means (NORMAL subsampled to min(n_R, n_D) of that half, K draws,
     mean of unit vectors, renormalised) -- the function is imported from step 2
  4.   within_D  = cos(d_D(A), d_D(B))
       within_R  = cos(d_R(A), d_R(B))
       cross     = 1/2 [cos(d_D(A), d_R(B)) + cos(d_R(A), d_D(B))]
       corrected = cross / sqrt(within_D * within_R), NaN if within_D * within_R <= 0
     cross pairs a direction from A with one from B, so the two never share a mu_N
     estimate; that removes step 3's shared-mu_N inflation. Dividing by the geometric
     mean of the split-half reliabilities is the Spearman disattenuation, so `corrected`
     is NOT bounded by 1 -- a value above 1 means "as aligned as the noise allows".
  5. repeated over --n-splits (200) seeds
  6. per layer: median(corrected) = the dot, 2.5th / 97.5th percentiles = the whisker
     (over non-NaN splits), plus mean(within_D), mean(within_R), mean(cross) and the
     number of NaN splits.

CPU only. Reads the `prompt_last` activations step 1 wrote and step 2's
selected_layers.json; writes nothing outside results/reports/step7[_smoke]/.

OUTPUTS (results/reports/step7/<slug>/)
  splits.parquet    one row per (seed, layer): the four cosines + corrected + half sizes
  summary.csv/.parquet  one row per layer: median, p2.5, p97.5, mean_within_D,
                    mean_within_R, mean_cross, n_nan, n_valid
  step7_<slug>.png/.pdf   dot-whisker plot
  step7.json        provenance: split sha, label counts, layers, K, n_splits, seed
and step7_combined.png/.pdf over every model with a summary (--stage plot).

Usage (one model at a time, so runs can overlap):
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


def corrected_score(within_D: float, within_R: float, cross: float) -> float:
    prod = within_D * within_R
    return float(cross / np.sqrt(prod)) if prod > 0 else float("nan")


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for layer, g in df.groupby("layer", sort=True):
        c = g["corrected"].to_numpy(dtype=float)
        ok = c[np.isfinite(c)]
        rows.append({
            "layer": int(layer),
            "median_corrected": float(np.median(ok)) if len(ok) else float("nan"),
            "p2_5": float(np.percentile(ok, 2.5)) if len(ok) else float("nan"),
            "p97_5": float(np.percentile(ok, 97.5)) if len(ok) else float("nan"),
            "mean_within_D": float(g["within_D"].mean()),
            "mean_within_R": float(g["within_R"].mean()),
            "mean_cross": float(g["cross"].mean()),
            "n_nan": int(len(c) - len(ok)),
            "n_valid": int(len(ok)),
        })
    return pd.DataFrame(rows)


def compute_model(cfg, ctx, args, splits, labels, out_root: Path, log) -> None:
    seed = int(args.seed)
    draws = int(args.balance_draws or cfg["directions"].get("balance_draws", 50))
    layers = ([int(x) for x in args.layers.split(",")] if args.layers
              else S2.load_selected_layers(cfg, ctx.slug, study_sha=splits.get("sha256")))
    for l in layers:
        if not 1 <= l <= ctx.n_blocks - 1:
            S2.fail(f"{ctx.slug}: layer row {l} is outside the steerable 1..{ctx.n_blocks - 1}")

    dir_ids = splits["splits"]["direction"]
    log(f"\n=== {ctx.slug}: split-half geometry, {len(dir_ids)} Direction-split prompts, "
        f"layers {layers}, {args.n_splits} splits, K={draws} ===")
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
            wD = float(np.dot(dDa, dDb))
            wR = float(np.dot(dRa, dRb))
            cross = 0.5 * (float(np.dot(dDa, dRb)) + float(np.dot(dRa, dDb)))
            recs.append({
                "seed_index": s, "layer": int(l),
                "within_D": wD, "within_R": wR, "cross": cross,
                "corrected": corrected_score(wD, wR, cross),
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
        "layers": layers, "n_prompts": len(kept), "label_counts": counts,
        "study_split_sha256": splits.get("sha256"),
        "method": {
            "halves": "50/50 of the Direction split, stratified by own C* label",
            "estimator": "step 2 balanced difference-of-means, per half",
            "within_D": "cos(d_D(A), d_D(B))", "within_R": "cos(d_R(A), d_R(B))",
            "cross": "0.5*[cos(d_D(A), d_R(B)) + cos(d_R(A), d_D(B))]",
            "corrected": "cross / sqrt(within_D*within_R); NaN if within_D*within_R <= 0",
            "interval": "2.5th / 97.5th percentile of non-NaN corrected over splits",
        },
        "summary": summ.to_dict(orient="records"),
    })
    log(f"  {len(df)} (split, layer) rows in {time.perf_counter() - t0:.0f}s -> "
        f"{S2.rel_to_root(out)}")
    log(f"\n  {'layer':>5} {'median':>8} {'p2.5':>8} {'p97.5':>8} {'wD':>7} {'wR':>7} "
        f"{'cross':>7} {'NaN':>4}")
    for r in summ.itertuples():
        log(f"  {r.layer:>5} {r.median_corrected:>8.3f} {r.p2_5:>8.3f} {r.p97_5:>8.3f} "
            f"{r.mean_within_D:>7.3f} {r.mean_within_R:>7.3f} {r.mean_cross:>7.3f} "
            f"{r.n_nan:>4}")
    plot_model(ctx.slug, summ, df, out, log)


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _dot_whisker(ax, summ: pd.DataFrame, label=None, color="C0"):
    x = np.arange(len(summ))
    med = summ["median_corrected"].to_numpy()
    lo = med - summ["p2_5"].to_numpy()
    hi = summ["p97_5"].to_numpy() - med
    ax.errorbar(x, med, yerr=[lo, hi], fmt="o", color=color, capsize=4, ms=6,
                lw=1.4, label=label)


def plot_model(slug: str, summ: pd.DataFrame, df: pd.DataFrame, out: Path, log) -> None:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(max(6.0, 0.7 * len(summ) + 2), 4.2))
    _dot_whisker(ax, summ, label="corrected: median, 95% split-half interval")
    x = np.arange(len(summ))
    ax.plot(x, summ["mean_cross"], "s", color="0.45", ms=4, mfc="none",
            label="mean cross (uncorrected)")
    ax.axhline(1.0, color="0.6", lw=0.8, ls="--")
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.set_xticks(x, [f"L{l}" for l in summ["layer"]])
    ax.set_xlabel("layer row (top-10 selected)")
    ax.set_ylabel("cross / sqrt(within_D x within_R)")
    n = int(df["seed_index"].nunique())
    ax.set_title(f"{slug}: split-half corrected cos(d_D, d_R), {n} splits")
    for i, r in enumerate(summ.itertuples()):
        if r.n_nan > 0:
            ax.annotate(f"NaN {r.n_nan}", (i, r.p97_5), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=7, color="C3")
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
    fig, axes = plt.subplots(1, len(found), figsize=(5.2 * len(found), 4.0),
                             sharey=True, squeeze=False)
    for ax, p in zip(axes[0], found):
        summ = pd.read_parquet(p)
        _dot_whisker(ax, summ)
        ax.axhline(1.0, color="0.6", lw=0.8, ls="--")
        ax.axhline(0.0, color="0.6", lw=0.8)
        ax.set_xticks(np.arange(len(summ)), [f"L{l}" for l in summ["layer"]], fontsize=8)
        ax.set_title(p.parent.name)
    axes[0][0].set_ylabel("corrected cos(d_D, d_R)\nmedian + 95% split-half interval")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_root / f"step7_combined.{ext}", dpi=200)
    plt.close(fig)
    log(f"  combined plot -> {S2.rel_to_root(out_root / 'step7_combined.png')} "
        f"({len(found)} model(s))")


def _selfcheck() -> None:
    """Cheap invariants of the arithmetic, run on every invocation."""
    assert corrected_score(0.81, 0.64, 0.36) == 0.36 / np.sqrt(0.81 * 0.64)
    assert np.isnan(corrected_score(-0.1, 0.9, 0.5))
    assert np.isnan(corrected_score(0.0, 0.9, 0.5))
    lab = np.array(["NORMAL"] * 11 + ["DISCLAIMER"] * 5 + ["REFUSAL"] * 4)
    m = stratified_halves(lab, np.random.default_rng(0))
    for c, n in (("NORMAL", 11), ("DISCLAIMER", 5), ("REFUSAL", 4)):
        a = int(m[lab == c].sum())
        assert a in (n // 2, n // 2 + 1), (c, a)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--stage", choices=("compute", "plot", "all"), default="all")
    ap.add_argument("--models", default=None, help="comma-separated slugs (default: roster)")
    ap.add_argument("--layers", default=None,
                    help="comma-separated layer rows; default: step 2's top-10")
    ap.add_argument("--n-splits", type=int, default=200)
    ap.add_argument("--balance-draws", type=int, default=0,
                    help="K per half; default: directions.balance_draws (50)")
    ap.add_argument("--seed", type=int, default=None, help="default: study.seed")
    ap.add_argument("--labels", default=None, help="override the C* label table path")
    ap.add_argument("--smoke", action="store_true", help="use the config's smoke block")
    args = ap.parse_args()

    _selfcheck()
    out_root = G.repo_path("results/reports/step7_smoke" if args.smoke
                           else "results/reports/step7")
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
