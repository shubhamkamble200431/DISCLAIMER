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

from src import generate as G

DEFAULT_CONFIG = "configs/study.yaml"
LABELS3 = ("NORMAL", "DISCLAIMER", "REFUSAL")


def import_directions_steering():
    path = ROOT / "scripts" / "directions_steering.py"
    spec = importlib.util.spec_from_file_location("directions_steering", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


DS = import_directions_steering()


def sub_seed(*parts: int) -> int:
    return int(np.random.SeedSequence([int(p) for p in parts]).generate_state(1)[0])


def stratified_halves(lab: np.ndarray, rng: np.random.Generator) -> np.ndarray:
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
    XN, XD, XR = Xl[lab == "NORMAL"], Xl[lab == "DISCLAIMER"], Xl[lab == "REFUSAL"]
    target_n = min(len(XR), len(XD))
    dD = DS.balanced_dim(XD, XN, target_n, draws, seed)
    dR = DS.balanced_dim(XR, XN, target_n, draws, seed + 1)
    return dD, dR


def top_by(table: list[dict], key: str, k: int) -> list[int]:
    cand = [r for r in table if r["steerable"] and r.get(key) is not None]
    if len(cand) < k:
        DS.fail(f"only {len(cand)} steerable rows with {key}, need {k}")
    cand.sort(key=lambda r: (-r[key], r["layer_row"]))
    return sorted(int(r["layer_row"]) for r in cand[:k])


def resolve_layers(cfg, ctx, args, splits) -> tuple[list[int], dict]:
    selected = DS.load_selected_layers(cfg, ctx.slug, study_sha=splits.get("sha256"))
    if args.layers:
        layers = sorted(int(x) for x in args.layers.split(","))
        return layers, {"layer_set": "manual"}
    if args.layer_set == "selected":
        return selected, {"layer_set": "selected (directions_steering's own union of top-k "
                                       "AUROC(R)/top-k AUROC(D))"}
    if args.layer_set == "all":
        return list(range(1, ctx.n_blocks)), {"layer_set": "all steerable rows"}
    doc = json.loads((DS.study_dir(cfg) / "selected_layers.json").read_text(encoding="utf-8"))
    table = (doc.get("per_layer_table") or {}).get(ctx.slug)
    if not table:
        DS.fail(f"{ctx.slug}: selected_layers.json has no per_layer_table entry")
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
            DS.fail(f"{ctx.slug}: layer row {l} is outside the steerable 1..{ctx.n_blocks - 1}")

    dir_ids = splits["splits"]["direction"]
    log(f"\n=== {ctx.slug}: split-half cross, {len(dir_ids)} Direction-split prompts, "
        f"{len(layers)} layers {layers}, {args.n_splits} splits, K={draws} ===")
    if "top_R" in layer_prov:
        log(f"  top-{layer_prov['top_k']} AUROC R: {layer_prov['top_R']}")
        log(f"  top-{layer_prov['top_k']} AUROC D: {layer_prov['top_D']}")
    X, kept = DS.load_stack(ctx, dir_ids, "prompt_last", log)
    X = X[:, layers, :]
    lab = (labels[labels.gen_model == ctx.slug].set_index("prompt_id")["label"]
           .reindex(kept))
    if lab.isna().any():
        DS.fail(f"{ctx.slug}: {int(lab.isna().sum())} Direction-split prompts have no C* label")
    lab = lab.to_numpy().astype(str)
    counts = {c: int((lab == c).sum()) for c in LABELS3}
    log(f"  labels (own C*): {counts}  -> ~{counts['REFUSAL'] // 2} R / "
        f"{counts['DISCLAIMER'] // 2} D per half")
    for c, n in counts.items():
        if n < 4:
            DS.fail(f"{ctx.slug}: {c} has n={n}; each half needs >= 2 rows per class")

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
    DS.write_json(out / "splithalf_geometry_cross.json", {
        "slug": ctx.slug, "hf_id": ctx.spec.hf_id, "written_utc": DS.utcnow(),
        "seed": seed, "n_splits": int(args.n_splits), "balance_draws": draws,
        "layers": layers, **layer_prov,
        "n_prompts": len(kept), "label_counts": counts,
        "study_split_sha256": splits.get("sha256"),
        "method": {
            "halves": "50/50 of the Direction split, stratified by own C* label",
            "estimator": "directions_steering balanced difference-of-means, per half",
            "cross": "0.5*[cos(d_D(A), d_R(B)) + cos(d_R(A), d_D(B))]",
            "interval": "2.5th / 97.5th percentile of cross over splits",
        },
        "summary": summ.to_dict(orient="records"),
    })
    log(f"  {len(df)} (split, layer) rows in {time.perf_counter() - t0:.0f}s -> "
        f"{DS.rel_to_root(out)}")
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
        fig.savefig(out / f"splithalf_geometry_cross_{slug}.{ext}", dpi=200)
    plt.close(fig)
    log(f"  plot -> {DS.rel_to_root(out / f'splithalf_geometry_cross_{slug}.png')}")


def plot_combined(out_root: Path, log) -> None:
    found = sorted(p for p in out_root.glob("*/summary.parquet"))
    if not found:
        log(f"  no */summary.parquet under {DS.rel_to_root(out_root)}; nothing to plot")
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
        fig.savefig(out_root / f"splithalf_geometry_cross_combined.{ext}", dpi=200)
    plt.close(fig)
    log(f"  combined plot -> {DS.rel_to_root(out_root / 'splithalf_geometry_cross_combined.png')} "
        f"({len(found)} model(s))")


def _selfcheck() -> None:
    lab = np.array(["NORMAL"] * 11 + ["DISCLAIMER"] * 5 + ["REFUSAL"] * 4)
    m = stratified_halves(lab, np.random.default_rng(0))
    for c, n in (("NORMAL", 11), ("DISCLAIMER", 5), ("REFUSAL", 4)):
        a = int(m[lab == c].sum())
        assert a in (n // 2, n // 2 + 1), (c, a)
    t = [{"layer_row": 0, "steerable": False, "auroc_R_vs_rest": 0.99, "auroc_D_vs_rest": 0.99},
         {"layer_row": 1, "steerable": True, "auroc_R_vs_rest": 0.9, "auroc_D_vs_rest": 0.6},
         {"layer_row": 2, "steerable": True, "auroc_R_vs_rest": 0.8, "auroc_D_vs_rest": 0.7},
         {"layer_row": 3, "steerable": True, "auroc_R_vs_rest": 0.8, "auroc_D_vs_rest": 0.9}]
    assert top_by(t, "auroc_R_vs_rest", 2) == [1, 2]
    assert top_by(t, "auroc_D_vs_rest", 2) == [2, 3]


def main() -> None:
    ap = argparse.ArgumentParser(description="splithalf_geometry_cross -- split-half geometry: median cross cos(d_D, d_R) with a 95% split-half interval.")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--stage", choices=("compute", "plot", "all"), default="all")
    ap.add_argument("--models", default=None, help="comma-separated slugs (default: roster)")
    ap.add_argument("--layer-set", choices=("union", "selected", "all"), default="union",
                    help="union = top-k AUROC(R) | top-k AUROC(D) (default); selected = "
                         "directions_steering.py's top-k by mean AUROC; all = every steerable row")
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
    out_root = G.repo_path("results/reports/splithalf_geometry_cross_smoke" if args.smoke
                           else "results/reports/splithalf_geometry_cross")
    log = DS.Tee(out_root / f"splithalf_geometry_cross_{args.stage}.log")
    try:
        if args.stage in ("compute", "all"):
            cfg = DS.load_cfg(args.config, args.smoke, log)
            if args.seed is None:
                args.seed = int(cfg["study"]["seed"])
            only = [s.strip() for s in args.models.split(",")] if args.models else None
            ctxs = DS.model_contexts(cfg, only, log)
            splits = DS.load_study_splits(cfg, log)
            labels = DS.load_labels(cfg, args.labels, log)
            for ctx in ctxs:
                compute_model(cfg, ctx, args, splits, labels, out_root, log)
        if args.stage in ("plot", "all"):
            plot_combined(out_root, log)
        log("\ndone")
    finally:
        log.close()


if __name__ == "__main__":
    main()
