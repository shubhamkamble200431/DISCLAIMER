"""Step 9 -- random-direction control: is d_D / d_R better than a random push of the same size?

At each model's chosen layer, steer with K random unit vectors (isotropic Gaussian draw,
L2-normalised) through step 2's EXACT steering path, score them with the same frozen C*,
and compare their percentage-point change against d_D / d_R at the same layer and alpha.

L2 MATCH. d_D and d_R are unit vectors and so is every random vector, so at the same alpha
the perturbation alpha * v has the same L2 norm |alpha| for all of them. Nothing else is
matched: the random vector's direction is uniform on the sphere.

LAYER CHOICE (user rule, 2026-10-04). From step 2's scored sweep, per direction, per layer:
  pp(alpha) = % target class at (layer, alpha) - % target class at the alpha=0 baseline
              (target: DISCLAIMER for dD, REFUSAL for dR)
  score     = |most negative pp| + highest pp   (each side counts only if it has that sign)
The layer with the highest score is picked, separately for dD and for dR. Random vectors are
steered at the union of the picked layers (one layer if dD and dR agree). The table is
recomputed here from data/steered/<slug>/predictions.parquet and written out; EXPECTED_LAYERS
is what the laptop computed from the same files and only produces a warning on mismatch.

STAGES (one model per run, so models can run side by side on separate GPUs)
  steer    GPU (base)        random vectors -> data/study/random_control/<slug>_L<layer>.{npz,json}
                             answers        -> data/steered_random/<slug>/<tag>/answers.jsonl
                             tags look like L022_rand0_ap20 (step 2's tag grammar).
                             The alpha=0 baseline is NOT regenerated: step 2's baseline_ap0
                             is reused, and the run refuses to start unless its batch plan
                             hash equals that baseline's (batch size read from its meta.json).
  score    GPU (refusal_clf) frozen C* Longformer via step 2's own scoring code (val-parity
                             gate included) -> data/steered_random/<slug>/predictions.parquet
  metrics  CPU               -> results/reports/step9/<slug>/
                             layer_selection.csv     the rule above, every layer, dD and dR
                             pp_long.csv             every (layer, direction, alpha, class) pp
                             comparison.csv          per alpha: found pp, rand_k pp,
                                                     rand mean, rand std (ddof=1)
                             scores.csv              the selection score for found vs random
                             step9_<slug>.xlsx       the tables above
                             step9_<slug>.png/.pdf   found curve vs random mean +- std
  all                        steer + score + metrics (laptop smoke only: the DGX needs two
                             conda envs, see the README)

Usage:
  python -X utf8 scripts/step9_random_control.py --models olmo3-7b --stage steer
  python -X utf8 scripts/step9_random_control.py --models olmo3-7b --stage score
  python -X utf8 scripts/step9_random_control.py --models olmo3-7b --stage metrics
  python -X utf8 scripts/step9_random_control.py --models olmo3-7b --stage steer --dry-run
  python -X utf8 scripts/step9_random_control.py --smoke --models smoke-olmo2-1b --stage all
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import generate as G  # noqa: E402
from src import steer as S     # noqa: E402

DEFAULT_CONFIG = "configs/study.yaml"
DEFAULT_DRAWS = 2              # user decision 2026-10-04 (K=2, ~28 GPU-h for all 3 models)
LABEL_COL = "longformer_label"
CLASSES = ("NORMAL", "DISCLAIMER", "REFUSAL")
TARGET = {"dD": "DISCLAIMER", "dR": "REFUSAL"}
FOUND = ("dD", "dR")
# Computed on the laptop 2026-10-04 from the scored step-2 predictions with the rule in the
# docstring. A mismatch is logged, never enforced: the live table below is what is used.
EXPECTED_LAYERS = {
    "olmo3-7b": {"dD": 22, "dR": 22},
    "qwen3-5-4b": {"dD": 18, "dR": 18},
    "qwen3-5-9b": {"dD": 20, "dR": 19},
}


def import_step2():
    path = ROOT / "scripts" / "step2_directions_steering.py"
    spec = importlib.util.spec_from_file_location("step2_directions_steering", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


S2 = import_step2()
fail = S2.fail


# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
def random_cfg(cfg) -> dict:
    """Copy of cfg whose steered_root is the random-control tree, so step 2's own
    generation/scoring code writes there and never touches the main sweep."""
    rc = json.loads(json.dumps(cfg))
    rc["output"]["steered_root"] = str(cfg["output"]["steered_root"]).rstrip("/") + "_random"
    return rc


def main_steered(cfg, slug: str) -> Path:
    return S2.steered_dir(cfg) / slug


def vectors_paths(cfg, slug: str, layer: int) -> tuple[Path, Path]:
    # One file per layer, so two jobs steering different layers of one model never write
    # the same file.
    d = S2.study_dir(cfg) / "random_control"
    return d / f"{slug}_L{layer:03d}.npz", d / f"{slug}_L{layer:03d}.json"


def report_dir(cfg, slug: str) -> Path:
    base = "results/reports/step9_smoke" if cfg.get("_smoke") else "results/reports/step9"
    return G.repo_path(base) / slug


# --------------------------------------------------------------------------- #
# the selection rule
# --------------------------------------------------------------------------- #
def range_score(pp) -> float:
    """|most negative pp| + highest pp; a side only counts if it has that sign."""
    pp = np.asarray(pp, dtype=float)
    return float(max(pp.max(), 0.0) + max(-pp.min(), 0.0))


def class_pct(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """% of rows per class, one column per class (all three always present)."""
    out = (df.groupby(by)[LABEL_COL].value_counts(normalize=True)
           .unstack(fill_value=0.0) * 100.0)
    for c in CLASSES:
        if c not in out.columns:
            out[c] = 0.0
    out = out[list(CLASSES)]
    out.columns.name = "class"
    return out


def load_main_predictions(cfg, slug: str) -> pd.DataFrame:
    p = main_steered(cfg, slug) / "predictions.parquet"
    if not p.exists():
        fail(f"{slug}: no {S2.rel_to_root(p)}. Step 2 steer + score must be complete "
             f"before the random control (it needs the baseline and the dD/dR cells).")
    df = pd.read_parquet(p, columns=["prompt_id", "condition_tag", "layer", "direction",
                                     "alpha", LABEL_COL])
    if (df.condition_tag == S.BASELINE_TAG).sum() == 0:
        fail(f"{slug}: {S2.rel_to_root(p)} has no {S.BASELINE_TAG} rows")
    return df


def baseline_pct(main: pd.DataFrame) -> pd.Series:
    b = main[main.condition_tag == S.BASELINE_TAG]
    return b[LABEL_COL].value_counts(normalize=True).reindex(CLASSES, fill_value=0.0) * 100.0


def layer_selection(main: pd.DataFrame, log) -> tuple[pd.DataFrame, dict]:
    base = baseline_pct(main)
    s = main[main.direction.isin(FOUND)]
    pct = class_pct(s, ["direction", "layer", "alpha"])
    rows, picks = [], {}
    for d in FOUND:
        t = TARGET[d]
        if d not in pct.index.get_level_values(0):
            fail(f"step-2 predictions have no {d} cells")
        pp = (pct.loc[d][t] - base[t]).unstack("alpha")
        for layer, r in pp.iterrows():
            v = r.to_numpy()
            rows.append({"direction": d, "target": t, "layer": int(layer),
                         "min_pp": round(float(v.min()), 2),
                         "max_pp": round(float(v.max()), 2),
                         "score": round(range_score(v), 2),
                         **{f"pp_a{a:g}": round(float(x), 2) for a, x in r.items()}})
    sel = pd.DataFrame(rows).sort_values(["direction", "score", "layer"],
                                         ascending=[True, False, True])
    sel["rank"] = sel.groupby("direction").cumcount() + 1
    for d in FOUND:
        picks[d] = int(sel[(sel.direction == d) & (sel["rank"] == 1)].layer.iloc[0])
    log("  layer selection (score = |most negative pp| + highest pp, target-class pp vs "
        "the alpha=0 baseline):")
    for d in FOUND:
        top = sel[sel.direction == d].head(3)
        log(f"    {d} -> {TARGET[d]}: " + ", ".join(
            f"L{r.layer} {r.score:.1f} (min {r.min_pp:+.1f}, max {r.max_pp:+.1f})"
            for r in top.itertuples()) + f"  => picked L{picks[d]}")
    return sel, picks


def resolve_layers(cfg, args, ctx, log) -> tuple[list[int], dict, pd.DataFrame]:
    main = load_main_predictions(cfg, ctx.slug)
    sel, picks = layer_selection(main, log)
    exp = EXPECTED_LAYERS.get(ctx.slug)
    if exp and exp != picks:
        log(f"  !! WARNING: picked {picks} but the laptop computed {exp} from the same "
            f"rule. Using the live picks; check that step 2's predictions are the final ones.")
    layers = sorted(set(picks.values()))
    if args.layers:
        want = sorted({int(x) for x in args.layers.split(",") if x.strip()})
        extra = [l for l in want if l not in layers]
        if extra:
            log(f"  !! --layers adds {extra}, which the rule did not pick")
        layers = want
    for l in layers:
        S.assert_steerable(l, ctx.n_blocks)
    log(f"  random directions will be steered at layer(s) {layers}")
    return layers, picks, sel


# --------------------------------------------------------------------------- #
# random vectors
# --------------------------------------------------------------------------- #
def draw_seed(base: int, slug: str, layer: int, k: int) -> int:
    # Fixed per (model, layer, draw): re-running one layer reproduces bit-for-bit and adding
    # a layer or a draw never shifts another vector.
    return int(base) + 7_919 * (int(layer) + 1) + 104_729 * int(k) + int(G.sha1(slug)[:6], 16)


def make_vectors(cfg, args, ctx, layers: list[int], log) -> dict:
    """Draw (deterministically) and record the random vectors; a re-run redraws the same
    vectors and refuses to continue if the file on disk disagrees."""
    arrays, ddoc = S2.load_directions(cfg, ctx.slug, log)
    out = {}
    for l in layers:
        npz_path, json_path = vectors_paths(cfg, ctx.slug, l)
        old = {}
        if npz_path.exists():
            with np.load(npz_path) as z:
                old = {k: z[k] for k in z.files}
        dD = arrays[f"dD_L{l:03d}"].astype(np.float64)
        dR = arrays[f"dR_L{l:03d}"].astype(np.float64)
        layer_vecs, meta = {}, {}
        for k in range(args.draws):
            sd = draw_seed(args.seed, ctx.slug, l, k)
            v = S.unit(np.random.default_rng(sd).standard_normal(ctx.hidden_size))
            key = f"rand{k}_L{l:03d}"
            v32 = v.astype(np.float32)
            if key in old and not np.array_equal(old[key], v32):
                fail(f"{key} on disk differs from its deterministic redraw (seed {sd}); "
                     f"move {S2.rel_to_root(npz_path)} aside and investigate")
            layer_vecs[key] = v32
            meta[key] = {"layer": l, "draw": k, "seed": sd,
                         "norm": round(float(np.linalg.norm(v32)), 6),
                         "norm_dD": round(float(np.linalg.norm(dD)), 6),
                         "norm_dR": round(float(np.linalg.norm(dR)), 6),
                         "cos_dD": round(float(v @ dD), 4),
                         "cos_dR": round(float(v @ dR), 4)}
            log(f"    {key}: seed {sd}  |v|={meta[key]['norm']:.6f} "
                f"(|dD|={meta[key]['norm_dD']:.6f}, |dR|={meta[key]['norm_dR']:.6f})  "
                f"cos(v,dD)={meta[key]['cos_dD']:+.4f} cos(v,dR)={meta[key]['cos_dR']:+.4f}")
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(npz_path, **{**old, **layer_vecs})
        S2.write_json(json_path, {
            "slug": ctx.slug, "layer": l, "written_utc": S2.utcnow(),
            "method": "v = unit(N(0, I_H)), one numpy default_rng seed per (model, layer, "
                      "draw); L2-matched to dD/dR because all are unit and share alpha",
            "hidden_size": ctx.hidden_size,
            "directions_sha256": ddoc.get("sha256"),
            "vectors": meta, "npz_sha256": S2.sha256_file(npz_path),
        })
        log(f"  wrote {S2.rel_to_root(npz_path)}")
        out.update(layer_vecs)
    return out


# --------------------------------------------------------------------------- #
# stage: steer
# --------------------------------------------------------------------------- #
def steering_rows(cfg, splits, args, log):
    """Step 2's Steering-split rows, built exactly as stage_steer builds them."""
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
    return rows


def stage_steer(cfg, args, ctx, splits, log) -> None:
    rcfg = random_cfg(cfg)
    layers, picks, _ = resolve_layers(cfg, args, ctx, log)
    alphas = S2.nonzero_alphas(cfg)
    if args.alphas:
        want = {float(x) for x in args.alphas.split(",") if x.strip()}
        alphas = [a for a in alphas if a in want]
    conds = [(S.condition_tag(l, f"rand{k}", a), l, f"rand{k}", float(a))
             for l in layers for k in range(args.draws) for a in alphas]

    bmeta_path = main_steered(cfg, ctx.slug) / S.BASELINE_TAG / "meta.json"
    if not bmeta_path.exists():
        fail(f"{ctx.slug}: no {S2.rel_to_root(bmeta_path)}; the step-2 baseline is the "
             f"subtrahend and must exist")
    bmeta = json.loads(bmeta_path.read_text(encoding="utf-8"))
    batch_size = int(args.batch_size or bmeta["batch_size_requested"])
    rows = steering_rows(cfg, splits, args, log)
    log(f"\n=== {ctx.slug}: random-direction control ===")
    log(f"  layers={layers} draws={args.draws} alphas={alphas}")
    log(f"  {len(conds)} condition(s) x {len(rows)} prompts = {len(conds) * len(rows):,} "
        f"generations; batch {batch_size} (step-2 baseline used "
        f"{bmeta['batch_size_requested']})")
    if args.dry_run:
        sec = [json.loads(p.read_text(encoding="utf-8")).get("seconds", 0)
               for l in layers for p in main_steered(cfg, ctx.slug).glob(f"L{l:03d}_*/meta.json")]
        if sec:
            per_cell = float(np.median(sec))
            log(f"  DRY RUN: step-2 cells at these layers took a median {per_cell / 60:.1f} "
                f"min each -> about {per_cell * len(conds) / 3600:.1f} GPU-h for this run")
        for t, *_ in conds:
            log(f"    {t}")
        return

    vecs = make_vectors(cfg, args, ctx, layers, log)
    _, ddoc = S2.load_directions(cfg, ctx.slug, log)
    pre = S2.require_preflight(cfg, ctx, args, log)

    runner = G.Runner(ctx.spec, cfg["generation"], cfg["capture"], log=log)
    try:
        if int(runner.n_blocks) != ctx.n_blocks or int(runner.hidden_size) != ctx.hidden_size:
            fail(f"{ctx.slug}: loaded model geometry does not match the manifest")
        gen = S.BatchedGenerator(runner, batch_size=batch_size,
                                 max_prompt_tokens=cfg["steering"].get("max_prompt_tokens"),
                                 log=log)
        batches = gen.plan_batches(rows)
        plan_hash = S2.batch_plan_hash(rows, batches, gen.batch_size)
        if plan_hash != bmeta.get("batch_plan_hash"):
            msg = (f"{ctx.slug}: batch plan {plan_hash[:16]} != step-2 baseline's "
                   f"{str(bmeta.get('batch_plan_hash'))[:16]}. Batched greedy decoding is "
                   f"not invariant to batch composition, so the pp vs that baseline would "
                   f"be invalid. Use the same --batch-size and prompt set as step 2 "
                   f"(baseline meta says {bmeta['batch_size_requested']}).")
            if not args.allow_plan_mismatch:
                fail(msg)
            log(f"  !! --allow-plan-mismatch (TESTING ONLY): {msg}")
        else:
            log(f"  batch plan {plan_hash[:16]} == step-2 baseline's ({len(batches)} batches)")
        S2.assert_plan_consistent(rcfg, ctx, plan_hash, log)

        done, t_start = 0, time.perf_counter()
        for ci, (tag, layer, dname, alpha) in enumerate(conds, 1):
            cdir = S2.condition_dir(rcfg, ctx.slug, tag)
            ans = cdir / "answers.jsonl"
            have = S.read_done_ids(ans)
            want = {int(r.prompt_id) for r in rows}
            if have >= want:
                log(f"  [{ci}/{len(conds)}] {tag}: complete ({len(have)}), skip")
                continue
            if have:
                log(f"  [{ci}/{len(conds)}] {tag}: partial ({len(have)}/{len(want)}); "
                    f"redoing the whole condition (static batch membership)")
                ans.unlink()
            vec = vecs[f"{dname}_L{layer:03d}"]
            sig = {d: S2.sigma_for(ddoc, layer, d) for d in FOUND}
            hook_ctx = (lambda m=runner.model, l=layer, v=vec, a=alpha:
                        S.steering_hook(m, l, v, a, tok=runner.tok, n_blocks=ctx.n_blocks))
            log(f"  [{ci}/{len(conds)}] {tag}  layer={layer} dir={dname} alpha={alpha}")
            t0 = time.perf_counter()
            records = gen.generate(rows, batches, hook_ctx=hook_ctx,
                                   extra={"condition": tag, "layer": layer,
                                          "direction": dname, "alpha": float(alpha)})
            cdir.mkdir(parents=True, exist_ok=True)
            S.write_answers_jsonl(ans, records, mode="w")
            S2.write_json(cdir / "meta.json", {
                "condition": tag, "layer": layer, "direction": dname, "alpha": float(alpha),
                "control": "random-unit, L2-matched (unit vector, same alpha)",
                "vector_norm": float(np.linalg.norm(vec)),
                "sigma_dD": sig["dD"], "sigma_dR": sig["dR"],
                "seed": int(args.seed), "n_rows": len(rows), "n_records": len(records),
                "batch_plan_hash": plan_hash,
                "baseline_batch_plan_hash": bmeta.get("batch_plan_hash"),
                "batch_size_requested": gen.requested_batch_size,
                "batch_size_effective": gen.batch_size,
                "n_batches": len(batches), "n_padded_repeats": gen.n_padded_repeats,
                "n_oom_halvings": gen.n_oom_halvings,
                "limited_run": bool(args.limit or cfg.get("_smoke")),
                "generation": dict(runner.gen_kwargs),
                "model": {"slug": ctx.slug, "hf_id": ctx.spec.hf_id},
                "directions_sha256": ddoc.get("sha256"),
                "study_split_sha256": splits.get("sha256"),
                "preflight": {"skipped": bool(pre.get("skipped"))},
                "answers_sha256": S2.sha256_file(ans),
                "seconds": round(time.perf_counter() - t0, 1),
                "written_utc": S2.utcnow(),
            })
            done += len(records)
            el = time.perf_counter() - t_start
            rate = done / max(el, 1e-9)
            log(f"      {len(records)} answers in {time.perf_counter() - t0:.0f}s | "
                f"{el / 3600:.2f} h so far | "
                f"~{(len(conds) - ci) * len(rows) / max(rate, 1e-9) / 3600:.1f} h left")
    finally:
        del runner
        S2._empty_cuda_cache()


# --------------------------------------------------------------------------- #
# stage: score
# --------------------------------------------------------------------------- #
def stage_score(cfg, args, ctx, log) -> None:
    rcfg = random_cfg(cfg)
    sargs = SimpleNamespace(shard_rows=args.shard_rows, token_budget=args.token_budget,
                            skip_val_parity=args.skip_val_parity)
    S2.stage_score(rcfg, sargs, [ctx], log)


# --------------------------------------------------------------------------- #
# stage: metrics
# --------------------------------------------------------------------------- #
def stage_metrics(cfg, args, ctx, log) -> None:
    rcfg = random_cfg(cfg)
    out = report_dir(cfg, ctx.slug)
    out.mkdir(parents=True, exist_ok=True)
    main = load_main_predictions(cfg, ctx.slug)
    sel, picks = layer_selection(main, log)
    sel.to_csv(out / "layer_selection.csv", index=False)

    rp = S2.predictions_path(rcfg, ctx.slug)
    if not rp.exists():
        fail(f"{ctx.slug}: no {S2.rel_to_root(rp)}; run --stage score first")
    rnd = pd.read_parquet(rp, columns=["prompt_id", "condition_tag", "layer", "direction",
                                       "alpha", LABEL_COL])
    rnd = rnd[rnd.direction.astype(str).str.startswith("rand")]
    base_ids = set(main.loc[main.condition_tag == S.BASELINE_TAG, "prompt_id"])
    for tag, g in rnd.groupby("condition_tag"):
        if set(g.prompt_id) != base_ids:
            fail(f"{tag}: its prompt set differs from the step-2 baseline's "
                 f"({len(set(g.prompt_id))} vs {len(base_ids)})")
    layers = sorted(int(l) for l in rnd.layer.unique())
    draws = sorted(rnd.direction.unique(), key=lambda s: int(s[4:]))
    n_alpha = rnd.groupby(["layer", "direction"]).alpha.nunique()
    log(f"  random rows: {len(rnd):,} over layers {layers}, draws {draws}, "
        f"alphas per (layer, draw) {sorted(set(n_alpha))}")

    base = baseline_pct(main)
    both = pd.concat([main[main.direction.isin(FOUND) & main.layer.isin(layers)], rnd])
    pct = class_pct(both, ["layer", "direction", "alpha"])
    long = (pct - base).stack().rename("pp").reset_index()
    long["layer"] = long.layer.astype(int)
    long["pp"] = long.pp.round(3)
    long.to_csv(out / "pp_long.csv", index=False)

    _, ddoc = S2.load_directions(cfg, ctx.slug, log)
    comp_rows, score_rows = [], []
    for l in layers:
        for d in FOUND:
            t = TARGET[d]
            sig = S2.sigma_for(ddoc, l, d)
            primary = picks[d] == l
            g = long[(long.layer == l) & (long["class"] == t)]
            wide = g.pivot(index="alpha", columns="direction", values="pp")
            if d not in wide.columns:
                continue
            wide = wide.dropna(subset=[d] + draws)
            for a, r in wide.iterrows():
                rv = r[draws].to_numpy(dtype=float)
                comp_rows.append({
                    "layer": l, "found_direction": d, "target": t,
                    "picked_for_this_direction": primary,
                    "alpha": a, "alpha_over_sigma": round(a / sig, 4) if sig else None,
                    "found_pp": round(float(r[d]), 2),
                    **{f"{k}_pp": round(float(r[k]), 2) for k in draws},
                    "rand_mean_pp": round(float(rv.mean()), 2),
                    "rand_std_pp": round(float(rv.std(ddof=1)), 2) if len(rv) > 1 else None,
                    "found_minus_rand_mean": round(float(r[d] - rv.mean()), 2),
                })
            rs = np.array([range_score(wide[k].to_numpy()) for k in draws])
            score_rows.append({
                "layer": l, "found_direction": d, "target": t,
                "picked_for_this_direction": primary,
                "found_score": round(range_score(wide[d].to_numpy()), 2),
                **{f"{k}_score": round(float(x), 2) for k, x in zip(draws, rs)},
                "rand_mean_score": round(float(rs.mean()), 2),
                "rand_std_score": round(float(rs.std(ddof=1)), 2) if len(rs) > 1 else None,
            })
    comp = pd.DataFrame(comp_rows)
    scores = pd.DataFrame(score_rows)
    comp.to_csv(out / "comparison.csv", index=False)
    scores.to_csv(out / "scores.csv", index=False)

    log(f"\n  found vs random, target-class pp (std over {len(draws)} draws, ddof=1):")
    for r in scores[scores.picked_for_this_direction].itertuples():
        sd = f"{r.rand_std_score:.1f}" if r.rand_std_score is not None else "--"
        log(f"    L{r.layer} {r.found_direction}->{r.target}: score found {r.found_score:.1f} "
            f"vs random {r.rand_mean_score:.1f} +- {sd}")
        c = comp[(comp.layer == r.layer) & (comp.found_direction == r.found_direction)]
        for x in c.itertuples():
            sd = f"{x.rand_std_pp:.1f}" if x.rand_std_pp is not None else "--"
            log(f"      alpha {x.alpha:>+6g}: found {x.found_pp:+6.1f}  random "
                f"{x.rand_mean_pp:+6.1f} +- {sd}")

    try:
        with pd.ExcelWriter(out / f"step9_{ctx.slug}.xlsx") as xw:
            pd.DataFrame({"README": [
                f"Random-direction control, {ctx.slug}. pp = % target class minus % at the "
                f"step-2 alpha=0 baseline (n={len(base_ids)} Steering-split prompts).",
                "Random vectors: unit N(0,I) draws, so alpha*v has the same L2 norm as "
                "alpha*dD / alpha*dR.",
                f"rand_std = sample std (ddof=1) over {len(draws)} draws.",
                "score = |most negative pp| + highest pp over the alphas (the layer-"
                "selection rule). picked_for_this_direction marks the layer the rule chose.",
                "alpha_over_sigma uses the FOUND direction's sigma at that layer.",
                f"Baseline %: N {base['NORMAL']:.1f} / D {base['DISCLAIMER']:.1f} / "
                f"R {base['REFUSAL']:.1f}"]}).to_excel(xw, sheet_name="README", index=False)
            scores.to_excel(xw, sheet_name="scores", index=False)
            comp.to_excel(xw, sheet_name="comparison", index=False)
            sel.to_excel(xw, sheet_name="layer_selection", index=False)
            long.to_excel(xw, sheet_name="pp_long", index=False)
        log(f"  wrote {S2.rel_to_root(out / f'step9_{ctx.slug}.xlsx')}")
    except ImportError as e:
        log(f"  (no xlsx: {e}; the csv files hold the same tables)")

    make_plot(comp, draws, ctx.slug, out, log)
    S2.write_json(out / "step9.json", {
        "slug": ctx.slug, "written_utc": S2.utcnow(), "layers": layers, "picks": picks,
        "draws": draws, "n_prompts": len(base_ids),
        "baseline_pct": {k: round(float(v), 3) for k, v in base.items()},
        "main_predictions_sha256": S2.sha256_file(main_steered(cfg, ctx.slug)
                                                  / "predictions.parquet"),
        "random_predictions_sha256": S2.sha256_file(rp),
    })


def make_plot(comp: pd.DataFrame, draws: list[str], slug: str, out: Path, log) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("  (no matplotlib: plot skipped)")
        return
    if comp.empty:
        return
    panels = comp[comp.picked_for_this_direction][["layer", "found_direction"]].drop_duplicates()
    if panels.empty:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4), squeeze=False)
    for ax, (l, d) in zip(axes[0], panels.itertuples(index=False)):
        c = comp[(comp.layer == l) & (comp.found_direction == d)].sort_values("alpha")
        use_sig = c.alpha_over_sigma.notna().all()
        x = c.alpha_over_sigma if use_sig else c.alpha
        ax.plot(x, c.found_pp, "o-", color="#c0392b", lw=2, label=f"{d} (found)")
        for k in draws:
            ax.plot(x, c[f"{k}_pp"], "-", color="#7f8c8d", lw=0.8, alpha=0.6)
        sd = c.rand_std_pp.fillna(0)
        ax.plot(x, c.rand_mean_pp, "s--", color="#2c3e50", lw=1.5,
                label=f"random mean ({len(draws)} draws)")
        ax.fill_between(x, c.rand_mean_pp - sd, c.rand_mean_pp + sd, color="#2c3e50",
                        alpha=0.15, label="random +- 1 std")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"{slug}  L{l}  {d} -> {TARGET[d]}")
        ax.set_xlabel(f"alpha / sigma_{d}" if use_sig else "alpha")
        ax.set_ylabel(f"{TARGET[d]} pp vs baseline")
        ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"step9_{slug}.{ext}", dpi=150)
    plt.close(fig)
    log(f"  wrote {S2.rel_to_root(out / f'step9_{slug}.png')}")


# --------------------------------------------------------------------------- #
def _selfcheck() -> None:
    assert range_score([-3.9, -4.4, 8.0, 29.7]) == 4.4 + 29.7
    assert range_score([1.0, 2.0, 15.6]) == 15.6
    assert range_score([-1.0, -2.0]) == 2.0
    assert S.parse_condition_tag(S.condition_tag(22, "rand1", -20))["direction"] == "rand1"
    a = draw_seed(42, "x", 22, 0)
    assert a != draw_seed(42, "x", 22, 1) and a != draw_seed(42, "x", 23, 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--stage", choices=("steer", "score", "metrics", "all"), default="all")
    ap.add_argument("--models", required=True, help="ONE slug per run")
    ap.add_argument("--draws", type=int, default=DEFAULT_DRAWS, help="random vectors per layer")
    ap.add_argument("--layers", default=None,
                    help="override the rule's layer(s), comma-separated (e.g. to split "
                         "qwen3-5-9b's two layers across two GPUs)")
    ap.add_argument("--alphas", default=None, help="subset of the non-zero alphas (testing)")
    ap.add_argument("--seed", type=int, default=None, help="default: study.seed")
    ap.add_argument("--batch-size", type=int, default=0,
                    help="default: the step-2 baseline's batch size (must match)")
    ap.add_argument("--limit", type=int, default=0, help="first N prompts (testing only)")
    ap.add_argument("--dry-run", action="store_true", help="steer: list cells + time estimate")
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--allow-plan-mismatch", action="store_true",
                    help="testing only: steer even if the batch plan differs from step 2's")
    ap.add_argument("--token-budget", type=int, default=16384)
    ap.add_argument("--shard-rows", type=int, default=20000)
    ap.add_argument("--skip-val-parity", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="use the config's smoke block")
    args = ap.parse_args()
    if "," in args.models:
        ap.error("--models takes ONE slug per run")
    if args.draws < 1:
        ap.error("--draws must be >= 1")

    _selfcheck()
    root = G.repo_path("results/reports/step9_smoke" if args.smoke else "results/reports/step9")
    log = S2.Tee(root / f"step9_{args.models}_{args.stage}.log")
    try:
        cfg = S2.load_cfg(args.config, args.smoke, log)
        if args.seed is None:
            args.seed = int(cfg["study"]["seed"])
        (ctx,) = S2.model_contexts(cfg, [args.models], log)
        log(f"stage={args.stage} model={ctx.slug} draws={args.draws} seed={args.seed}")
        stages = ("steer", "score", "metrics") if args.stage == "all" else (args.stage,)
        if "steer" in stages:
            stage_steer(cfg, args, ctx, S2.load_study_splits(cfg, log), log)
            if args.dry_run:
                return
        if "score" in stages:
            stage_score(cfg, args, ctx, log)
        if "metrics" in stages:
            stage_metrics(cfg, args, ctx, log)
        log("\ndone")
    finally:
        log.close()


if __name__ == "__main__":
    main()
