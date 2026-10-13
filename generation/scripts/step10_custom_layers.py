"""Step 10 -- steer d_R / d_D at hand-picked layers (user's choice, 2026-10-05).

Same experiment as step 2's sweep, at layers OUTSIDE (or partly inside) each model's step-2
top-10. Each direction is steered ONLY at its own layers -- nothing else:

    model        dR = unit(mu_R - mu_N)    dD = unit(mu_D - mu_N)
    qwen3-5-9b   L25, L28                  L21, L25
    qwen3-5-4b   L29, L30                  L16
    olmo3-7b     (none)                    L19

No cross-steering: no dD at 9B L28, no dR at 4B L16, no dR in OLMo. The vectors are step 2's
own directions (data/study/directions/<slug>.npz, saved for every steerable layer), so
nothing is re-estimated. Layer numbers are step 2's row index (the L### of its condition
tags; row l = output of decoder block l-1).

STEP-2 PARITY. Same 700 Steering-split prompts, the config's non-zero alphas, same hook
(h' = h + alpha * d_hat at the last prefill token only), same generation settings, same
frozen C* Longformer. The alpha=0 baseline is NOT regenerated: step 2's baseline_ap0 is
reused and the run refuses to start unless its batch plan hash equals that baseline's
(batch size read from the baseline's meta.json).

REUSE. A (layer, direction, alpha) cell that step 2 already produced -- complete, under the
baseline's batch plan -- is not regenerated (qwen3-5-9b L25 is in its step-2 top-10). The
metrics stage takes those rows from step 2's predictions.parquet and marks them
source=step2.

STAGES (one model per run, so models can run side by side on separate GPUs)
  steer    GPU (base)        -> data/steered_custom/<slug>/<tag>/answers.jsonl + meta.json
                             tags are step 2's grammar, e.g. L028_dR_ap20
  score    GPU (refusal_clf) frozen C* via step 2's own scoring code (val-parity gate
                             included) -> data/steered_custom/<slug>/predictions.parquet
  metrics  CPU               step 2's metrics only -> results/reports/step10/<slug>/
                             pp_table.parquet/.csv          class-wise pp vs baseline +
                                                            paired bootstrap CI (step 2's
                                                            estimator and index seed)
                             transitions_table.parquet/.csv  3x3 before->after, step-2 format
                             transitions_per_prompt.parquet
                             pp_<slug>_<dir>_<subset>.png, transitions_bar_*.png
                             step10.json                    inputs, cell sources, shas
  all                        steer + score + metrics (laptop smoke only)

Usage:
  python -X utf8 scripts/step10_custom_layers.py --models olmo3-7b --stage steer --dry-run
  python -X utf8 scripts/step10_custom_layers.py --models olmo3-7b --stage steer
  python -X utf8 scripts/step10_custom_layers.py --models qwen3-5-9b --stage steer --layers 28
  python -X utf8 scripts/step10_custom_layers.py --models olmo3-7b --stage score
  python -X utf8 scripts/step10_custom_layers.py --models olmo3-7b --stage metrics
  python -X utf8 scripts/step10_custom_layers.py --smoke --models smoke-olmo2-1b --stage all
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
LABEL_COL = "longformer_label"
FOUND = ("dD", "dR")
# THE single source of what is steered (user, 2026-10-05). Direction -> its layers only.
LAYER_MAP = {
    "qwen3-5-9b": {"dR": [25, 28], "dD": [21, 25]},
    "qwen3-5-4b": {"dR": [29, 30], "dD": [16]},
    "olmo3-7b":   {"dR": [],       "dD": [19]},
}
# --smoke only. L15 dR is in the step-2 smoke sweep, so the smoke exercises the reuse path
# as well as fresh generation.
SMOKE_LAYER_MAP = {
    "smoke-olmo2-1b": {"dR": [13, 15], "dD": [12]},
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
# paths / the cell list
# --------------------------------------------------------------------------- #
def custom_cfg(cfg) -> dict:
    """Copy of cfg whose steered_root is the step-10 tree, so step 2's own generation and
    scoring code writes there and never touches the main sweep."""
    cc = json.loads(json.dumps(cfg))
    cc["output"]["steered_root"] = str(cfg["output"]["steered_root"]).rstrip("/") + "_custom"
    return cc


def report_dir(cfg, slug: str) -> Path:
    base = "results/reports/step10_smoke" if cfg.get("_smoke") else "results/reports/step10"
    return G.repo_path(base) / slug


def layer_map(cfg, slug: str) -> dict:
    m = (SMOKE_LAYER_MAP if cfg.get("_smoke") else LAYER_MAP).get(slug)
    if m is None:
        fail(f"{slug} is not in the step-10 layer map; models: "
             f"{sorted(SMOKE_LAYER_MAP if cfg.get('_smoke') else LAYER_MAP)}")
    return m


def planned_cells(cfg, args, ctx) -> list[tuple[str, int, str, float]]:
    """(tag, layer, direction, alpha) for this model, direction-major, from the map only.
    --layers narrows the map; a layer the map does not have is an error, not an addition."""
    m = layer_map(cfg, ctx.slug)
    all_layers = sorted({l for d in FOUND for l in m.get(d, [])})
    keep = all_layers
    if args.layers:
        want = sorted({int(x) for x in args.layers.split(",") if x.strip()})
        bad = [l for l in want if l not in all_layers]
        if bad:
            fail(f"--layers {bad} is not in {ctx.slug}'s step-10 map {m}")
        keep = want
    alphas = S2.nonzero_alphas(cfg)
    if args.alphas:
        want_a = {float(x) for x in args.alphas.split(",") if x.strip()}
        alphas = [a for a in alphas if a in want_a]
    out = []
    for d in FOUND:
        for l in m.get(d, []):
            if l not in keep:
                continue
            S.assert_steerable(l, ctx.n_blocks)
            for a in alphas:
                out.append((S.condition_tag(l, d, a), l, d, float(a)))
    if not out:
        fail(f"{ctx.slug}: no cells to run (map {m}, --layers {args.layers})")
    return out


def steering_rows(cfg, splits, args):
    """Step 2's Steering-split rows, built exactly as its stage_steer builds them."""
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


def cell_complete(root_cfg, slug: str, tag: str, want: set[int], plan_hash) -> bool:
    """answers.jsonl covers every prompt and meta.json carries the expected batch plan."""
    cdir = S2.condition_dir(root_cfg, slug, tag)
    ans, meta = cdir / "answers.jsonl", cdir / "meta.json"
    if not (ans.exists() and meta.exists()):
        return False
    if not S.read_done_ids(ans) >= want:
        return False
    doc = json.loads(meta.read_text(encoding="utf-8"))
    return doc.get("batch_plan_hash") == plan_hash


# --------------------------------------------------------------------------- #
# stage: steer
# --------------------------------------------------------------------------- #
def stage_steer(cfg, args, ctx, splits, log) -> None:
    ccfg = custom_cfg(cfg)
    conds = planned_cells(cfg, args, ctx)
    bmeta_path = S2.condition_dir(cfg, ctx.slug, S.BASELINE_TAG) / "meta.json"
    if not bmeta_path.exists():
        fail(f"{ctx.slug}: no {S2.rel_to_root(bmeta_path)}; the step-2 baseline is the "
             f"subtrahend and must exist")
    bmeta = json.loads(bmeta_path.read_text(encoding="utf-8"))
    base_hash = bmeta.get("batch_plan_hash")
    batch_size = int(args.batch_size or bmeta["batch_size_requested"])
    rows = steering_rows(cfg, splits, args)
    want = {int(r.prompt_id) for r in rows}

    # Status of every planned cell, BEFORE the model is loaded, so --dry-run shows it too.
    status = {}
    for tag, *_ in conds:
        if cell_complete(cfg, ctx.slug, tag, want, base_hash):
            status[tag] = "from step 2"
        elif cell_complete(ccfg, ctx.slug, tag, want, base_hash):
            status[tag] = "done (step 10)"
        else:
            status[tag] = "TO RUN"
    todo = [c for c in conds if status[c[0]] == "TO RUN"]

    m = layer_map(cfg, ctx.slug)
    log(f"\n=== {ctx.slug}: step 10, hand-picked layers ===")
    log(f"  map: dR -> {m.get('dR') or 'none'}   dD -> {m.get('dD') or 'none'}"
        + (f"   (--layers {args.layers})" if args.layers else ""))
    log(f"  {len(conds)} cell(s): {sum(v == 'from step 2' for v in status.values())} from "
        f"step 2, {sum(v == 'done (step 10)' for v in status.values())} already done here, "
        f"{len(todo)} to run x {len(rows)} prompts = {len(todo) * len(rows):,} generations; "
        f"batch {batch_size} (step-2 baseline used {bmeta['batch_size_requested']})")
    for tag, *_ in conds:
        log(f"    {tag:<16} {status[tag]}")

    if args.dry_run:
        sec = [json.loads(p.read_text(encoding="utf-8")).get("seconds", 0)
               for p in (S2.steered_dir(cfg) / ctx.slug).glob("L*/meta.json")]
        sec = [s for s in sec if s]
        if sec and todo:
            per_cell = float(np.median(sec))
            log(f"  DRY RUN: step-2 cells of this model took a median {per_cell / 60:.1f} "
                f"min each -> about {per_cell * len(todo) / 3600:.1f} GPU-h for this run")
        return
    if not todo:
        log("  nothing to generate")
        return

    arrays, ddoc = S2.load_directions(cfg, ctx.slug, log)
    for _, l, d, _ in todo:
        if f"{d}_L{l:03d}" not in arrays:
            fail(f"{ctx.slug}: {d}_L{l:03d} is not in the directions npz")
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
        if plan_hash != base_hash:
            msg = (f"{ctx.slug}: batch plan {plan_hash[:16]} != step-2 baseline's "
                   f"{str(base_hash)[:16]}. Batched greedy decoding is not invariant to "
                   f"batch composition, so the pp vs that baseline would be invalid. Use "
                   f"the same --batch-size and prompt set as step 2 (baseline meta says "
                   f"{bmeta['batch_size_requested']}).")
            if not args.allow_plan_mismatch:
                fail(msg)
            log(f"  !! --allow-plan-mismatch (TESTING ONLY): {msg}")
        else:
            log(f"  batch plan {plan_hash[:16]} == step-2 baseline's ({len(batches)} batches)")
        S2.assert_plan_consistent(ccfg, ctx, plan_hash, log)

        done, t_start = 0, time.perf_counter()
        for ci, (tag, layer, dname, alpha) in enumerate(todo, 1):
            cdir = S2.condition_dir(ccfg, ctx.slug, tag)
            ans = cdir / "answers.jsonl"
            if ans.exists():
                log(f"  [{ci}/{len(todo)}] {tag}: partial; redoing the whole condition "
                    f"(static batch membership)")
                ans.unlink()
            vec = arrays[f"{dname}_L{layer:03d}"]
            sigma = S2.sigma_for(ddoc, layer, dname)
            hook_ctx = (lambda m=runner.model, l=layer, v=vec, a=alpha:
                        S.steering_hook(m, l, v, a, tok=runner.tok, n_blocks=ctx.n_blocks))
            log(f"  [{ci}/{len(todo)}] {tag}  layer={layer} dir={dname} alpha={alpha}"
                + (f"  alpha/sigma={alpha / sigma:.3f}" if sigma else ""))
            t0 = time.perf_counter()
            records = gen.generate(rows, batches, hook_ctx=hook_ctx,
                                   extra={"condition": tag, "layer": layer,
                                          "direction": dname, "alpha": float(alpha)})
            cdir.mkdir(parents=True, exist_ok=True)
            S.write_answers_jsonl(ans, records, mode="w")
            S2.write_json(cdir / "meta.json", {
                "condition": tag, "layer": layer, "direction": dname, "alpha": float(alpha),
                "sigma_l": sigma,
                "alpha_over_sigma": (float(alpha) / sigma) if sigma else None,
                "x_axis": cfg["steering"].get("x_axis", "alpha_over_sigma"),
                "step": "step10 hand-picked layers",
                "seed": int(args.seed), "n_rows": len(rows), "n_records": len(records),
                "batch_plan_hash": plan_hash,
                "baseline_batch_plan_hash": base_hash,
                "batch_size_requested": gen.requested_batch_size,
                "batch_size_effective": gen.batch_size,
                "n_batches": len(batches), "n_padded_repeats": gen.n_padded_repeats,
                "n_oom_halvings": gen.n_oom_halvings,
                "limited_run": bool(args.limit or cfg.get("_smoke")),
                "generation": dict(runner.gen_kwargs),
                "model": {"slug": ctx.slug, "hf_id": ctx.spec.hf_id,
                          "dtype": ctx.spec.dtype,
                          "attn_implementation": ctx.spec.attn_implementation,
                          "n_blocks": ctx.n_blocks, "hidden_size": ctx.hidden_size,
                          "system_prompt": ctx.spec.system_prompt,
                          "chat_template_kwargs": ctx.spec.chat_template_kwargs},
                "directions_sha256": ddoc.get("sha256"),
                "study_split_sha256": splits.get("sha256"),
                "preflight": {"skipped": bool(pre.get("skipped")),
                              "worst_layer": pre.get("worst_layer")},
                "answers_sha256": S2.sha256_file(ans),
                "seconds": round(time.perf_counter() - t0, 1),
                "written_utc": S2.utcnow(),
            })
            done += len(records)
            el = time.perf_counter() - t_start
            rate = done / max(el, 1e-9)
            log(f"      {len(records)} answers in {time.perf_counter() - t0:.0f}s | "
                f"{el / 3600:.2f} h so far | "
                f"~{(len(todo) - ci) * len(rows) / max(rate, 1e-9) / 3600:.1f} h left")
    finally:
        del runner
        S2._empty_cuda_cache()


# --------------------------------------------------------------------------- #
# stage: score
# --------------------------------------------------------------------------- #
def stage_score(cfg, args, ctx, log) -> None:
    sargs = SimpleNamespace(shard_rows=args.shard_rows, token_budget=args.token_budget,
                            skip_val_parity=args.skip_val_parity)
    S2.stage_score(custom_cfg(cfg), sargs, [ctx], log)


# --------------------------------------------------------------------------- #
# stage: metrics (step 2's estimators, nothing new)
# --------------------------------------------------------------------------- #
def gather_predictions(cfg, args, ctx, log) -> tuple[pd.DataFrame, pd.DataFrame, dict, list]:
    """Baseline rows (step 2) + one block of rows per planned cell, taken from the step-10
    tree when it has the cell, else from step 2's predictions."""
    main_p = S2.predictions_path(cfg, ctx.slug)
    if not main_p.exists():
        fail(f"{ctx.slug}: no {S2.rel_to_root(main_p)}; step 2 score must be complete (the "
             f"baseline is the subtrahend)")
    cols = ["prompt_id", "condition_tag", "layer", "direction", "alpha", LABEL_COL]
    main = pd.read_parquet(main_p, columns=cols)
    base = main[main.condition_tag == S.BASELINE_TAG]
    if base.empty:
        fail(f"{ctx.slug}: {S2.rel_to_root(main_p)} has no {S.BASELINE_TAG} rows")
    cust_p = S2.predictions_path(custom_cfg(cfg), ctx.slug)
    cust = (pd.read_parquet(cust_p, columns=cols) if cust_p.exists()
            else pd.DataFrame(columns=cols))
    have_c, have_m = set(cust.condition_tag), set(main.condition_tag)

    pieces, source, missing = [], {}, []
    conds = planned_cells(cfg, args, ctx)
    for tag, *_ in conds:
        if tag in have_c:
            pieces.append(cust[cust.condition_tag == tag])
            source[tag] = "step10"
        elif tag in have_m:
            pieces.append(main[main.condition_tag == tag])
            source[tag] = "step2"
        else:
            missing.append(tag)
    if missing:
        fail(f"{ctx.slug}: {len(missing)} planned cell(s) are scored nowhere: {missing[:6]}"
             f"{' ...' if len(missing) > 6 else ''}; run --stage steer and --stage score")
    pred = pd.concat(pieces, ignore_index=True)
    pred["prompt_id"] = pred["prompt_id"].astype("int64")
    base = base.assign(prompt_id=base.prompt_id.astype("int64"))
    log(f"  {len(conds)} cell(s): {sum(v == 'step10' for v in source.values())} from step 10, "
        f"{sum(v == 'step2' for v in source.values())} from step 2; baseline n={len(base)}")
    inputs = [main_p] + ([cust_p] if cust_p.exists() else [])
    return pred, base, source, inputs


def stage_metrics(cfg, args, ctx, log) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = report_dir(cfg, ctx.slug)
    out.mkdir(parents=True, exist_ok=True)
    bcfg = cfg["study"]["bootstrap"]
    B = int(args.bootstrap or bcfg.get("B", 10000))
    ci = float(bcfg.get("ci", 0.95))
    labels3 = list(cfg["study"]["labels3"])
    min_n = int(cfg["study"].get("min_n", 30))

    labels = S2.load_labels(cfg, None, log)
    cstar_split = labels.drop_duplicates("prompt_id").set_index("prompt_id")["cstar_split"]
    clean_ids = set(int(p) for p in cstar_split[cstar_split == "test"].index)

    pred, base, source, inputs = gather_predictions(cfg, args, ctx, log)
    base = base.drop_duplicates("prompt_id").set_index("prompt_id")
    _, ddoc = S2.load_directions(cfg, ctx.slug, log)
    conds = sorted(pred.condition_tag.unique())

    pp_rows, tt_rows, tr_rows = [], [], []
    for subset, ids in (("full", sorted(base.index)),
                        ("cstar_test_clean", sorted(set(base.index) & clean_ids))):
        ids = np.asarray(ids, dtype=np.int64)
        n = len(ids)
        if n < min_n:
            log(f"  subset {subset}: n={n} < min_n={min_n}; reported as counts only")
        if n == 0:
            continue
        b_lab = base.loc[ids][LABEL_COL].to_numpy()
        b_onehot = {c: (b_lab == c).astype(np.float64) for c in labels3}
        base_pct = {c: float(100 * b_onehot[c].mean()) for c in labels3}
        # Step 2's index seed, so the CIs here are drawn from the same resamples.
        idx = S2.make_boot_index(n, B, int(args.seed) + 7919 * (subset == "full"))
        log(f"  subset {subset}: n={n}  baseline "
            + " ".join(f"{c}={base_pct[c]:.1f}%" for c in labels3))
        for tag in conds:
            sub = pred[pred.condition_tag == tag].drop_duplicates("prompt_id")
            sub = sub.set_index("prompt_id")
            miss = [int(p) for p in ids if p not in sub.index]
            if miss:
                fail(f"{tag}: {len(miss)} baseline prompt(s) unscored in this cell")
            s_lab = sub.loc[ids][LABEL_COL].to_numpy()
            meta = S.parse_condition_tag(tag)
            layer, dname, alpha = meta["layer"], meta["direction"], meta["alpha"]
            sigma = S2.sigma_for(ddoc, layer, dname)
            aos = (float(alpha) / sigma) if sigma else None
            total = 0.0
            for c in labels3:
                s_one = (s_lab == c).astype(np.float64)
                d = s_one - b_onehot[c]                    # paired per-prompt difference
                pp = float(100 * d.mean())
                lo, hi = S2.boot_ci(100 * d[idx].mean(axis=1), ci)
                total += pp
                pp_rows.append({"model": ctx.slug, "subset": subset, "condition_tag": tag,
                                "layer": layer, "direction": dname, "alpha": float(alpha),
                                "sigma_l": sigma, "alpha_over_sigma": aos, "class": c,
                                "n": int(n), "baseline_pct": base_pct[c],
                                "steered_pct": float(100 * s_one.mean()),
                                "pp": pp, "ci_lo": lo, "ci_hi": hi,
                                "source": source[tag]})
            if abs(total) > 1e-6:
                fail(f"{tag} {subset}: class-wise pp sums to {total}, must be 0")
            aos_f = float("nan") if aos is None else float(aos)
            for i_c in labels3:
                m = b_onehot[i_c].astype(bool)
                denom = int(m.sum())
                for j in labels3:
                    cnt = int(((s_lab == j) & m).sum())
                    tt_rows.append({"model": ctx.slug, "subset": subset,
                                    "condition_tag": tag, "layer": int(layer),
                                    "direction": str(dname), "alpha": float(alpha),
                                    "alpha_over_sigma": aos_f, "from_class": i_c,
                                    "to_class": j, "count": cnt, "denom": denom,
                                    "rate_pct": (100.0 * cnt / denom) if denom
                                                else float("nan"),
                                    "below_min_n": bool(denom < min_n)})
            if subset == "full":
                csp = cstar_split.reindex(ids).fillna("unknown").astype(str).to_numpy()
                for p, b_l, s_l, cs in zip(ids, b_lab, s_lab, csp):
                    tr_rows.append({"prompt_id": int(p), "model": ctx.slug,
                                    "layer": int(layer), "direction": str(dname),
                                    "alpha": float(alpha), "alpha_over_sigma": aos_f,
                                    "before": str(b_l), "after": str(s_l),
                                    "changed": bool(b_l != s_l), "cstar_split": str(cs)})

    df = pd.DataFrame(pp_rows)
    df.to_parquet(out / "pp_table.parquet", index=False)
    df.to_csv(out / "pp_table.csv", index=False)
    tt = pd.DataFrame(tt_rows, columns=S2.TRANSITION_TABLE_COLUMNS)
    tt = tt.astype({"layer": "int64", "alpha": "float64", "alpha_over_sigma": "float64",
                    "count": "int64", "denom": "int64", "rate_pct": "float64",
                    "below_min_n": "bool"})
    tt.to_parquet(out / "transitions_table.parquet", index=False)
    tt.to_csv(out / "transitions_table.csv", index=False)
    tr = pd.DataFrame(tr_rows, columns=S2.TRANSITION_PP_COLUMNS)
    tr.to_parquet(out / "transitions_per_prompt.parquet", index=False)

    # Step 2's own figures. An empty report skips its best-layer heatmap, which ranks
    # layers -- not wanted for hand-picked layers.
    figs = S2.make_plots(cfg, {"models": {}}, df, out, plt, log)
    bars = S2.make_transition_bars(cfg, tt, out, plt, log)

    log(f"\n  target-class pp vs baseline (full subset), by alpha:")
    full = df[df.subset == "full"]
    for d in FOUND:
        t = S2.target_class(d)
        g = full[(full.direction == d) & (full["class"] == t)]
        for l, gl in g.groupby("layer"):
            gl = gl.sort_values("alpha")
            log(f"    L{int(l)} {d}->{t} [{gl.source.iloc[0]}]: " + "  ".join(
                f"{a:+g}:{p:+.1f}" for a, p in zip(gl.alpha, gl.pp)))

    S2.write_json(out / "step10.json", {
        "slug": ctx.slug, "written_utc": S2.utcnow(),
        "layer_map": layer_map(cfg, ctx.slug), "layers_arg": args.layers,
        "cell_source": source, "n_cells": len(conds), "n_prompts_full": int(len(base)),
        "bootstrap": {"B": B, "ci": ci, "paired_over": "prompt_id",
                      "index_seed": "study.seed + 7919 * (subset == 'full'), as step 2"},
        "inputs": S2.file_inventory(inputs),
        "figures": figs + bars,
    })
    log(f"  wrote {S2.rel_to_root(out)}/ (pp_table, transitions_table, "
        f"transitions_per_prompt, {len(figs) + len(bars)} figure(s), step10.json)")


# --------------------------------------------------------------------------- #
def _selfcheck() -> None:
    n = {s: sum(len(v) for v in m.values()) * 10 for s, m in LAYER_MAP.items()}
    assert n == {"qwen3-5-9b": 40, "qwen3-5-4b": 30, "olmo3-7b": 10}, n
    for m in list(LAYER_MAP.values()) + list(SMOKE_LAYER_MAP.values()):
        assert set(m) <= set(FOUND)
        for v in m.values():
            assert len(v) == len(set(v))
    assert S.parse_condition_tag(S.condition_tag(28, "dR", -20))["layer"] == 28


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--stage", choices=("steer", "score", "metrics", "all"), default="all")
    ap.add_argument("--models", required=True, help="ONE slug per run")
    ap.add_argument("--layers", default=None,
                    help="run only these of the model's map layers, comma-separated (to "
                         "split one model across GPUs)")
    ap.add_argument("--alphas", default=None, help="subset of the non-zero alphas (testing)")
    ap.add_argument("--seed", type=int, default=None, help="default: study.seed")
    ap.add_argument("--batch-size", type=int, default=0,
                    help="default: the step-2 baseline's batch size (must match)")
    ap.add_argument("--limit", type=int, default=0, help="first N prompts (testing only)")
    ap.add_argument("--dry-run", action="store_true", help="steer: list cells + time estimate")
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--allow-plan-mismatch", action="store_true",
                    help="testing only: steer even if the batch plan differs from step 2's")
    ap.add_argument("--bootstrap", type=int, default=0, help="B (default: study.bootstrap.B)")
    ap.add_argument("--token-budget", type=int, default=16384)
    ap.add_argument("--shard-rows", type=int, default=20000)
    ap.add_argument("--skip-val-parity", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="use the config's smoke block")
    args = ap.parse_args()
    if "," in args.models:
        ap.error("--models takes ONE slug per run")

    _selfcheck()
    root = G.repo_path("results/reports/step10_smoke" if args.smoke else "results/reports/step10")
    log = S2.Tee(root / f"step10_{args.models}_{args.stage}.log")
    try:
        cfg = S2.load_cfg(args.config, args.smoke, log)
        if args.seed is None:
            args.seed = int(cfg["study"]["seed"])
        (ctx,) = S2.model_contexts(cfg, [args.models], log)
        log(f"stage={args.stage} model={ctx.slug} seed={args.seed}")
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
