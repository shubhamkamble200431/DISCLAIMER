from __future__ import annotations

import argparse
import html
import importlib.util
import math
import sys
import time
from pathlib import Path

import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import generate as G
from src import steer as S

DEFAULT_CONFIG = "configs/study.yaml"
TOP_FRAC = 0.20
PARITY_MIN_COS = 0.99


def import_directions_steering():
    path = ROOT / "scripts" / "directions_steering.py"
    spec = importlib.util.spec_from_file_location("directions_steering", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


DS = import_directions_steering()


def top_k(n: int, frac: float = TOP_FRAC) -> int:
    return max(1, math.ceil(frac * n - 1e-9))


def top_set(scores: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-scores, kind="stable")[:k]


def j20(sD: np.ndarray, sR: np.ndarray, frac: float = TOP_FRAC) -> tuple[float, int, int, int]:
    n = len(sD)
    k = top_k(n, frac)
    tD, tR = set(top_set(sD, k).tolist()), set(top_set(sR, k).tolist())
    inter, union = len(tD & tR), len(tD | tR)
    return inter / union, k, inter, union


def boot_median_ci(x: np.ndarray, B: int, seed: int, ci: float = 0.95):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    meds = np.median(x[idx], axis=1)
    a = (1 - ci) / 2
    return float(np.percentile(meds, 100 * a)), float(np.percentile(meds, 100 * (1 - a)))


def chance_band(n: np.ndarray, k: np.ndarray, R: int, seed: int, ci: float = 0.95) -> dict:
    rng = np.random.default_rng(seed)
    n = np.asarray(n, dtype=np.int64)
    k = np.asarray(k, dtype=np.int64)
    I = rng.hypergeometric(k, n - k, k, size=(R, len(n)))
    J = I / (2 * k - I)
    med = np.median(J, axis=1)
    a = (1 - ci) / 2
    return {"median": float(np.median(med)), "lo": float(np.percentile(med, 100 * a)),
            "hi": float(np.percentile(med, 100 * (1 - a))), "R": int(R),
            "mean_J_per_response": float(J.mean())}


def _selfcheck() -> None:
    sD = np.array([.9, .8, .1, 0, 0, 0, 0, 0, 0, 0])
    sR = np.array([.9, .1, .8, 0, 0, 0, 0, 0, 0, 0])
    J, k, i, u = j20(sD, sR)
    assert (k, i, u) == (2, 1, 3) and abs(J - 1 / 3) < 1e-12
    assert top_k(10) == 2 and top_k(11) == 3 and top_k(1) == 1 and top_k(5) == 1
    rng = np.random.default_rng(0)
    I = rng.hypergeometric(2, 8, 2, size=400_000)
    assert abs((I / (4 - I)).mean() - 19 / 135) < 2e-3
    I = rng.hypergeometric(200, 800, 200, size=200_000)
    assert abs((I / (400 - I)).mean() - 0.2 / 1.8) < 2e-3


def out_dir_for(args, slug: str) -> Path:
    root = "results/reports/token_overlap_smoke" if args.smoke else "results/reports/token_overlap"
    return G.repo_path(root) / slug


def baseline_dir(cfg, slug: str) -> Path:
    return DS.steered_dir(cfg) / slug / S.BASELINE_TAG


def load_baseline(cfg, ctx, splits, ddoc, log) -> pd.DataFrame:
    bdir = baseline_dir(cfg, ctx.slug)
    ans, meta_p = bdir / "answers.jsonl", bdir / "meta.json"
    if not ans.exists():
        DS.fail(f"{ctx.slug}: no {DS.rel_to_root(ans)}; directions_steering.py's alpha=0 baseline is the input")
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    if meta.get("study_split_sha256") and meta["study_split_sha256"] != splits.get("sha256"):
        DS.fail(f"{ctx.slug}: baseline answers were generated on split "
                f"{meta['study_split_sha256'][:16]}, loaded split is {splits['sha256'][:16]}")
    if meta.get("directions_sha256") and meta["directions_sha256"] != ddoc.get("sha256"):
        DS.fail(f"{ctx.slug}: baseline was generated against directions "
                f"{meta['directions_sha256'][:16]} but data/study/directions holds "
                f"{str(ddoc.get('sha256'))[:16]}; the 'frozen' directions changed")
    df = pd.read_json(ans, lines=True, dtype={"prompt_id": "int64"})
    df = df.drop_duplicates("prompt_id").sort_values("prompt_id").reset_index(drop=True)
    want = set(int(p) for p in splits["splits"]["steering"])
    got = set(df.prompt_id.astype(int))
    if got - want:
        DS.fail(f"{ctx.slug}: {len(got - want)} baseline prompts are not in the Steering split")
    if want - got:
        msg = f"{ctx.slug}: baseline covers {len(got)} of {len(want)} Steering-split prompts"
        if not (meta.get("limited_run") or cfg.get("_smoke")):
            DS.fail(msg)
        log(f"  NOTE {msg} (limited/smoke run)")
    log(f"  baseline answers: {len(df)} from {DS.rel_to_root(ans)}")
    return df


def baseline_labels(cfg, slug: str) -> pd.Series | None:
    p = DS.steered_dir(cfg) / slug / "predictions.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p, columns=["prompt_id", "condition_tag", "longformer_label"])
    df = df[df.condition_tag == S.BASELINE_TAG]
    return df.set_index("prompt_id")["longformer_label"]


def stage_forward(cfg, args, ctx, splits, log) -> None:
    import torch

    out = out_dir_for(args, ctx.slug)
    out.mkdir(parents=True, exist_ok=True)
    layers = ([int(x) for x in args.layers.split(",")] if args.layers
              else DS.load_selected_layers(cfg, ctx.slug, study_sha=splits.get("sha256")))
    for l in layers:
        S.assert_steerable(l, ctx.n_blocks)
    arrays, ddoc = DS.load_directions(cfg, ctx.slug, log)
    dirs = {}
    for l in layers:
        for name in ("dD", "dR"):
            v = arrays[f"{name}_L{l:03d}"].astype(np.float64)
            dirs[(name, l)] = v / np.linalg.norm(v)
    base = load_baseline(cfg, ctx, splits, ddoc, log)
    if args.limit:
        base = base.head(int(args.limit))
    table = G.build_prompt_table(cfg).set_index("prompt_id")
    missing = set(base.prompt_id) - set(table.index)
    if missing:
        DS.fail(f"{ctx.slug}: {len(missing)} baseline prompt ids are not in the prompt table")

    log(f"\n=== {ctx.slug}: teacher-forced forward, {len(base)} answers, layers {layers} ===")
    runner = G.Runner(ctx.spec, cfg["generation"], cfg["capture"], log=log)
    if int(runner.n_blocks) != ctx.n_blocks or int(runner.hidden_size) != ctx.hidden_size:
        DS.fail(f"{ctx.slug}: loaded model geometry differs from the generate_label_split manifest")
    tok, model = runner.tok, runner.model
    device = next(model.parameters()).device
    D = torch.tensor(np.stack([dirs[("dD", l)] for l in layers]), dtype=torch.float32,
                     device=device)
    R = torch.tensor(np.stack([dirs[("dR", l)] for l in layers]), dtype=torch.float32,
                     device=device)
    stop_ids = getattr(runner, "_eos_ids", set()) or {tok.eos_token_id}

    tok_rows, score_parts, checks = [], [], []
    t0 = time.perf_counter()
    for i, r in enumerate(base.itertuples(index=False)):
        pid = int(r.prompt_id)
        _, enc = runner.render(table.loc[pid, "prompt"])
        p_ids = [int(v) for v in enc["input_ids"][0].tolist()]
        a_ids = [int(v) for v in tok(str(r.answer), add_special_tokens=False)["input_ids"]]
        expected = int(r.n_generated_tokens) - int(getattr(r, "finish_reason", "") == "eos")
        chk = {"prompt_id": pid, "n_prompt": len(p_ids), "n_answer": len(a_ids),
               "n_expected": expected, "retok_diff": len(a_ids) - expected,
               "special_in_answer": int(sum(t in stop_ids for t in a_ids))}
        if not a_ids:
            checks.append({**chk, "parity_min_cos": None})
            continue
        ids = torch.tensor([p_ids + a_ids], device=device)
        with torch.inference_mode():
            hs = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                       output_hidden_states=True, use_cache=False).hidden_states
            if len(hs) != ctx.n_rows:
                DS.fail(f"{ctx.slug}: forward returned {len(hs)} hidden-state rows, "
                        f"manifest says {ctx.n_rows}")
            P = len(p_ids)
            H = torch.stack([hs[l][0, P:, :] for l in layers]).float()
            last = torch.stack([hs[l][0, P - 1, :] for l in layers]).float()
            nrm = H.norm(dim=-1).clamp_min(1e-12)
            sD = (torch.einsum("lnh,lh->ln", H, D) / nrm).cpu().numpy()
            sR = (torch.einsum("lnh,lh->ln", H, R) / nrm).cpu().numpy()
            last = last.cpu().numpy()
            del hs, H
        chk["parity_min_cos"] = parity_against_capture(ctx, pid, last, layers)
        checks.append(chk)
        n = len(a_ids)
        for j, l in enumerate(layers):
            score_parts.append(pd.DataFrame({
                "prompt_id": np.full(n, pid, dtype=np.int64),
                "layer": np.full(n, l, dtype=np.int16),
                "pos": np.arange(n, dtype=np.int32),
                "S_D": sD[j].astype(np.float32), "S_R": sR[j].astype(np.float32)}))
        tok_rows.append(pd.DataFrame({
            "prompt_id": np.full(n, pid, dtype=np.int64), "pos": np.arange(n, dtype=np.int32),
            "token_id": np.asarray(a_ids, dtype=np.int64),
            "token_text": [tok.decode([t]) for t in a_ids]}))
        if (i + 1) % 50 == 0:
            el = time.perf_counter() - t0
            log(f"    {i + 1}/{len(base)}  {el:.0f}s  "
                f"(~{el / (i + 1) * (len(base) - i - 1):.0f}s left)")
        if torch.cuda.is_available() and (i + 1) % 20 == 0:
            torch.cuda.empty_cache()

    scores = pd.concat(score_parts, ignore_index=True)
    tokens = pd.concat(tok_rows, ignore_index=True)
    ck = pd.DataFrame(checks)
    scores.to_parquet(out / "scores.parquet", index=False)
    tokens.to_parquet(out / "tokens.parquet", index=False)
    ck.to_parquet(out / "forward_checks.parquet", index=False)

    par = ck["parity_min_cos"].dropna()
    diff = ck["retok_diff"]
    rep = {
        "slug": ctx.slug, "hf_id": ctx.spec.hf_id, "written_utc": DS.utcnow(),
        "layers": layers, "n_answers": int(len(base)),
        "n_scored": int(scores.prompt_id.nunique()), "n_empty": int((ck.n_answer == 0).sum()),
        "n_answer_tokens": int(len(tokens)),
        "study_split_sha256": splits.get("sha256"), "directions_sha256": ddoc.get("sha256"),
        "input": DS.rel_to_root(baseline_dir(cfg, ctx.slug) / "answers.jsonl"),
        "retokenisation": {
            "n_exact": int((diff == 0).sum()), "n_differ": int((diff != 0).sum()),
            "abs_diff_max": int(diff.abs().max()) if len(diff) else 0,
            "abs_diff_median_of_differing":
                float(diff[diff != 0].abs().median()) if (diff != 0).any() else 0.0,
            "n_with_special_token": int((ck.special_in_answer > 0).sum())},
        "capture_parity": {"n_checked": int(len(par)),
                           "min_cos": float(par.min()) if len(par) else None,
                           "n_below": int((par < PARITY_MIN_COS).sum()),
                           "threshold": PARITY_MIN_COS},
        "seconds": round(time.perf_counter() - t0, 1),
    }
    DS.write_json(out / "forward.json", rep)
    log(f"  scored {rep['n_scored']} answers, {rep['n_answer_tokens']:,} tokens, "
        f"{rep['n_empty']} empty, {rep['seconds']:.0f}s")
    log(f"  retokenisation: {rep['retokenisation']}")
    log(f"  capture parity vs generate_label_split prompt_last: {rep['capture_parity']}")
    if len(par) and rep["capture_parity"]["n_below"]:
        log(f"  !! WARNING {rep['capture_parity']['n_below']} prompts below cos "
            f"{PARITY_MIN_COS}: the teacher-forced pass does not reproduce generate_label_split.py's "
            f"capture there (template or row mismatch). Investigate before using J20.")


def parity_against_capture(ctx, pid: int, last: np.ndarray, layers: list[int]):
    hid = ctx.root / "hidden"
    target = hid / str(pid) if (hid / str(pid)).is_dir() else hid / f"{pid}.npz"
    if not Path(target).exists():
        return None
    arr, _ = G.load_hidden(target, "prompt_last", None)
    ref = arr[layers].astype(np.float64)
    a = last.astype(np.float64)
    cos = (a * ref).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(ref, axis=1) + 1e-12)
    return float(cos.min())


def per_response_j20(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pid, l), g in scores.groupby(["prompt_id", "layer"], sort=True):
        g = g.sort_values("pos")
        J, k, inter, union = j20(g["S_D"].to_numpy(np.float64), g["S_R"].to_numpy(np.float64))
        rows.append({"prompt_id": int(pid), "layer": int(l), "n_tokens": len(g), "k": k,
                     "intersection": inter, "union": union, "J20": J})
    return pd.DataFrame(rows)


def stage_metrics(cfg, args, ctx, log) -> None:
    out = out_dir_for(args, ctx.slug)
    sp = out / "scores.parquet"
    if not sp.exists():
        DS.fail(f"{ctx.slug}: no {DS.rel_to_root(sp)}; run --stage forward first")
    seed = int(args.seed)
    B = int(args.bootstrap)
    log(f"\n=== {ctx.slug}: J20 metrics, B={B} ===")
    per = per_response_j20(pd.read_parquet(sp))
    lab = baseline_labels(cfg, ctx.slug)
    if lab is not None:
        per["cstar_label"] = per["prompt_id"].map(lab)
    per.to_parquet(out / "j20_per_response.parquet", index=False)

    rows = []
    for l, g in per.groupby("layer", sort=True):
        x = g["J20"].to_numpy(np.float64)
        lo, hi = boot_median_ci(x, B, seed + int(l))
        rows.append({"layer": int(l), "n": int(len(x)), "median_J20": float(np.median(x)),
                     "ci_lo": lo, "ci_hi": hi})
    summ = pd.DataFrame(rows)
    summ.to_parquet(out / "summary.parquet", index=False)
    summ.to_csv(out / "summary.csv", index=False)

    one = per[per.layer == per.layer.min()]
    band = chance_band(one["n_tokens"].to_numpy(), one["k"].to_numpy(), B, seed + 999)
    n_short = int((one["n_tokens"] < 5).sum())

    top = (per.sort_values(["layer", "J20", "n_tokens"], ascending=[True, False, False])
           .groupby("layer").head(int(args.top_n)))
    top.to_csv(out / "top_overlap.csv", index=False)

    DS.write_json(out / "token_overlap.json", {
        "slug": ctx.slug, "written_utc": DS.utcnow(), "seed": seed, "bootstrap_B": B,
        "top_frac": TOP_FRAC, "k_rule": "max(1, ceil(0.2 * n_tokens)), ties -> earlier token",
        "n_responses": int(len(one)), "n_tokens_median": float(one["n_tokens"].median()),
        "n_responses_under_5_tokens": n_short,
        "chance_band": {**band, "definition":
                        "median over responses of J = I/(2k-I), I ~ Hypergeom(n_i, k_i, k_i); "
                        "2.5/97.5 percentiles over R replicates"},
        "summary": summ.to_dict(orient="records"),
    })
    log(f"  {len(one)} responses/layer, median {one['n_tokens'].median():.0f} tokens, "
        f"{n_short} under 5 tokens")
    log(f"  chance (independent rankings): median J20 {band['median']:.3f} "
        f"[{band['lo']:.3f}, {band['hi']:.3f}]")
    log(f"\n  {'layer':>5} {'n':>5} {'median':>8} {'ci_lo':>8} {'ci_hi':>8}")
    for r in summ.itertuples():
        log(f"  {r.layer:>5} {r.n:>5} {r.median_J20:>8.3f} {r.ci_lo:>8.3f} {r.ci_hi:>8.3f}")
    plot_summary(ctx.slug, summ, band, out, log)


def plot_summary(slug: str, summ: pd.DataFrame, band: dict, out: Path, log) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(summ))
    fig, ax = plt.subplots(figsize=(max(6.0, 0.7 * len(summ) + 2), 4.2))
    ax.axhspan(band["lo"], band["hi"], color="0.85", zorder=0,
               label=f"chance (random top-20% sets): {band['median']:.2f}")
    ax.axhline(band["median"], color="0.55", lw=0.9, ls="--", zorder=1)
    med = summ["median_J20"].to_numpy()
    ax.errorbar(x, med, yerr=[med - summ["ci_lo"], summ["ci_hi"] - med], fmt="o",
                color="C0", capsize=4, ms=6, lw=1.4, zorder=3,
                label="median J20, 95% bootstrap CI")
    ax.set_xticks(x, [f"L{l}" for l in summ["layer"]])
    ax.set_ylim(0, 1)
    ax.set_xlabel("layer row (top-10 selected)")
    ax.set_ylabel("J20 = |T_D n T_R| / |T_D u T_R|")
    ax.set_title(f"{slug}: top-20% token overlap, cos(h, d_D) vs cos(h, d_R)\n"
                 f"alpha = 0 answers, n = {int(summ['n'].iloc[0])}", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"token_overlap_{slug}.{ext}", dpi=200)
    plt.close(fig)
    log(f"  plot -> {DS.rel_to_root(out / f'token_overlap_{slug}.png')}")


CSS = """
body{font-family:system-ui,sans-serif;margin:24px;max-width:1100px;color:#222;background:#fff}
h1{font-size:20px} h2{font-size:16px;margin-top:32px;border-top:1px solid #ccc;padding-top:16px}
.meta{color:#555;font-size:13px}
.prompt{background:#f4f4f4;padding:8px 12px;white-space:pre-wrap;font-size:14px}
.ans{line-height:1.9;font-size:14px;white-space:pre-wrap;font-family:ui-monospace,Consolas,monospace}
.t{border-radius:3px;padding:1px 0}
.both{background:#c9a3f0;outline:2px solid #6a1fb8} .d{background:#9cc8f5} .r{background:#f7c08a}
.legend span{display:inline-block;padding:2px 8px;margin-right:8px;border-radius:3px;font-size:13px}
"""


def render_response(pid: int, toks: pd.DataFrame, sc: pd.DataFrame, prompt: str,
                    row: pd.Series) -> str:
    sc = sc.sort_values("pos")
    toks = toks.sort_values("pos")
    sD, sR = sc["S_D"].to_numpy(np.float64), sc["S_R"].to_numpy(np.float64)
    k = top_k(len(sD))
    tD, tR = set(top_set(sD, k).tolist()), set(top_set(sR, k).tolist())
    spans = []
    for p, text in zip(toks["pos"], toks["token_text"]):
        cls = "both" if p in tD and p in tR else "d" if p in tD else "r" if p in tR else ""
        tip = f"pos {p}  S_D={sD[p]:.4f}  S_R={sR[p]:.4f}"
        spans.append(f'<span class="t {cls}" title="{html.escape(tip)}">'
                     f"{html.escape(text)}</span>")
    lab = row.get("cstar_label", None)
    lab_txt = f", C* label {html.escape(lab)}" if isinstance(lab, str) else ""
    return (f"<h2>prompt {pid} &mdash; J20 = {row['J20']:.3f} "
            f"({int(row['intersection'])}/{int(row['union'])}), {int(row['n_tokens'])} "
            f"tokens, k = {k}{lab_txt}</h2>"
            f'<div class="prompt">{html.escape(prompt)}</div>'
            f'<div class="ans">{"".join(spans)}</div>')


def stage_viz(cfg, args, ctx, log) -> None:
    out = out_dir_for(args, ctx.slug)
    per_p = out / "j20_per_response.parquet"
    if not per_p.exists():
        DS.fail(f"{ctx.slug}: no {DS.rel_to_root(per_p)}; run --stage metrics first")
    per = pd.read_parquet(per_p)
    if args.viz_layer is not None:
        layer = int(args.viz_layer)
    else:
        layer = int(per.groupby("layer")["J20"].median().idxmax())
    pool = per[(per.layer == layer) & (per.n_tokens >= int(args.viz_min_tokens))]
    if pool.empty:
        DS.fail(f"{ctx.slug}: no responses at layer {layer} with >= {args.viz_min_tokens} tokens")
    pick = pool.sort_values(["J20", "n_tokens"], ascending=[False, False]).head(int(args.viz_n))
    scores = pd.read_parquet(out / "scores.parquet")
    scores = scores[scores.layer == layer]
    tokens = pd.read_parquet(out / "tokens.parquet")
    table = G.build_prompt_table(cfg).set_index("prompt_id")

    parts = ["<!doctype html><html><head><meta charset='utf-8'>"
             "<meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>Token overlap L{layer}</title><style>{CSS}</style></head><body>",
             f"<h1>{html.escape(ctx.slug)}: top-20% tokens by cos(h, d_D) vs cos(h, d_R), "
             f"layer {layer}</h1>",
             f"<p class='meta'>Unsteered (alpha = 0) answers, Steering split. The "
             f"{len(pick)} highest-J20 responses with at least {args.viz_min_tokens} answer "
             f"tokens. Hover a token for its scores.</p>",
             "<p class='legend'><span class='both'>in T_D and T_R</span>"
             "<span class='d'>T_D only</span><span class='r'>T_R only</span></p>"]
    for _, row in pick.iterrows():
        pid = int(row["prompt_id"])
        parts.append(render_response(pid, tokens[tokens.prompt_id == pid],
                                     scores[scores.prompt_id == pid],
                                     str(table.loc[pid, "prompt"]), row))
    parts.append("</body></html>")
    dest = out / f"token_overlap_L{layer:03d}.html"
    dest.write_text("".join(parts), encoding="utf-8", newline="\n")
    log(f"  viz layer {layer}: {len(pick)} responses (max J20 {pick['J20'].iloc[0]:.3f}, "
        f"prompt {int(pick['prompt_id'].iloc[0])}) -> {DS.rel_to_root(dest)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="token_overlap -- token analysis: do d_D and d_R light up the SAME answer tokens?")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--stage", choices=("forward", "metrics", "viz", "all"), default="all")
    ap.add_argument("--models", required=True,
                    help="ONE slug per run (e.g. olmo3-7b), so models can run side by side")
    ap.add_argument("--layers", default=None, help="default: directions_steering.py's top-10")
    ap.add_argument("--limit", type=int, default=0, help="first N answers only (testing)")
    ap.add_argument("--bootstrap", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=None, help="default: study.seed")
    ap.add_argument("--top-n", type=int, default=20, help="rows per layer in top_overlap.csv")
    ap.add_argument("--viz-layer", type=int, default=None,
                    help="default: the layer with the highest median J20")
    ap.add_argument("--viz-n", type=int, default=5, help="responses rendered in the HTML")
    ap.add_argument("--viz-min-tokens", type=int, default=20,
                    help="skip very short answers when picking the max-J20 example")
    ap.add_argument("--smoke", action="store_true", help="use the config's smoke block")
    args = ap.parse_args()
    if "," in args.models:
        ap.error("--models takes ONE slug per run")

    _selfcheck()
    root = G.repo_path("results/reports/token_overlap_smoke" if args.smoke else "results/reports/token_overlap")
    log = DS.Tee(root / f"token_overlap_{args.models}_{args.stage}.log")
    try:
        cfg = DS.load_cfg(args.config, args.smoke, log)
        if args.seed is None:
            args.seed = int(cfg["study"]["seed"])
        (ctx,) = DS.model_contexts(cfg, [args.models], log)
        stages = ("forward", "metrics", "viz") if args.stage == "all" else (args.stage,)
        if "forward" in stages:
            splits = DS.load_study_splits(cfg, log)
            stage_forward(cfg, args, ctx, splits, log)
        if "metrics" in stages:
            stage_metrics(cfg, args, ctx, log)
        if "viz" in stages:
            stage_viz(cfg, args, ctx, log)
        log("\ndone")
    finally:
        log.close()


if __name__ == "__main__":
    main()
