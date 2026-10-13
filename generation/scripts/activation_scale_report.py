"""Is a single ABSOLUTE alpha grid viable across layers and across the three models?

Reads the saved `prompt_last` activations and reports, per layer per model:

  ||h||            median / p10 / p90 residual-stream norm
  d_R, d_D         difference-in-means directions from that model's own C* labels
  sigma_R          sd of the data projected onto d_R  (the units a perturbation is
                   actually felt in -- adding alpha along d_R moves the distribution
                   by alpha / sigma_R standard deviations)
  alpha as % ||h|| what alpha in {1,2,5,10} does to the state, as a fraction of its norm
  alpha in sigma   the same alphas expressed in sigma_R units

Verdict rule (printed at the end): an absolute grid is viable only if some single alpha
is simultaneously >= MIN_EFFECT sigma (it does something) and <= MAX_SAFE fraction of
||h|| (it does not obliterate the state) at the SAME layer in ALL THREE models.

Usage:
  python -X utf8 scripts/activation_scale_report.py
  python -X utf8 scripts/activation_scale_report.py --n-prompts 800 --models qwen3-5-4b
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GEN_ROOT = ROOT / "data" / "generated"
PRED = ROOT.parent / "refusal_classifier" / "results" / "predictions" / "generated_predictions.parquet"
OUT = ROOT / "data" / "generated" / "activation_scale_report.json"

MODELS = ["qwen3-5-4b", "gemma-4-e2b", "qwen3-5-9b"]
ALPHAS = [1, 2, 5, 10]

# viability thresholds for the printed verdict
MIN_EFFECT_SIGMA = 0.25      # below this, a perturbation is lost in the noise
MAX_SAFE_FRAC = 0.50         # above this, alpha*d dominates the residual stream


def load_prompt_stack(model_root: Path, prompt_ids, n_rows, log):
    """[n_prompts, n_layer_rows, H] float32, reading each prompt's folder once."""
    hid = model_root / "hidden"
    mats, kept = [], []
    t0 = time.perf_counter()
    for i, pid in enumerate(prompt_ids):
        d = hid / str(pid)
        fs = sorted(d.glob("prompt_last_layer_*.npy"))
        if len(fs) != n_rows:
            continue
        mats.append(np.stack([np.load(f).astype(np.float32).ravel() for f in fs]))
        kept.append(pid)
        if (i + 1) % 200 == 0:
            log(f"      {i + 1}/{len(prompt_ids)} prompts  {time.perf_counter() - t0:.0f}s")
    return np.stack(mats), kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prompts", type=int, default=600)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    log = print

    pred = pd.read_parquet(PRED)
    report = {"n_prompts_requested": args.n_prompts, "alphas": ALPHAS, "models": {}}

    for slug in [s.strip() for s in args.models.split(",") if s.strip()]:
        root = GEN_ROOT / slug
        man = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        n_rows = man["model"]["n_layer_rows"]
        n_blocks = man["model"]["n_blocks"]
        H = man["model"]["hidden_size"]
        log(f"\n=== {slug}  H={H}  blocks={n_blocks}  rows={n_rows} ===")

        sub = pred[pred.gen_model == slug]
        # stratify the sample so mu_D and mu_R are estimable
        rng = np.random.default_rng(args.seed)
        per = max(1, args.n_prompts // 3)
        pick = []
        for lab in ["NORMAL", "DISCLAIMER", "REFUSAL"]:
            ids = sub[sub.longformer_label == lab].prompt_id.to_numpy()
            pick.append(rng.choice(ids, size=min(per, len(ids)), replace=False))
        pids = np.concatenate(pick)

        A, kept = load_prompt_stack(root, pids, n_rows, log)
        lab = sub.set_index("prompt_id").loc[kept, "longformer_label"].to_numpy()
        nN, nD, nR = (lab == "NORMAL").sum(), (lab == "DISCLAIMER").sum(), (lab == "REFUSAL").sum()
        log(f"    loaded {A.shape}  labels N={nN} D={nD} R={nR}")

        muN = A[lab == "NORMAL"].mean(0)
        muD = A[lab == "DISCLAIMER"].mean(0)
        muR = A[lab == "REFUSAL"].mean(0)

        rows = []
        for l in range(n_rows):
            norms = np.linalg.norm(A[:, l, :], axis=1)
            med = float(np.median(norms))

            dR_raw, dD_raw = muR[l] - muN[l], muD[l] - muN[l]
            nR_, nD_ = float(np.linalg.norm(dR_raw)), float(np.linalg.norm(dD_raw))
            if nR_ > 1e-8:
                dR = dR_raw / nR_
                sigR = float(np.std(A[:, l, :] @ dR))
                cos_DR = (float(np.dot(dR, dD_raw / nD_)) if nD_ > 1e-8 else None)
            else:
                dR, sigR, cos_DR = None, 0.0, None

            rows.append({
                "layer_row": l,
                "steerable": 1 <= l <= n_blocks - 1,
                "median_norm": round(med, 3),
                "p10_norm": round(float(np.percentile(norms, 10)), 3),
                "p90_norm": round(float(np.percentile(norms, 90)), 3),
                "sep_R": round(nR_, 4),
                "sep_D": round(nD_, 4),
                "sigma_R": round(sigR, 4),
                "cos_dD_dR": (round(cos_DR, 4) if cos_DR is not None else None),
                "alpha_pct_of_norm": {a: round(100 * a / med, 2) for a in ALPHAS} if med else None,
                "alpha_in_sigma": ({a: round(a / sigR, 3) for a in ALPHAS}
                                   if sigR > 1e-8 else None),
            })

        report["models"][slug] = {
            "hidden_size": H, "n_blocks": n_blocks, "n_layer_rows": n_rows,
            "n_prompts_used": int(len(kept)),
            "label_counts": {"NORMAL": int(nN), "DISCLAIMER": int(nD), "REFUSAL": int(nR)},
            "layers": rows,
        }

        # ---- printed table, middle layers first ---------------------------- #
        mid_lo, mid_hi = n_blocks // 3, (2 * n_blocks) // 3
        def show(sel, title):
            log(f"\n  {title}")
            log(f"  {'row':>4} {'||h||':>9} {'sigma_R':>8} {'sep_R':>7} {'cos(dD,dR)':>10} | "
                + " ".join(f"a={a:<2d}%||h||" for a in ALPHAS))
            for r in rows:
                if not sel(r["layer_row"]):
                    continue
                pn = r["alpha_pct_of_norm"] or {}
                sg = r["alpha_in_sigma"] or {}
                cs = f"{r['cos_dD_dR']:.3f}" if r["cos_dD_dR"] is not None else "   --"
                log(f"  {r['layer_row']:>4} {r['median_norm']:>9.2f} {r['sigma_R']:>8.3f} "
                    f"{r['sep_R']:>7.3f} {cs:>10} | "
                    + " ".join(f"{pn.get(a, 0):7.1f}%" for a in ALPHAS))
                log(f"       {'':>9} {'':>8} {'':>7} {'':>10} | "
                    + " ".join(f"{sg.get(a, 0):6.2f}s " for a in ALPHAS))

        show(lambda l: mid_lo <= l <= mid_hi, f"MIDDLE LAYERS (rows {mid_lo}-{mid_hi})")
        show(lambda l: l < mid_lo or l > mid_hi, "REMAINING LAYERS")

    # ---- verdict ----------------------------------------------------------- #
    log("\n" + "=" * 78)
    log("VERDICT: is one absolute alpha grid viable across all three models?")
    log(f"  a layer/alpha is USABLE if >= {MIN_EFFECT_SIGMA} sigma AND <= {MAX_SAFE_FRAC:.0%} of ||h||")
    any_ok = False
    for a in ALPHAS:
        ok_by_model = {}
        for slug, m in report["models"].items():
            ok = [r["layer_row"] for r in m["layers"]
                  if r["steerable"] and r["alpha_in_sigma"] and r["alpha_pct_of_norm"]
                  and r["alpha_in_sigma"][a] >= MIN_EFFECT_SIGMA
                  and r["alpha_pct_of_norm"][a] <= 100 * MAX_SAFE_FRAC]
            ok_by_model[slug] = ok
        shared = set.intersection(*(set(v) for v in ok_by_model.values())) if ok_by_model else set()
        log(f"  alpha={a:<3d} usable rows: "
            + "  ".join(f"{s}={len(v)}" for s, v in ok_by_model.items())
            + f"   | shared across all 3: {len(shared)}"
            + (f" -> {sorted(shared)[:12]}" if shared else ""))
        any_ok = any_ok or bool(shared)
    log(f"\n  => {'VIABLE at some alphas' if any_ok else 'NOT VIABLE — use relative/normalised alpha'}")

    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
