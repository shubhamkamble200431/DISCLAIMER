from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "data" / "generated"
STUDY = ROOT / "data" / "study"
PRED = ROOT.parent / "classifier" / "results" / "predictions" / "generated_predictions.parquet"

MODELS = ["qwen3-5-4b", "olmo3-7b"]
LABELS = ["NORMAL", "DISCLAIMER", "REFUSAL"]


def load_stack(model_root: Path, prompt_ids, n_rows, log):
    hid = model_root / "hidden"
    mats, kept = [], []
    t0 = time.perf_counter()
    for i, pid in enumerate(prompt_ids):
        fs = sorted((hid / str(pid)).glob("prompt_last_layer_*.npy"))
        if len(fs) != n_rows:
            continue
        mats.append(np.stack([np.load(f).astype(np.float32).ravel() for f in fs]))
        kept.append(int(pid))
        if (i + 1) % 100 == 0:
            log(f"      {i + 1}/{len(prompt_ids)}  {time.perf_counter() - t0:.0f}s")
    return np.stack(mats), kept


def cv_bacc(X, y, seed, folds=5):
    u, c = np.unique(y, return_counts=True)
    if len(u) < 2 or c.min() < folds:
        return None
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, C=1.0, n_jobs=None))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    preds = np.empty_like(y)
    for tr, te in skf.split(X, y):
        preds[te] = clf.fit(X[tr], y[tr]).predict(X[te])
    return float(balanced_accuracy_score(y, preds))


def dim_direction(X, lab, pos, neg):
    a, b = X[lab == pos], X[lab == neg]
    if len(a) < 2 or len(b) < 2:
        return None
    d = a.mean(0) - b.mean(0)
    n = np.linalg.norm(d)
    return (d / n) if n > 1e-8 else None


def split_half_reliability(X, lab, pos, neg, n_halves, rng):
    idx = np.arange(len(X))
    cos = []
    for _ in range(n_halves):
        perm = rng.permutation(idx)
        h1, h2 = perm[: len(perm) // 2], perm[len(perm) // 2:]
        d1 = dim_direction(X[h1], lab[h1], pos, neg)
        d2 = dim_direction(X[h2], lab[h2], pos, neg)
        if d1 is None or d2 is None:
            continue
        cos.append(float(np.dot(d1, d2)))
    return (float(np.mean(cos)), float(np.std(cos))) if cos else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--n-halves", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", default=",".join(MODELS))
    args = ap.parse_args()
    log = print

    splits = json.loads((STUDY / "study_splits.json").read_text(encoding="utf-8"))
    probe_ids = splits["splits"]["probing"]
    dir_ids = splits["splits"]["direction"]
    log(f"study split sha256={splits['sha256'][:16]}  probing n={len(probe_ids)}  "
        f"direction n={len(dir_ids)}")

    pred = pd.read_parquet(PRED)
    report = {"study_split_sha256": splits["sha256"], "top_k": args.top_k,
              "n_halves": args.n_halves, "seed": args.seed, "models": {}}
    selected = {}

    for slug in [s.strip() for s in args.models.split(",") if s.strip()]:
        root = GEN / slug
        man = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        n_rows, n_blocks = man["model"]["n_layer_rows"], man["model"]["n_blocks"]
        log(f"\n=== {slug}  blocks={n_blocks} rows={n_rows} H={man['model']['hidden_size']} ===")

        lab_map = pred[pred.gen_model == slug].set_index("prompt_id")["longformer_label"]

        log("    loading probing split")
        Xp, kept_p = load_stack(root, probe_ids, n_rows, log)
        yp = lab_map.loc[kept_p].to_numpy()
        log("    loading direction split (for the supplementary reliability column)")
        Xd, kept_d = load_stack(root, dir_ids, n_rows, log)
        yd = lab_map.loc[kept_d].to_numpy()

        cnt = {c: int((yp == c).sum()) for c in LABELS}
        log(f"    probing labels {cnt}   direction n={len(kept_d)}")

        y3 = np.searchsorted(np.array(LABELS), yp)
        y3 = np.array([LABELS.index(v) for v in yp])

        rows = []
        for l in range(n_rows):
            Xl, Xdl = Xp[:, l, :], Xd[:, l, :]
            rng = np.random.default_rng(args.seed + l)

            b3 = cv_bacc(Xl, y3, args.seed)
            mRN = np.isin(yp, ["REFUSAL", "NORMAL"])
            mDN = np.isin(yp, ["DISCLAIMER", "NORMAL"])
            bRN = cv_bacc(Xl[mRN], (yp[mRN] == "REFUSAL").astype(int), args.seed)
            bDN = cv_bacc(Xl[mDN], (yp[mDN] == "DISCLAIMER").astype(int), args.seed)

            rR, sR = split_half_reliability(Xl, yp, "REFUSAL", "NORMAL", args.n_halves, rng)
            rD, sD = split_half_reliability(Xl, yp, "DISCLAIMER", "NORMAL", args.n_halves,
                                            np.random.default_rng(args.seed + 1000 + l))
            rR_d, _ = split_half_reliability(Xdl, yd, "REFUSAL", "NORMAL", args.n_halves,
                                             np.random.default_rng(args.seed + 2000 + l))
            rD_d, _ = split_half_reliability(Xdl, yd, "DISCLAIMER", "NORMAL", args.n_halves,
                                             np.random.default_rng(args.seed + 3000 + l))

            rel = (np.mean([r for r in (rR, rD) if r is not None])
                   if (rR is not None or rD is not None) else None)
            score = (b3 * rel) if (b3 is not None and rel is not None) else None

            rows.append({
                "layer_row": l,
                "steerable": bool(1 <= l <= n_blocks - 1),
                "probe_bacc_3": None if b3 is None else round(b3, 4),
                "probe_bacc_RN": None if bRN is None else round(bRN, 4),
                "probe_bacc_DN": None if bDN is None else round(bDN, 4),
                "rel_R": None if rR is None else round(rR, 4),
                "rel_R_sd": None if sR is None else round(sR, 4),
                "rel_D": None if rD is None else round(rD, 4),
                "rel_D_sd": None if sD is None else round(sD, 4),
                "rel_R_direction_split": None if rR_d is None else round(rR_d, 4),
                "rel_D_direction_split": None if rD_d is None else round(rD_d, 4),
                "score": None if score is None else round(score, 4),
            })

        cand = [r for r in rows if r["steerable"] and r["score"] is not None]
        cand.sort(key=lambda r: -r["score"])
        top = sorted(r["layer_row"] for r in cand[: args.top_k])
        selected[slug] = top

        report["models"][slug] = {
            "n_blocks": n_blocks, "n_layer_rows": n_rows,
            "probing_label_counts": cnt, "n_probing": len(kept_p),
            "n_direction": len(kept_d),
            "layers": rows,
            "top_layers": top,
            "top_layers_relative_depth": [round(l / n_blocks, 3) for l in top],
        }

        log(f"\n  {'row':>4} {'bacc3':>7} {'baccRN':>7} {'baccDN':>7} "
            f"{'rel_R':>7} {'rel_D':>7} {'score':>7}  {'(dir-split rel_R/rel_D)':>24}")
        for r in rows:
            if not r["steerable"]:
                log(f"  {r['layer_row']:>4}   -- excluded from selection "
                    f"({'embedding row, no direction' if r['layer_row'] == 0 else 'post-final-norm'})")
                continue
            mark = " <<" if r["layer_row"] in top else ""
            f = lambda v: f"{v:>7.3f}" if v is not None else "     --"
            log(f"  {r['layer_row']:>4} {f(r['probe_bacc_3'])} {f(r['probe_bacc_RN'])} "
                f"{f(r['probe_bacc_DN'])} {f(r['rel_R'])} {f(r['rel_D'])} {f(r['score'])}"
                f"   {f(r['rel_R_direction_split'])}/{f(r['rel_D_direction_split'])}{mark}")
        log(f"\n  TOP {args.top_k}: {top}")
        log(f"  relative depth: {[round(l / n_blocks, 2) for l in top]}")

    payload = json.dumps({k: selected[k] for k in sorted(selected)}, sort_keys=True)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    report["selected_layers"] = selected
    report["selected_sha256"] = digest

    (STUDY / "layer_probe.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (STUDY / "selected_layers.json").write_text(
        json.dumps({"sha256": digest, "study_split_sha256": splits["sha256"],
                    "selected": selected}, indent=2), encoding="utf-8")
    log(f"\nselected-layer sha256 = {digest[:16]}...")
    log(f"wrote {STUDY / 'layer_probe.json'}")
    log(f"wrote {STUDY / 'selected_layers.json'}")


if __name__ == "__main__":
    main()
