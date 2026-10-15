from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import batched_capture as BC
from src import generate as G
from src import steer as S

MIN_COS = 0.998
MAX_REL_L2 = 0.05


def capture(runner, texts):
    try:
        return BC.prefill_last_states(runner, texts)
    except torch.cuda.OutOfMemoryError:
        pass
    gc.collect()
    torch.cuda.empty_cache()
    if len(texts) == 1:
        raise RuntimeError("OOM on a single prompt")
    h = len(texts) // 2
    return torch.cat([capture(runner, texts[:h]), capture(runner, texts[h:])])


def run_plan(runner, texts, batches):
    out = [None] * len(texts)
    for b in batches:
        uniq = list(dict.fromkeys(b))
        vec = capture(runner, [texts[i] for i in uniq])
        for k, i in enumerate(uniq):
            out[i] = vec[k]
    return torch.stack(out)


def compare(ref, got, n_blocks):
    rows = slice(1, n_blocks)
    a, b = got[:, rows].double(), ref[:, rows].double()
    cos = (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-12)
    rel = (a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-12)
    min_cos_per_row = cos.min(0).values
    max_rel_per_row = rel.max(0).values
    worst = int(min_cos_per_row.argmin())
    return {
        "min_cos": float(cos.min()), "max_rel_l2": float(rel.max()),
        "worst_row": worst + 1,
        "worst_row_min_cos": float(min_cos_per_row[worst]),
        "median_cos": float(cos.median()), "median_rel_l2": float(rel.median()),
        "row0_max_abs_diff": float((got[:, 0] - ref[:, 0]).abs().max()),
        "per_row_min_cos": [round(float(x), 6) for x in min_cos_per_row],
        "per_row_max_rel_l2": [round(float(x), 6) for x in max_rel_per_row],
        "pass": bool(cos.min() >= MIN_COS and rel.max() <= MAX_REL_L2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/study.yaml")
    ap.add_argument("--models", default=None)
    ap.add_argument("--n-prompts", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--mixed-batch-size", type=int, default=8)
    ap.add_argument("--out", default="data/study/generate_label_split_batch_parity.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = G.load_config(G.repo_path(args.config))
    if args.smoke:
        cfg["models"] = cfg["smoke"]["models"]
    specs = G.parse_models(cfg, args.models.split(",") if args.models else None)
    table = G.build_prompt_table(cfg)
    rows_all = list(table.itertuples(index=False))

    out_path = G.repo_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = (json.loads(out_path.read_text(encoding="utf-8"))
           if out_path.exists() else {"models": {}})
    doc["criteria"] = {"min_cos": MIN_COS, "max_rel_l2": MAX_REL_L2,
                       "rows": "steerable rows 1..n_blocks-1", "n_prompts": args.n_prompts,
                       "bucketed_batch_size": args.batch_size,
                       "mixed_batch_size": args.mixed_batch_size}

    for spec in specs:
        print(f"\n=== {spec.slug}  ({spec.hf_id}) ===", flush=True)
        t0 = time.perf_counter()
        runner = G.Runner(spec, cfg["generation"], cfg["capture"], log=print)
        gen = S.BatchedGenerator(runner, batch_size=args.batch_size, max_prompt_tokens=2048,
                                 log=lambda *a: None)

        lengths = [gen._render(r)[1] for r in rows_all]
        eligible = sorted((i for i, n in enumerate(lengths) if n <= 2048),
                          key=lambda i: (lengths[i], int(rows_all[i].prompt_id)))
        pick = [eligible[j] for j in
                np.unique(np.linspace(0, len(eligible) - 1, args.n_prompts).astype(int))]
        rows = [rows_all[i] for i in pick]
        texts = [gen._render(r)[0] for r in rows]
        print(f"  {len(rows)} prompts, rendered length {lengths[pick[0]]}..{lengths[pick[-1]]}",
              flush=True)

        ref = torch.stack([BC.single_prompt_states(runner, r.prompt) for r in rows])
        print(f"  batch-1 reference captured in {time.perf_counter() - t0:.0f}s", flush=True)

        bucketed = run_plan(runner, texts, gen.plan_batches(rows))
        rng = np.random.default_rng(0)
        order = rng.permutation(len(rows)).tolist()
        mixed_batches = [order[k:k + args.mixed_batch_size]
                         for k in range(0, len(order), args.mixed_batch_size)]
        mixed = run_plan(runner, texts, mixed_batches)

        res_b = compare(ref, bucketed, runner.n_blocks)
        res_m = compare(ref, mixed, runner.n_blocks)
        ok = res_b["pass"] and res_m["pass"]
        for name, r in (("bucketed", res_b), ("mixed   ", res_m)):
            print(f"  {name}: min_cos={r['min_cos']:.6f} max_rel_l2={r['max_rel_l2']:.4f} "
                  f"worst_row={r['worst_row']} (min_cos {r['worst_row_min_cos']:.6f}) "
                  f"row0_diff={r['row0_max_abs_diff']:.3g}  -> {'PASS' if r['pass'] else 'FAIL'}",
                  flush=True)
        print(f"  {spec.slug}: {'PASS -> batched capture allowed' if ok else 'FAIL -> stay at batch 1'}",
              flush=True)

        doc["models"][spec.slug] = {
            "hf_id": spec.hf_id, "pass": ok,
            "chosen_generate_batch_size": args.batch_size if ok else 1,
            "n_prompts": len(rows), "bucketed": res_b, "mixed": res_m,
            "checked_utc": datetime.now(timezone.utc).isoformat()}
        doc["written_utc"] = datetime.now(timezone.utc).isoformat()
        out_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")

        del runner, gen
        torch.cuda.empty_cache()

    print(f"\nwrote {out_path}")
    for slug, r in doc["models"].items():
        print(f"  {slug:14s} pass={r['pass']}  generate batch size -> "
              f"{r['chosen_generate_batch_size']}")


if __name__ == "__main__":
    main()
