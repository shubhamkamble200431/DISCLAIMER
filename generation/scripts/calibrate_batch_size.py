"""Per-model batch-size calibration for the real steering run.

For each model in configs/study.yaml's real roster, try batch sizes descending
through a candidate list on a small real sample (default 50 prompts) at the REAL
generation.max_new_tokens (1024). The largest batch size that completes with ZERO
OOM halvings is written as that model's calibrated batch size.

This does NOT touch data/steered/ or any study-tracked artifact -- it calls
S.BatchedGenerator directly (the same primitive step2's steer stage uses) on an
unsteered (hook_ctx=None) pass, so there is nothing for the batch-plan-hash /
stale-artifact guards to object to later. Safe to run before or after step 1.

Usage:
  python -X utf8 scripts/calibrate_batch_size.py --config configs/study.yaml \
      --candidates 32,16,8,4 --n-prompts 50
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import generate as G   # noqa: E402
from src import steer as S      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/study.yaml")
    ap.add_argument("--candidates", default="32,16,8,4")
    ap.add_argument("--n-prompts", type=int, default=50)
    ap.add_argument("--models", default=None, help="comma-separated slugs; default: all")
    ap.add_argument("--out", default="data/study/batch_size_calibration.json")
    args = ap.parse_args()

    cfg = G.load_config(G.repo_path(args.config))
    candidates = [int(x) for x in args.candidates.split(",") if x.strip()]
    specs = G.parse_models(cfg, args.models.split(",") if args.models else None)

    table = G.build_prompt_table(cfg)
    rows = list(table.head(args.n_prompts).itertuples(index=False))
    print(f"calibrating on {len(rows)} prompts, candidates={candidates}, "
          f"max_new_tokens={cfg['generation']['max_new_tokens']}")

    results = {}
    for spec in specs:
        print(f"\n=== {spec.slug}  ({spec.hf_id}) ===")
        t0 = time.perf_counter()
        runner = G.Runner(spec, cfg["generation"], cfg["capture"], log=print)
        print(f"  loaded in {time.perf_counter() - t0:.1f}s")

        chosen = None
        per_candidate = {}
        for bs in candidates:
            gen = S.BatchedGenerator(runner, batch_size=bs, log=print)
            plan = gen.plan_batches(rows)
            torch.cuda.reset_peak_memory_stats()
            t1 = time.perf_counter()
            oom_hard = False
            try:
                gen.generate(rows, plan, hook_ctx=None)
            except torch.cuda.OutOfMemoryError:
                oom_hard = True
                torch.cuda.empty_cache()
            dt = time.perf_counter() - t1
            peak = torch.cuda.max_memory_allocated() / 1e9
            ok = (not oom_hard) and gen.n_oom_halvings == 0
            per_candidate[bs] = {"ok": ok, "oom_halvings": gen.n_oom_halvings,
                                 "hard_oom": oom_hard, "peak_vram_gb": round(peak, 2),
                                 "seconds": round(dt, 1)}
            print(f"  bs={bs:3d}  ok={ok}  oom_halvings={gen.n_oom_halvings}  "
                  f"hard_oom={oom_hard}  peakVRAM={peak:.2f}GB  {dt:.1f}s")
            if ok and chosen is None:
                chosen = bs
                # Largest-first candidate list: first clean one wins, stop testing
                # smaller sizes -- they would only confirm what we already know.
                break

        if chosen is None:
            print(f"  !! no candidate batch size was clean for {spec.slug}; "
                  f"falling back to the smallest candidate {min(candidates)}")
            chosen = min(candidates)

        results[spec.slug] = {"hf_id": spec.hf_id, "chosen_batch_size": chosen,
                              "candidates": per_candidate}
        del runner
        torch.cuda.empty_cache()

    out_path = G.repo_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"generated_utc": datetime.now(timezone.utc).isoformat(),
           "n_prompts": len(rows), "candidates": candidates,
           "max_new_tokens": cfg["generation"]["max_new_tokens"], "results": results}
    out_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")
    print("\nchosen batch sizes:")
    for slug, r in results.items():
        print(f"  {slug:14s} {r['chosen_batch_size']}")


if __name__ == "__main__":
    main()
