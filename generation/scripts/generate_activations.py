from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import generate as G


class Tee:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def env_info():
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except Exception:
        pass
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_capability"] = list(torch.cuda.get_device_capability(0))
        info["gpu_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    return info


def do_preflight(specs, log):
    log("preflight (config.json only, no weights downloaded):")
    ok = True
    for s in specs:
        try:
            p = G.preflight(s)
        except Exception as e:
            log(f"  [FAIL] {s.slug:16s} {s.hf_id}  {type(e).__name__}: {str(e)[:160]}")
            ok = False
            continue
        rows = (p["num_hidden_layers"] + 1) if p["num_hidden_layers"] else "?"
        mark = "ok  " if p["supported"] else "FAIL"
        log(f"  [{mark}] {s.slug:16s} {s.hf_id:34s} type={p['model_type']:10s} "
            f"hidden={p['hidden_size']} blocks={p['num_hidden_layers']} rows={rows}")
        if not p["supported"]:
            log(f"         model_type '{p['model_type']}' is unknown to transformers "
                f"{env_info().get('transformers')}; upgrade transformers or set "
                f"trust_remote_code: true")
            ok = False
    return ok


def estimate_bytes(spec_rows, hidden, n_points, n_prompts, dtype_bytes=2):
    return spec_rows * hidden * n_points * n_prompts * dtype_bytes


def run_model(spec, prompts, cfg, args, log_root):
    root, ans_dir, hid_dir = G.model_dirs(cfg["output"]["root"], spec.slug)
    ans_dir.mkdir(parents=True, exist_ok=True)
    hid_dir.mkdir(parents=True, exist_ok=True)
    log = Tee(root / "run.log")
    log(f"\n=== {spec.slug} | {spec.hf_id} | {datetime.now(timezone.utc).isoformat()} ===")

    layout = cfg["output"].get("hidden_layout", "per_layer")
    todo = [r for r in prompts.itertuples(index=False)
            if args.overwrite
            or not G.already_done(ans_dir, hid_dir, int(r.prompt_id), layout)]
    n_skip = len(prompts) - len(todo)
    log(f"  prompts: {len(prompts)} total, {n_skip} already done, {len(todo)} to run")
    if not todo:
        log("  nothing to do")
        log.close()
        return {"model": spec.slug, "n_done": 0, "n_skipped": n_skip, "n_failed": 0}

    runner = G.Runner(spec, cfg["generation"], cfg["capture"], log=log)
    n_points = len([p for p in cfg["capture"]["points"]])
    est = estimate_bytes(runner.n_blocks + 1, runner.hidden_size, n_points, len(todo))
    n_files = (runner.n_blocks + 1) * n_points * len(todo) if layout == "per_layer" else len(todo)
    log(f"  capture: {n_points} points x {runner.n_blocks + 1} layer rows x "
        f"{runner.hidden_size} dims -> ~{est / 1e9:.2f} GB for {len(todo)} prompts")
    log(f"  layout:  {layout} -> {n_files:,} hidden files")

    store_dtype = cfg["output"].get("store_dtype", "float16")
    compress = bool(cfg["output"].get("compress", False))
    jsonl = open(root / "answers.jsonl", "a", encoding="utf-8")

    t_start = time.perf_counter()
    n_done, n_failed, failures, promoted_total = 0, 0, [], 0
    gen_tokens = 0
    for i, row in enumerate(todo, 1):
        pid = int(row.prompt_id)
        try:
            answer, vecs, meta = runner.run(row.prompt)
            rec = G.write_prompt(ans_dir, hid_dir, row, answer, vecs, meta,
                                 spec, store_dtype, compress, layout)
            jsonl.write(json.dumps({k: rec[k] for k in
                                    ("prompt_id", "prompt", "answer", "splits",
                                     "n_prompt_tokens", "n_generated_tokens",
                                     "hit_max_new_tokens")},
                                   ensure_ascii=False) + "\n")
            jsonl.flush()
            n_done += 1
            gen_tokens += meta["n_generated_tokens"]
            promoted_total += len(rec["fp16_overflow_promoted"])
            if meta["missing_gen_points"]:
                log(f"    [{pid}] produced 0 tokens; missing {meta['missing_gen_points']}")
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            n_failed += 1
            failures.append({"prompt_id": pid, "error": "OutOfMemoryError", "msg": str(e)[:300]})
            log(f"    [{pid}] OOM -- skipped (lower max_new_tokens or use a device with more memory)")
        except Exception as e:
            n_failed += 1
            failures.append({"prompt_id": pid, "error": type(e).__name__, "msg": str(e)[:300]})
            log(f"    [{pid}] {type(e).__name__}: {str(e)[:200]}")
            if args.strict:
                traceback.print_exc()
                raise
        if i % max(1, args.log_every) == 0 or i == len(todo):
            el = time.perf_counter() - t_start
            rate = i / el
            peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            log(f"  [{i}/{len(todo)}] {rate * 60:.1f} prompt/min  "
                f"{gen_tokens / el:.1f} tok/s  eta {(len(todo) - i) / rate / 60:.1f} min  "
                f"peakVRAM {peak:.2f} GB  failed {n_failed}")

    jsonl.close()
    elapsed = time.perf_counter() - t_start

    manifest = {
        "model": {"name": spec.name, "hf_id": spec.hf_id, "slug": spec.slug,
                  "dtype": spec.dtype, "device_map": spec.device_map,
                  "attn_implementation": spec.attn_implementation,
                  "system_prompt": spec.system_prompt,
                  "chat_template_kwargs": spec.chat_template_kwargs,
                  "hidden_size": runner.hidden_size,
                  "n_blocks": runner.n_blocks,
                  "n_layer_rows": runner.n_blocks + 1},
        "generation": runner.gen_kwargs,
        "capture": cfg["capture"],
        "output": cfg["output"],
        "prompts": {"n_total": int(len(prompts)), "n_run": n_done,
                    "n_skipped_existing": n_skip, "n_failed": n_failed,
                    "splits": args.splits},
        "totals": {"generated_tokens": gen_tokens,
                   "wall_seconds": round(elapsed, 1),
                   "tokens_per_second": round(gen_tokens / elapsed, 2) if elapsed else None,
                   "arrays_promoted_to_fp32": promoted_total},
        "failures": failures,
        "env": env_info(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    log(f"  done: {n_done} ok, {n_failed} failed, {n_skip} skipped in {elapsed / 60:.1f} min")
    log(f"  -> {root}")
    log.close()

    del runner
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    return {"model": spec.slug, "n_done": n_done, "n_skipped": n_skip, "n_failed": n_failed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/generate.yaml")
    ap.add_argument("--models", default=None,
                    help="comma-separated slugs/hf_ids; default = every model in the config")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--limit", type=int, default=None, help="first N prompts (after sort)")
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny model + few prompts + short generations, from config.smoke")
    ap.add_argument("--list-models", action="store_true",
                    help="print the roster from the config and exit (no network)")
    ap.add_argument("--preflight", action="store_true",
                    help="check arch support and print sizes, then exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve prompts and report what would run, then exit")
    ap.add_argument("--strict", action="store_true", help="abort on the first prompt failure")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = G.load_config(G.repo_path(args.config))
    args.splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    if args.smoke:
        sm = cfg.get("smoke") or {}
        cfg["models"] = sm.get("models", cfg["models"])
        cfg["generation"] = {**cfg["generation"], **(sm.get("generation") or {})}
        cfg["output"] = {**cfg["output"],
                         "root": sm.get("root", cfg["output"]["root"] + "_smoke")}
        if args.limit is None:
            args.limit = int(sm.get("n_prompts", 3))
    if args.max_new_tokens is not None:
        cfg["generation"]["max_new_tokens"] = args.max_new_tokens

    seed = args.seed if args.seed is not None else cfg["generation"].get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    out_root = G.repo_path(cfg["output"]["root"])
    out_root.mkdir(parents=True, exist_ok=True)
    log = print

    log(f"config      {args.config}")
    log(f"output root {out_root}")
    for k, v in env_info().items():
        log(f"  {k:16s} {v}")

    if args.list_models:
        all_specs = G.parse_models(cfg)
        log("")
        log("models in this config (pass a slug to --models):")
        log("")
        log(f"  {'slug':18s} {'hf_id':36s} {'dtype':10s} system_prompt")
        for s in all_specs:
            log(f"  {s.slug:18s} {s.hf_id:36s} {s.dtype:10s} {s.system_prompt}")
        log("")
        log("  --models takes one slug, several comma-separated, or a full hf_id.")
        log("  Omit --models to run every model above, one after another.")
        log("")
        sys.exit(0)

    specs = G.parse_models(cfg, args.models.split(",") if args.models else None)
    log(f"models      {[s.slug for s in specs]}")

    if args.preflight:
        sys.exit(0 if do_preflight(specs, log) else 1)

    prompts = G.build_prompt_table(cfg, args.splits)
    if args.limit:
        prompts = prompts.head(args.limit).copy()
    prompts.to_parquet(out_root / "prompts.parquet", index=False)
    log(f"prompts     {len(prompts)} unique from splits={args.splits} "
        f"(chars: median {int(prompts.n_chars.median())}, max {int(prompts.n_chars.max())})")

    if args.dry_run:
        for s in specs:
            root, ans_dir, hid_dir = G.model_dirs(cfg["output"]["root"], s.slug)
            done = sum(1 for r in prompts.itertuples(index=False)
                       if G.already_done(ans_dir, hid_dir, int(r.prompt_id),
                                         cfg["output"].get("hidden_layout", "per_layer")))
            log(f"  {s.slug:16s} {s.hf_id:34s} {done} done / {len(prompts) - done} to run "
                f"-> {root}")
        sys.exit(0)

    summary = []
    for s in specs:
        try:
            summary.append(run_model(s, prompts, cfg, args, log))
        except Exception as e:
            log(f"!! {s.slug} aborted: {type(e).__name__}: {e}")
            traceback.print_exc()
            summary.append({"model": s.slug, "error": f"{type(e).__name__}: {e}"})
            if args.strict:
                raise

    log("\nsummary")
    for s in summary:
        log("  " + json.dumps(s))
    with open(out_root / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump({"finished_utc": datetime.now(timezone.utc).isoformat(),
                   "splits": args.splits, "n_prompts": int(len(prompts)),
                   "results": summary}, f, indent=2)


if __name__ == "__main__":
    main()
