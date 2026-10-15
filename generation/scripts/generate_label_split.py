from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import generate as G
from src import steer as S

STAGES = ("preflight", "generate", "label", "split", "all")

SPLIT_ORDER = S.SPLIT_NAMES

CSTAR_CANDIDATE = "longformer"


class Tee:

    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass


def canonical_sha256(payload) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def commit_json(path: Path, payload: dict, log) -> str:
    digest = canonical_sha256(payload)
    out = dict(payload)
    out["sha256"] = digest
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
        f.write("\n")
    log(f"  wrote {path}  sha256={digest[:16]}...")
    return digest


def verify_json(path: Path, payload: dict, log) -> bool:
    if not path.exists():
        log(f"  [VERIFY] {path} does not exist")
        return False
    old = json.loads(path.read_text(encoding="utf-8"))
    stored = str(old.get("sha256"))

    self_digest = canonical_sha256({k: v for k, v in old.items() if k != "sha256"})
    self_ok = self_digest == stored
    if not self_ok:
        log(f"  [VERIFY] {path.name}: SELF-HASH MISMATCH -- the file body does not hash "
            f"to the sha256 stored inside it (stored {stored[:16]}, body "
            f"{self_digest[:16]}). The artifact was edited or truncated after it was "
            f"committed; do not trust anything computed from it.")

    got = canonical_sha256(payload)
    ok = stored == got and self_ok
    log(f"  [VERIFY] {path.name}: committed {stored[:16]}  "
        f"recomputed {got[:16]}  {'MATCH' if stored == got else 'MISMATCH'}"
        f"  self-hash {'ok' if self_ok else 'BROKEN'}")
    if stored != got:
        for k in sorted(set(old) | set(payload)):
            if k == "sha256":
                continue
            if old.get(k) != payload.get(k):
                log(f"           first differing key: {k!r}")
                break
    return ok


def env_info() -> dict:
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
        info["gpu_total_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    return info


def prompt_table(cfg, args, log) -> pd.DataFrame:
    tbl = G.build_prompt_table(cfg, ["train", "val", "test"])
    bad = tbl[tbl["splits"].astype(str).str.contains(",")]
    if len(bad):
        raise RuntimeError(
            f"{len(bad)} prompt_id(s) appear in more than one of C*'s train/val/test "
            f"splits (e.g. {bad.prompt_id.head(3).tolist()}); cstar_split would not be a "
            f"single value and the stratification would be wrong")
    tbl = tbl.rename(columns={"splits": "cstar_split"})
    if args.limit:
        tbl = tbl.head(args.limit).copy()
    log(f"prompts     {len(tbl)} unique  cstar_split="
        f"{tbl.cstar_split.value_counts().to_dict()}  "
        f"(chars: median {int(tbl.n_chars.median())}, max {int(tbl.n_chars.max())})")
    return tbl.reset_index(drop=True)


def render_probe(spec, prompt: str, log) -> dict:
    from transformers import AutoTokenizer

    out = {"chat_template_available": False, "error": None}
    tok = AutoTokenizer.from_pretrained(spec.hf_id, trust_remote_code=spec.trust_remote_code)
    out["tokenizer_class"] = type(tok).__name__
    out["has_chat_template_attr"] = bool(getattr(tok, "chat_template", None))

    msgs = []
    if spec.system_prompt:
        msgs.append({"role": "system", "content": spec.system_prompt})
    msgs.append({"role": "user", "content": prompt})
    ctk = spec.chat_template_kwargs or {}
    try:
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **ctk)
        ref = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                      **ctk)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        return out

    seq = ref["input_ids"] if (hasattr(ref, "keys") and "input_ids" in ref) else ref
    if hasattr(seq, "tolist"):
        seq = seq.tolist()
    if len(seq) and isinstance(seq[0], (list, tuple)):
        seq = seq[0]
    ref_ids = [int(v) for v in seq]
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    ids = [int(v) for v in ids]

    out.update({
        "chat_template_available": True,
        "system_prompt": spec.system_prompt,
        "chat_template_kwargs": ctk,
        "n_prompt_tokens": len(ref_ids),
        "rendered_tail": text[-240:],
        "final_token_id": ref_ids[-1] if ref_ids else None,
        "final_token": tok.convert_ids_to_tokens(ref_ids[-1]) if ref_ids else None,
        "final_token_decoded": tok.decode(ref_ids[-1:]) if ref_ids else None,
        "last_5_tokens": tok.convert_ids_to_tokens(ref_ids[-5:]),
        "tokenizer_parity": ids == ref_ids,
        "pad_token": tok.pad_token,
        "eos_token": tok.eos_token,
    })
    return out


def stage_preflight(cfg, specs, prompts, args, study_dir, log) -> dict:
    log("\n=== stage: preflight ===")
    log("  AutoConfig only for arch support; AutoTokenizer for the chat-template dump. "
        "No weights are downloaded.")
    probe_prompt = str(prompts.iloc[0]["prompt"])
    probe_pid = int(prompts.iloc[0]["prompt_id"])
    log(f"  chat-template probe uses prompt_id {probe_pid} ({len(probe_prompt)} chars)")

    models, ok = {}, True
    for spec in specs:
        log(f"\n  -- {spec.slug}  {spec.hf_id}")
        rec = {"name": spec.name, "hf_id": spec.hf_id, "slug": spec.slug,
               "dtype": spec.dtype, "attn_implementation": spec.attn_implementation,
               "trust_remote_code": spec.trust_remote_code, "note": spec.note}
        try:
            p = G.preflight(spec)
        except Exception as e:
            log(f"     [FAIL] AutoConfig: {type(e).__name__}: {str(e)[:200]}")
            rec["config_error"] = f"{type(e).__name__}: {str(e)[:200]}"
            models[spec.slug], ok = rec, False
            continue

        rows = (p["num_hidden_layers"] + 1) if p["num_hidden_layers"] else None
        rec.update(p)
        rec["n_layer_rows"] = rows
        rec["steerable_rows"] = [1, (p["num_hidden_layers"] - 1)] if p["num_hidden_layers"] else None
        log(f"     arch   type={p['model_type']} hidden={p['hidden_size']} "
            f"blocks={p['num_hidden_layers']} rows={rows} "
            f"supported={p['supported']} architectures={p['architectures']}")
        if not p["supported"]:
            log(f"     [FAIL] model_type {p['model_type']!r} is unknown to transformers "
                f"{env_info().get('transformers')}; upgrade transformers or set "
                f"trust_remote_code: true")
            ok = False

        if spec.attn_implementation not in S.ALLOWED_ATTN:
            log(f"     !! WARNING: attn_implementation={spec.attn_implementation!r} is not "
                f"one of {list(S.ALLOWED_ATTN)}. directions_steering.py will clamp batch size to "
                f"{S.EAGER_MAX_BATCH} and cap prompts at {S.EAGER_MAX_PROMPT_TOKENS} "
                f"tokens, which multiplies the steering budget by ~2.")
            rec["batching_clamped"] = True

        try:
            tp = render_probe(spec, probe_prompt, log)
        except Exception as e:
            log(f"     [FAIL] tokenizer: {type(e).__name__}: {str(e)[:200]}")
            rec["template_error"] = f"{type(e).__name__}: {str(e)[:200]}"
            models[spec.slug], ok = rec, False
            continue

        rec["chat_template"] = tp
        if not tp["chat_template_available"]:
            log(f"     [FAIL] no usable chat template ({tp['error']}). G.Runner.render() "
                f"would fall back to the RAW prompt, so prompt_last would be the last "
                f"token of the user's question, not the assistant turn's opening. "
                f"Disqualifying.")
            ok = False
        else:
            log(f"     tokens {tp['n_prompt_tokens']}  final token "
                f"{tp['final_token']!r} -> decoded {tp['final_token_decoded']!r}")
            log(f"     last 5 {tp['last_5_tokens']}")
            log(f"     rendered tail: {tp['rendered_tail']!r}")
            if not tp["tokenizer_parity"]:
                log(f"     [FAIL] re-tokenising the rendered text with "
                    f"add_special_tokens=False does NOT reproduce "
                    f"apply_chat_template(tokenize=True). directions_steering.py's batched path would "
                    f"steer a different position than the one captured here.")
                ok = False
        models[spec.slug] = rec

    payload = {
        "stage": "preflight",
        "seed": int(cfg["study"]["seed"]),
        "inputs": {"config": str(args.config),
                   "probe_prompt_id": probe_pid,
                   "probe_prompt_sha1": G.sha1(probe_prompt),
                   "n_prompts": int(len(prompts)),
                   "roster": [s.slug for s in specs]},
        "all_ok": bool(ok),
        "models": models,
        "env": env_info(),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    dest = study_dir / "preflight.json"
    if args.verify:
        payload.pop("env"), payload.pop("generated_utc")
        old = json.loads(dest.read_text(encoding="utf-8")) if dest.exists() else {}
        old.pop("env", None), old.pop("generated_utc", None), old.pop("sha256", None)
        same = old == payload
        log(f"  [VERIFY] preflight.json: {'MATCH' if same else 'MISMATCH'}")
        return {"ok": same}

    commit_json(dest, payload, log)
    log(f"\n  preflight {'PASSED' if ok else 'FAILED'} for {len(specs)} model(s)")
    return {"ok": ok, "preflight": payload}


_RUN_SIGNATURE_KEYS = ("hf_id", "dtype", "system_prompt", "chat_template_kwargs",
                       "max_new_tokens", "do_sample", "capture_points",
                       "capture_layers", "hidden_layout", "store_dtype")


def run_signature(spec, cfg) -> dict:
    gen = {**(cfg["generation"] or {}), **(spec.generation or {})}
    return {
        "hf_id": spec.hf_id,
        "dtype": spec.dtype,
        "system_prompt": spec.system_prompt,
        "chat_template_kwargs": spec.chat_template_kwargs or {},
        "max_new_tokens": gen.get("max_new_tokens"),
        "do_sample": bool(gen.get("do_sample", False)),
        "capture_points": list(cfg["capture"]["points"]),
        "capture_layers": cfg["capture"].get("layers", "all"),
        "hidden_layout": cfg["output"].get("hidden_layout", "stacked"),
        "store_dtype": cfg["output"].get("store_dtype", "float16"),
    }


def signature_from_manifest(man: dict) -> dict:
    m, g = man.get("model", {}), man.get("generation", {})
    c, o = man.get("capture", {}), man.get("output", {})
    return {
        "hf_id": m.get("hf_id"),
        "dtype": m.get("dtype"),
        "system_prompt": m.get("system_prompt"),
        "chat_template_kwargs": m.get("chat_template_kwargs") or {},
        "max_new_tokens": g.get("max_new_tokens"),
        "do_sample": bool(g.get("do_sample", False)),
        "capture_points": list(c.get("points", [])),
        "capture_layers": c.get("layers", "all"),
        "hidden_layout": o.get("hidden_layout", "per_layer"),
        "store_dtype": o.get("store_dtype", "float16"),
    }


def guard_model_dir(root: Path, spec, cfg, args, log) -> None:
    man_path = root / "manifest.json"
    if not man_path.exists():
        return
    want = run_signature(spec, cfg)
    have = signature_from_manifest(json.loads(man_path.read_text(encoding="utf-8")))
    diff = {k: (have.get(k), want.get(k)) for k in _RUN_SIGNATURE_KEYS
            if have.get(k) != want.get(k)}
    if not diff:
        return
    lines = "\n".join(f"      {k}: on disk {a!r}  config wants {b!r}"
                      for k, (a, b) in sorted(diff.items()))
    if args.overwrite:
        log(f"  !! {root} was produced by a different run configuration:\n{lines}\n"
            f"     --overwrite given: answers/, hidden/ and answers.jsonl are being "
            f"rewritten from scratch.")
        return
    raise RuntimeError(
        f"{root} holds answers from a DIFFERENT run configuration:\n{lines}\n"
        f"    Resuming would mix two experiments inside one answers.jsonl. Either move "
        f"that directory aside (e.g. rename it to {root.name}_mnt512) or re-run this "
        f"stage with --overwrite, which rewrites it in full.")


def reconcile_jsonl(root: Path, ans_dir: Path, log) -> set[int]:
    path = root / "answers.jsonl"
    seen, dup = [], set()
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            try:
                seen.append(int(json.loads(line)["prompt_id"]))
            except Exception:
                if i == len(lines) - 1:
                    log(f"  warning: dropping truncated final line of {path}")
                else:
                    raise ValueError(f"{path}:{i + 1} is not a valid record")
        dup = {p for p in seen if seen.count(p) > 1}
    if dup:
        raise RuntimeError(
            f"{path} contains {len(dup)} duplicated prompt_id(s) (e.g. "
            f"{sorted(dup)[:5]}); C* would score those answers twice. Rebuild the file "
            f"from answers/*.json or re-run this stage with --overwrite.")

    have = set(seen)
    orphans = sorted(int(p.stem) for p in ans_dir.glob("*.json")
                     if int(p.stem) not in have)
    if orphans:
        log(f"  reconciling answers.jsonl: {len(orphans)} finished prompt(s) had no "
            f"jsonl line (run killed mid-append); re-emitting from answers/*.json")
        recs = []
        for pid in orphans:
            r = json.loads((ans_dir / f"{pid}.json").read_text(encoding="utf-8"))
            recs.append({k: r[k] for k in JSONL_KEYS})
        S.write_answers_jsonl(path, recs, mode="a")
        have |= set(orphans)
    return have


JSONL_KEYS = ("prompt_id", "prompt", "answer", "splits", "n_prompt_tokens",
              "n_generated_tokens", "hit_max_new_tokens")


def run_batched_capture(runner, spec, cfg, todo, ans_dir, hid_dir, jsonl, done_ids,
                        layout, store_dtype, compress, batch_size, log, log_every) -> dict:
    from src import batched_capture as BC

    points = list(cfg["capture"]["points"])
    if points != ["prompt_last"]:
        raise RuntimeError(f"batched capture only implements prompt_last, config has {points}")
    n_rows = runner.n_blocks + 1
    layers = G.select_layers(n_rows, cfg["capture"].get("layers", "all"))
    max_new = int(runner.gen_kwargs.get("max_new_tokens", 0))
    c = {"n_done": 0, "gen_tokens": 0, "n_truncated": 0, "promoted": 0}
    t0 = time.perf_counter()

    def on_record(i, rec, vec, text):
        row = todo[i]
        pid = int(row.prompt_id)
        n_gen = int(rec["n_generated_tokens"])
        meta = {
            "n_prompt_tokens": int(rec["n_prompt_tokens"]),
            "n_generated_tokens": n_gen,
            "hit_max_new_tokens": bool(rec["truncated"]),
            "n_layer_rows_total": n_rows,
            "layers_saved": layers,
            "hidden_size": runner.hidden_size,
            "n_gen_positions_captured": 0,
            "gen_last_exact": False,
            "final_token_is_eos": rec["finish_reason"] == "eos",
            "missing_gen_points": [],
            "generate_seconds": rec["batch_seconds"],
            "tokens_per_second": (round(n_gen / rec["gen_s"], 2) if rec["gen_s"] else None),
            "rendered_prompt": text,
        }
        r = G.write_prompt(ans_dir, hid_dir, row, rec["answer"], {"prompt_last": vec[layers]},
                           meta, spec, store_dtype, compress, layout)
        if pid not in done_ids:
            jsonl.write(json.dumps({k: r[k] for k in JSONL_KEYS}, ensure_ascii=False) + "\n")
            jsonl.flush()
            done_ids.add(pid)
        c["n_done"] += 1
        c["gen_tokens"] += n_gen
        c["n_truncated"] += int(bool(rec["truncated"]))
        c["promoted"] += len(r["fp16_overflow_promoted"])
        if c["n_done"] % max(1, log_every) == 0 or c["n_done"] == len(todo):
            el = time.perf_counter() - t0
            rate = c["n_done"] / el
            peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            log(f"  [{c['n_done']}/{len(todo)}] {el / c['n_done']:.2f} s/gen  {rate * 60:.1f} "
                f"prompt/min  {c['gen_tokens'] / el:.1f} tok/s  eta "
                f"{(len(todo) - c['n_done']) / rate / 60:.1f} min  peakVRAM {peak:.2f} GB")

    gen = BC.CapturingGenerator(runner, batch_size=batch_size, max_prompt_tokens=2048,
                                log=log, on_record=on_record)
    plan = gen.plan_batches(todo)
    gen.generate(todo, plan)
    c["n_oom_halvings"] = gen.n_oom_halvings
    return c


def run_model(spec, prompts, cfg, args, log_root) -> dict:
    root, ans_dir, hid_dir = G.model_dirs(cfg["output"]["root"], spec.slug)
    ans_dir.mkdir(parents=True, exist_ok=True)
    hid_dir.mkdir(parents=True, exist_ok=True)
    log = Tee(root / "run.log")
    try:
        log(f"\n=== {spec.slug} | {spec.hf_id} | "
            f"{datetime.now(timezone.utc).isoformat()} ===")
        guard_model_dir(root, spec, cfg, args, log)

        layout = cfg["output"].get("hidden_layout", "stacked")
        if args.overwrite and (root / "answers.jsonl").exists():
            (root / "answers.jsonl").unlink()
        done_ids = reconcile_jsonl(root, ans_dir, log)

        todo = [r for r in prompts.itertuples(index=False)
                if args.overwrite
                or not G.already_done(ans_dir, hid_dir, int(r.prompt_id), layout)]
        n_skip = len(prompts) - len(todo)
        log(f"  prompts: {len(prompts)} total, {n_skip} already done, {len(todo)} to run "
            f"({len(done_ids)} lines in answers.jsonl)")
        if not todo:
            log("  nothing to do")
            return {"model": spec.slug, "n_done": 0, "n_skipped": n_skip, "n_failed": 0}

        runner = G.Runner(spec, cfg["generation"], cfg["capture"], log=log)
        log(f"  layer rows 0..{runner.n_blocks} stored; rows 1..{runner.n_blocks - 1} are "
            f"the analysable/steerable ones (row 0 = inputs_embeds, row "
            f"{runner.n_blocks} = post-final-norm)")
        n_points = len(cfg["capture"]["points"])
        est = (runner.n_blocks + 1) * runner.hidden_size * n_points * len(todo) * 2
        log(f"  capture: {n_points} point(s) x {runner.n_blocks + 1} rows x "
            f"{runner.hidden_size} dims -> ~{est / 1e9:.2f} GB; layout={layout} -> "
            f"{len(todo) if layout == 'stacked' else (runner.n_blocks + 1) * n_points * len(todo):,} files")
        batch_size = int(getattr(args, "batch_size", 1) or 1)
        if batch_size > 1:
            log(f"  batch size {batch_size} (OPT-IN, --batch-size): left-padded, length-"
                f"bucketed, halves on OOM. Only valid if scripts/parity_batched_capture.py "
                f"passed for this model.")
        else:
            log(f"  batch size 1 (fact 2: padding perturbs hidden states, and this pass "
                f"exists to produce clean activations)")

        store_dtype = cfg["output"].get("store_dtype", "float16")
        compress = bool(cfg["output"].get("compress", False))
        jsonl = open(root / "answers.jsonl", "a", encoding="utf-8", newline="\n")

        t_start = time.perf_counter()
        n_done = n_failed = promoted_total = gen_tokens = 0
        failures, n_truncated, n_oom_halvings = [], 0, 0
        serial_todo = todo
        try:
            if batch_size > 1:
                bc = run_batched_capture(runner, spec, cfg, todo, ans_dir, hid_dir, jsonl,
                                         done_ids, layout, store_dtype, compress, batch_size,
                                         log, args.log_every)
                n_done, gen_tokens = bc["n_done"], bc["gen_tokens"]
                n_truncated, promoted_total = bc["n_truncated"], bc["promoted"]
                n_oom_halvings = bc["n_oom_halvings"]
                serial_todo = []
            for i, row in enumerate(serial_todo, 1):
                pid = int(row.prompt_id)
                try:
                    answer, vecs, meta = runner.run(row.prompt)
                    rec = G.write_prompt(ans_dir, hid_dir, row, answer, vecs, meta,
                                         spec, store_dtype, compress, layout)
                    if pid not in done_ids:
                        jsonl.write(json.dumps({k: rec[k] for k in JSONL_KEYS},
                                               ensure_ascii=False) + "\n")
                        jsonl.flush()
                        done_ids.add(pid)
                    n_done += 1
                    gen_tokens += meta["n_generated_tokens"]
                    n_truncated += int(bool(meta["hit_max_new_tokens"]))
                    promoted_total += len(rec["fp16_overflow_promoted"])
                    if meta["missing_gen_points"]:
                        log(f"    [{pid}] produced 0 tokens; missing "
                            f"{meta['missing_gen_points']}")
                except torch.cuda.OutOfMemoryError as e:
                    torch.cuda.empty_cache()
                    n_failed += 1
                    failures.append({"prompt_id": pid, "error": "OutOfMemoryError",
                                     "msg": str(e)[:300]})
                    log(f"    [{pid}] OOM -- skipped (lower max_new_tokens or use a "
                        f"device with more memory)")
                except Exception as e:
                    n_failed += 1
                    failures.append({"prompt_id": pid, "error": type(e).__name__,
                                     "msg": str(e)[:300]})
                    log(f"    [{pid}] {type(e).__name__}: {str(e)[:200]}")
                    if args.strict:
                        traceback.print_exc()
                        raise
                if i % max(1, args.log_every) == 0 or i == len(todo):
                    el = time.perf_counter() - t_start
                    rate = i / el
                    peak = (torch.cuda.max_memory_allocated() / 1e9
                            if torch.cuda.is_available() else 0)
                    log(f"  [{i}/{len(todo)}] {el / i:.2f} s/gen  {rate * 60:.1f} "
                        f"prompt/min  {gen_tokens / el:.1f} tok/s  eta "
                        f"{(len(todo) - i) / rate / 60:.1f} min  peakVRAM {peak:.2f} GB  "
                        f"failed {n_failed}")
        finally:
            jsonl.close()
        elapsed = time.perf_counter() - t_start
        s_per_gen = elapsed / max(1, n_done)

        manifest = {
            "model": {"name": spec.name, "hf_id": spec.hf_id, "slug": spec.slug,
                      "dtype": spec.dtype, "device_map": spec.device_map,
                      "attn_implementation": spec.attn_implementation,
                      "system_prompt": spec.system_prompt,
                      "chat_template_kwargs": spec.chat_template_kwargs,
                      "hidden_size": runner.hidden_size,
                      "n_blocks": runner.n_blocks,
                      "n_layer_rows": runner.n_blocks + 1,
                      "steerable_rows": [1, runner.n_blocks - 1]},
            "stage": "generate_label_split.generate",
            "generation": runner.gen_kwargs,
            "capture": cfg["capture"],
            "output": cfg["output"],
            "batch_size": batch_size,
            "n_oom_halvings": n_oom_halvings,
            "prompts": {"n_total": int(len(prompts)), "n_run": n_done,
                        "n_skipped_existing": n_skip, "n_failed": n_failed,
                        "n_hit_max_new_tokens": n_truncated},
            "totals": {"generated_tokens": gen_tokens,
                       "wall_seconds": round(elapsed, 1),
                       "seconds_per_generation": round(s_per_gen, 3),
                       "tokens_per_second": round(gen_tokens / elapsed, 2) if elapsed else None,
                       "arrays_promoted_to_fp32": promoted_total},
            "failures": failures,
            "env": env_info(),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
        }
        with open(root / "manifest.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        log(f"  done: {n_done} ok, {n_failed} failed, {n_skip} skipped in "
            f"{elapsed / 60:.1f} min")
        log(f"  MEASURED {s_per_gen:.2f} s/gen at batch {batch_size}; {n_truncated} answer(s) hit "
            f"max_new_tokens={runner.gen_kwargs.get('max_new_tokens')}")
        log(f"  -> {root}")

        del runner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        return {"model": spec.slug, "n_done": n_done, "n_skipped": n_skip,
                "n_failed": n_failed, "seconds_per_generation": round(s_per_gen, 3)}
    finally:
        log.close()


def stage_generate(cfg, specs, prompts, args, log) -> dict:
    log("\n=== stage: generate ===")
    out_root = G.repo_path(cfg["output"]["root"])
    out_root.mkdir(parents=True, exist_ok=True)

    ptbl = out_root / "prompts.parquet"
    if args.limit or args.smoke:
        log(f"  not rewriting {ptbl.name}: this is a --limit/--smoke run and the "
            f"{len(prompts)}-row table would replace the full prompt table other "
            f"scripts read.")
    else:
        prompts.rename(columns={"cstar_split": "splits"}).to_parquet(ptbl, index=False)

    rows = prompts.rename(columns={"cstar_split": "splits"})

    summary = []
    for spec in specs:
        try:
            summary.append(run_model(spec, rows, cfg, args, log))
        except Exception as e:
            log(f"!! {spec.slug} aborted: {type(e).__name__}: {e}")
            traceback.print_exc()
            summary.append({"model": spec.slug, "error": f"{type(e).__name__}: {e}"})
            if args.strict:
                raise
    log("\nsummary")
    for s in summary:
        log("  " + json.dumps(s))
    ok = all("error" not in s for s in summary)
    return {"ok": ok, "generate": summary}


class Classifier:

    def __init__(self, predict_mod, repo: Path):
        self.P = predict_mod
        self.D = predict_mod.D
        self.M = predict_mod.M
        self.T = predict_mod.T
        self.repo = repo


def import_classifier(clf_repo: Path, log) -> Classifier:
    clf_repo = Path(clf_repo).resolve()
    predict_py = clf_repo / "scripts" / "predict.py"
    if not predict_py.exists():
        raise FileNotFoundError(f"classifier predict.py not found at {predict_py}")

    saved_modules = {k: v for k, v in sys.modules.items()
                     if k == "src" or k.startswith("src.")}
    saved_path = list(sys.path)
    for k in saved_modules:
        del sys.modules[k]
    sys.path.insert(0, str(clf_repo))
    try:
        spec = importlib.util.spec_from_file_location("cstar_predict", predict_py)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["cstar_predict"] = mod
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = saved_path
        for k in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
            del sys.modules[k]
        sys.modules.update(saved_modules)

    log(f"  imported {predict_py} as 'cstar_predict' "
        f"(MIN_VAL_AGREEMENT={mod.MIN_VAL_AGREEMENT}, LABELS3={mod.M.LABELS3}, "
        f"MIN_N={mod.M.MIN_N})")
    return Classifier(mod, clf_repo)


def val_parity_gate(clf: Classifier, run_dir: Path, cfg_c, tok, model, device, amp,
                    token_budget: int, log) -> dict:
    val = pd.read_parquet(clf.D.repo_path("data/processed/val.parquet"))
    pv, _ = clf.P.predict(val["response_text"].tolist(), cfg_c, tok, model, device, amp,
                          token_budget)
    saved = np.load(run_dir / "val_probs.npy")[:len(val)]
    agree = float((pv.argmax(1) == saved.argmax(1)).mean())
    parity = {"n": int(len(val)), "pred_agreement": agree,
              "max_abs_prob_diff": float(np.abs(pv - saved).max()),
              "macro_f1_repro": clf.M.classification_metrics(
                  val["label_id"], pv.argmax(1), pv)["macro_f1"],
              "min_val_agreement": float(clf.P.MIN_VAL_AGREEMENT)}
    log(f"  val parity: agreement={agree:.4f} "
        f"max|dp|={parity['max_abs_prob_diff']:.4f} "
        f"macroF1={parity['macro_f1_repro']:.4f} (gate >= {clf.P.MIN_VAL_AGREEMENT})")
    if agree < clf.P.MIN_VAL_AGREEMENT:
        raise RuntimeError(
            f"val-parity gate FAILED: re-predicted val agrees with the run's saved "
            f"val_probs.npy on only {agree:.2%} of rows (gate "
            f"{clf.P.MIN_VAL_AGREEMENT:.0%}). The checkpoint or tokenisation was not "
            f"restored correctly; every label this stage would write comes from a "
            f"different classifier than C*.")
    return parity


def label_mix_table(df: pd.DataFrame, labels3, min_n: int) -> dict:
    ct = pd.crosstab(df["gen_model"], df[f"{CSTAR_CANDIDATE}_label"]).reindex(
        columns=list(labels3), fill_value=0)
    return {str(m): {"n": int(row.sum()),
                     **{lab: {"n": int(row[lab]),
                              "pct": round(100 * row[lab] / max(1, row.sum()), 1),
                              "evidence": "ok" if row[lab] >= min_n
                                          else "insufficient_evidence"}
                        for lab in labels3}}
            for m, row in ct.iterrows()}


def stage_label(cfg, specs, prompts, args, study_dir, log) -> dict:
    log("\n=== stage: label ===")
    slugs = [s.slug for s in specs]
    gen_root = G.repo_path(cfg["output"].get("generated_root", cfg["output"]["root"]))
    run_dir = G.repo_path(cfg["study"]["classifier_run_dir"])
    clf_repo = G.repo_path(cfg["study"]["classifier_repo"])
    dest = study_dir / "labels.parquet"
    meta_dest = study_dir / "labels.json"

    clf = import_classifier(clf_repo, log)
    labels3 = list(clf.M.LABELS3)
    if list(cfg["study"]["labels3"]) != labels3:
        raise RuntimeError(
            f"study.yaml labels3={cfg['study']['labels3']} but the classifier's "
            f"M.LABELS3={labels3}. The label order IS the probability column order; a "
            f"mismatch silently relabels every row.")

    gen = clf.P.load_generated(gen_root, args.limit)
    extra = sorted(set(gen.gen_model.unique()) - set(slugs))
    if extra:
        log(f"  ignoring {len(extra)} non-roster model dir(s) under {gen_root}: {extra}")
    gen = gen[gen.gen_model.isin(slugs)].copy()
    missing = [s for s in slugs if s not in set(gen.gen_model.unique())]
    if missing:
        raise RuntimeError(
            f"no generated answers for roster model(s) {missing} under {gen_root}; "
            f"run --stage generate first")
    gen = gen.sort_values(["gen_model", "prompt_id"], kind="stable").reset_index(drop=True)

    dup = gen.duplicated(subset=["gen_model", "prompt_id"])
    if dup.any():
        raise RuntimeError(
            f"{int(dup.sum())} duplicated (gen_model, prompt_id) row(s) in the generated "
            f"answers; one answers.jsonl holds two runs. Re-run --stage generate "
            f"--overwrite for the affected model.")

    cstar = prompts.set_index("prompt_id")["cstar_split"]
    unknown = sorted(set(gen.prompt_id) - set(cstar.index))
    if unknown:
        raise RuntimeError(
            f"{len(unknown)} generated prompt_id(s) are not in the study prompt table "
            f"(e.g. {unknown[:5]}); the generated tree and data/processed disagree")

    log(f"  {len(gen)} answers over {gen.prompt_id.nunique()} prompts x "
        f"{len(slugs)} models  {gen.gen_model.value_counts().to_dict()}")
    log(f"  empty answers: {int(gen['empty_answer'].sum())}  "
        f"hit_max_new_tokens: {int(gen['hit_max_new_tokens'].sum())}")

    if args.dry_run:
        n_shards = -(-len(gen) // args.shard_rows)
        log(f"  would score {len(gen)} answers in {n_shards} shard(s) of "
            f"{args.shard_rows} with C* = {run_dir.name}")
        return {"ok": True}

    shard_dir = study_dir / "label_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    inputs = {"n_rows": int(len(gen)), "shard_rows": int(args.shard_rows),
              "run_dir": run_dir.name, "models": slugs,
              "rows_sha256": hashlib.sha256(
                  "\n".join(f"{m},{p}" for m, p in
                            zip(gen.gen_model, gen.prompt_id)).encode()).hexdigest()}
    plan_path = shard_dir / "plan.json"
    if plan_path.exists():
        old = json.loads(plan_path.read_text(encoding="utf-8"))
        if old != inputs:
            raise RuntimeError(
                f"{shard_dir} holds shards for a different input set (rows or shard size "
                f"changed since they were written). Delete that directory and re-run; "
                f"resuming across a changed row order would misalign every shard.")
    else:
        plan_path.write_text(json.dumps(inputs, indent=2), encoding="utf-8")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_c, tok, model, amp = clf.P.load_classifier(run_dir, device, log)
    if cfg_c["candidate"] != CSTAR_CANDIDATE:
        raise RuntimeError(
            f"C* is the Longformer run, but "
            f"{run_dir} is candidate={cfg_c['candidate']!r}. "
            f"Check study.classifier_run_dir.")

    parity = val_parity_gate(clf, run_dir, cfg_c, tok, model, device, amp,
                             args.token_budget, log)

    n_shards = -(-len(gen) // args.shard_rows)
    t0, n_trunc_total = time.perf_counter(), 0
    for si in range(n_shards):
        out_path = shard_dir / f"shard_{si:04d}.parquet"
        if out_path.exists() and not args.overwrite:
            log(f"  shard {si + 1}/{n_shards}: already scored, skipping")
            continue
        chunk = gen.iloc[si * args.shard_rows:(si + 1) * args.shard_rows]
        ts = time.perf_counter()
        probs, n_trunc = clf.P.predict(chunk["answer"].tolist(), cfg_c, tok, model,
                                       device, amp, args.token_budget)
        n_trunc_total += n_trunc
        out = pd.concat([
            pd.DataFrame({
                "prompt_id": chunk["prompt_id"].to_numpy(),
                "gen_model": chunk["gen_model"].to_numpy(),
                "answer_sha1": [G.sha1(a) for a in chunk["answer"]],
                "n_generated_tokens": chunk["n_generated_tokens"].to_numpy(),
                "truncated": chunk["hit_max_new_tokens"].to_numpy().astype(bool),
                "empty_answer": chunk["empty_answer"].to_numpy().astype(bool),
            }).reset_index(drop=True),
            clf.P.prob_columns(CSTAR_CANDIDATE, probs).reset_index(drop=True),
        ], axis=1)
        out["cstar_split"] = cstar.reindex(out["prompt_id"]).to_numpy()
        out.to_parquet(out_path, index=False)
        log(f"  shard {si + 1}/{n_shards}: {len(chunk)} rows in "
            f"{time.perf_counter() - ts:.1f}s  truncated_at_max_length={n_trunc}  "
            f"-> {out_path.name}")

    frames = [pd.read_parquet(p) for p in sorted(shard_dir.glob("shard_*.parquet"))]
    lab = pd.concat(frames, ignore_index=True)
    cols = ["prompt_id", "gen_model", "answer_sha1",
            f"{CSTAR_CANDIDATE}_label",
            *[f"{CSTAR_CANDIDATE}_p_{c.lower()}" for c in labels3],
            f"{CSTAR_CANDIDATE}_confidence",
            "n_generated_tokens", "truncated", "empty_answer", "cstar_split"]
    lab = lab[cols].sort_values(["gen_model", "prompt_id"],
                               kind="stable").reset_index(drop=True)
    if lab["cstar_split"].isna().any():
        raise RuntimeError("cstar_split is null for some rows; the prompt table and the "
                           "generated answers disagree")
    lab.to_parquet(dest, index=False)

    min_n = int(clf.M.MIN_N)
    mix = label_mix_table(lab, labels3, min_n)
    clean = lab[lab.cstar_split == "test"]
    mix_clean = label_mix_table(clean, labels3, min_n) if len(clean) else {}

    log(f"\n  label mix by generation model (C* = {CSTAR_CANDIDATE}, argmax label)")
    log("  " + pd.crosstab(lab["gen_model"], lab[f"{CSTAR_CANDIDATE}_label"])
        .reindex(columns=labels3, fill_value=0).to_string().replace("\n", "\n  "))
    log(f"\n  same, restricted to the C*-test-clean subset (cstar_split == 'test', "
        f"n={len(clean)}): 1,223 of the 1,750 prompts are in C*'s OWN training split, "
        f"so every headline is reported twice")
    if len(clean):
        log("  " + pd.crosstab(clean["gen_model"], clean[f"{CSTAR_CANDIDATE}_label"])
            .reindex(columns=labels3, fill_value=0).to_string().replace("\n", "\n  "))
    thin = [(m, lab_) for m, d in mix.items() for lab_ in labels3
            if d[lab_]["evidence"] != "ok"]
    if thin:
        log(f"  !! {len(thin)} (model, class) cell(s) below MIN_N={min_n}: {thin}. "
            f"The Direction split gets only 40% of these, so the balanced "
            f"difference-of-means in directions_steering.py will be estimated from very few rows.")

    payload = {
        "stage": "label",
        "seed": int(cfg["study"]["seed"]),
        "classifier": {"name": CSTAR_CANDIDATE, "run_dir": str(run_dir),
                       "model": cfg_c["model"]["name"],
                       "max_length": cfg_c["model"]["max_length"],
                       "truncation": cfg_c["train"].get("truncation", "head"),
                       "labels3": labels3, "min_n": min_n},
        "val_parity": parity,
        "inputs": {"generated_root": str(gen_root), "models": slugs,
                   "n_rows": int(len(lab)),
                   "n_prompts": int(lab.prompt_id.nunique()),
                   "rows_sha256": inputs["rows_sha256"],
                   "shard_rows": int(args.shard_rows),
                   "token_budget": int(args.token_budget)},
        "counts": {"n_truncated_at_classifier_max_length": int(n_trunc_total),
                   "n_hit_max_new_tokens": int(lab["truncated"].sum()),
                   "n_empty_answers": int(lab["empty_answer"].sum())},
        "label_mix": mix,
        "label_mix_cstar_test_clean": mix_clean,
        "labels_sha256": hashlib.sha256("\n".join(
            f"{m},{p},{s},{l}" for m, p, s, l in zip(
                lab.gen_model, lab.prompt_id, lab.answer_sha1,
                lab[f"{CSTAR_CANDIDATE}_label"])).encode()).hexdigest(),
        "seconds": round(time.perf_counter() - t0, 1),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    commit_json(meta_dest, {k: v for k, v in payload.items()
                            if k not in ("seconds", "generated_utc")}, log)
    log(f"  wrote {dest}  ({len(lab)} rows)")
    return {"ok": True, "labels": lab}


def verify_label(study_dir, log) -> dict:
    dest, meta_dest = study_dir / "labels.parquet", study_dir / "labels.json"
    if not (dest.exists() and meta_dest.exists()):
        log(f"  [VERIFY] {dest} / {meta_dest} missing")
        return {"ok": False}
    lab = pd.read_parquet(dest)
    old = json.loads(meta_dest.read_text(encoding="utf-8"))
    got = hashlib.sha256("\n".join(
        f"{m},{p},{s},{l}" for m, p, s, l in zip(
            lab.gen_model, lab.prompt_id, lab.answer_sha1,
            lab[f"{CSTAR_CANDIDATE}_label"])).encode()).hexdigest()
    ok = old.get("labels_sha256") == got
    log(f"  [VERIFY] labels.parquet: committed {str(old.get('labels_sha256'))[:16]}  "
        f"recomputed {got[:16]}  {'MATCH' if ok else 'MISMATCH'}")
    return {"ok": ok}


def allocate(n: int, fracs: dict, rng: np.random.Generator) -> dict:
    names = list(fracs)
    exact = np.array([n * fracs[k] for k in names])
    base = np.floor(exact).astype(int)
    rem = n - base.sum()
    if rem:
        order = np.lexsort((rng.random(len(names)), -(exact - base)))
        base[order[:rem]] += 1
    return dict(zip(names, base.tolist()))


def assign_splits(strata: pd.Series, fracs: dict, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    assign = pd.Series(index=strata.index, dtype=object)
    for _, grp in strata.groupby(strata, sort=True):
        ids = rng.permutation(grp.index.to_numpy())
        counts = allocate(len(ids), fracs, rng)
        i = 0
        for name in fracs:
            assign.loc[ids[i:i + counts[name]]] = name
            i += counts[name]
    assign = assign.sort_index()
    if assign.isna().any():
        raise RuntimeError(f"{int(assign.isna().sum())} prompts unassigned")
    return assign


def split_digest(assign: pd.Series) -> str:
    return hashlib.sha256(
        "\n".join(f"{p},{s}" for p, s in assign.sort_index().items()).encode()).hexdigest()


def per_model_counts(assign: pd.Series, wide: pd.DataFrame, labels3, fracs) -> dict:
    out = {}
    for m in wide.columns:
        lab = wide[m].reindex(assign.index)
        ct = pd.crosstab(assign, lab).reindex(index=list(fracs), columns=list(labels3),
                                              fill_value=0)
        out[str(m)] = {s: {c: int(ct.loc[s, c]) for c in labels3} for s in fracs}
    return out


def duplicate_text_report(prompts: pd.DataFrame, assign: pd.Series, log) -> dict:
    df = prompts[["prompt_id", "prompt_sha1"]].copy()
    df["split"] = assign.reindex(df.prompt_id).to_numpy()
    g = df.groupby("prompt_sha1")["split"].agg(["nunique", "size"])
    straddling = g[(g["size"] > 1) & (g["nunique"] > 1)]
    dup_clusters = g[g["size"] > 1]
    rep = {"n_prompt_ids": int(len(df)),
           "n_distinct_texts": int(df.prompt_sha1.nunique()),
           "n_duplicate_text_clusters": int(len(dup_clusters)),
           "n_clusters_straddling_splits": int(len(straddling)),
           "n_prompt_ids_in_straddling_clusters": int(straddling["size"].sum()),
           "decision": "kept by user decision; reported, not merged"}
    log(f"  duplicate prompt TEXT: {rep['n_prompt_ids']} ids -> "
        f"{rep['n_distinct_texts']} distinct texts; "
        f"{rep['n_duplicate_text_clusters']} duplicate clusters, "
        f"{rep['n_clusters_straddling_splits']} of which straddle two or more splits "
        f"({rep['n_prompt_ids_in_straddling_clusters']} prompt_ids). "
        f"WARNING only -- kept by user decision.")
    return rep


def build_partition(name: str, strata: pd.Series, wide: pd.DataFrame,
                    prompts: pd.DataFrame, cstar: pd.Series, cfg, args,
                    stratified_on, labels3, log) -> dict:
    fracs = S.split_fractions(cfg)
    assign = assign_splits(strata, fracs, args.seed)
    digest = split_digest(assign)

    sizes = assign.value_counts().reindex(list(fracs)).to_dict()
    log(f"\n  partition '{name}': n={len(assign)}  sha256={digest[:16]}...")
    log(f"  split sizes {sizes}  "
        f"actual fractions "
        f"{ {k: round(v / len(assign), 4) for k, v in sizes.items()} }")

    log("\n  C* split x study split (the circularity control)")
    ct_cstar = pd.crosstab(cstar.reindex(assign.index), assign).reindex(
        columns=list(fracs), fill_value=0)
    log("  " + ct_cstar.to_string().replace("\n", "\n  "))

    pmc = per_model_counts(assign, wide, labels3, fracs)
    log("\n  PER-MODEL class counts by study split (each model's OWN C* label).")
    log("  These DIFFER between models by construction: one shared prompt partition is "
        "labelled once per generation model, so the class balance inside a "
        "split is a property of that model's responses, not of the split.")
    for m, d in pmc.items():
        log(f"\n    {m}")
        log("    " + pd.DataFrame(d).T.reindex(index=list(fracs), columns=list(labels3))
            .to_string().replace("\n", "\n    "))

    min_n = int(cfg["study"]["min_n"])
    thin = [(m, s, c) for m, d in pmc.items() for s, row in d.items()
            for c in labels3 if row[c] < min_n]
    if thin:
        log(f"\n  !! {len(thin)} (model, split, class) cell(s) below min_n={min_n}: "
            f"{thin[:12]}{' ...' if len(thin) > 12 else ''}")
        log(f"     The balanced difference-of-means subsamples NORMAL down to "
            f"min(n_R, n_D) inside the Direction split, so a thin R or D cell there "
            f"caps the estimate for BOTH directions.")

    strat_tbl = (strata.value_counts().sort_index().rename("n").reset_index()
                 .rename(columns={"index": "stratum"}))
    strat_tbl.columns = ["stratum", "n"]
    per_stratum = pd.crosstab(strata.reindex(assign.index), assign).reindex(
        columns=list(fracs), fill_value=0)

    payload = {
        "stage": "split",
        "partition": name,
        "seed": int(args.seed),
        "n_prompts": int(len(assign)),
        "fractions": fracs,
        "actual_fractions": {k: round(v / len(assign), 6) for k, v in sizes.items()},
        "split_sizes": {k: int(v) for k, v in sizes.items()},
        "stratified_on": stratified_on,
        "note_per_model_counts":
            "Per-model class counts inside each split DIFFER between models and that is "
            "expected: one shared prompt partition is labelled once per "
            "generation model, so class balance inside a split is a property of that "
            "model's responses. The label triple is used for STRATIFICATION ONLY; class "
            "means inside the Direction split always use each model's OWN C* label.",
        "note_cstar_split":
            "cstar_split is C*'s own train/val/test membership, NOT a study split. "
            "1,223 of the 1,750 prompts are in C*'s training split, which is why it is a "
            "stratification variable and why every headline is also reported on the "
            "cstar_split == 'test' subset.",
        "labels3": list(labels3),
        "min_n": min_n,
        "cells_below_min_n": [list(t) for t in thin],
        "n_strata": int(strata.nunique()),
        "stratum_table": [{"stratum": str(r.stratum), "n": int(r.n),
                           **{k: int(per_stratum.loc[r.stratum, k]) for k in fracs}}
                          for r in strat_tbl.itertuples(index=False)],
        "cstar_split_by_study_split": {
            str(c): {str(s): int(ct_cstar.loc[c, s]) for s in fracs}
            for c in ct_cstar.index},
        "per_model_class_counts": pmc,
        "duplicate_text": duplicate_text_report(prompts, assign, log),
        "splits": {k: sorted(int(p) for p in assign[assign == k].index) for k in fracs},
        "split_sha256": digest,
    }
    return {"payload": payload, "assign": assign, "strata": strata, "digest": digest}


def stage_split(cfg, specs, prompts, args, study_dir, log) -> dict:
    log("\n=== stage: split ===")
    slugs = [s.slug for s in specs]
    lab_path = study_dir / "labels.parquet"
    if not lab_path.exists():
        raise FileNotFoundError(f"{lab_path} not found; run --stage label first")
    lab = pd.read_parquet(lab_path)
    lab = lab[lab.gen_model.isin(slugs)]
    labels3 = list(cfg["study"]["labels3"])

    wide = lab.pivot(index="prompt_id", columns="gen_model",
                     values=f"{CSTAR_CANDIDATE}_label")
    missing_cols = [s for s in slugs if s not in wide.columns]
    if missing_cols:
        raise RuntimeError(f"labels.parquet has no rows for {missing_cols}")
    wide = wide[slugs]
    if wide.isna().any().any():
        n = int(wide.isna().any(axis=1).sum())
        raise RuntimeError(
            f"{n} prompt(s) lack a C* label from at least one roster model; the label "
            f"triple is undefined for them. Re-run --stage generate/--stage label.")

    prompts = prompts[prompts.prompt_id.isin(wide.index)].reset_index(drop=True)
    cstar = prompts.set_index("prompt_id")["cstar_split"].reindex(wide.index)
    log(f"  {len(wide)} prompts x {len(slugs)} models; label triple over {labels3}")

    results, ok = {}, True
    if args.partition == "shared":
        strata = pd.Series(
            ["|".join(wide.loc[p, m] for m in slugs) + "|" + str(cstar.loc[p])
             for p in wide.index], index=wide.index, name="stratum")
        stratified_on = [f"label_triple({', '.join(slugs)})",
                         "cstar_split (C*'s own train/val/test)"]
        r = build_partition("shared", strata, wide, prompts, cstar, cfg, args,
                            stratified_on, labels3, log)
        dest = study_dir / "study_splits.json"
        if args.verify:
            ok = verify_split(dest, r, log)
        else:
            warn_superseded(dest, stratified_on, log)
            commit_split(dest, study_dir / "study_splits.parquet", r, wide, cstar,
                         slugs, log)
        results["shared"] = r["digest"]
    else:
        for m in slugs:
            strata = pd.Series([f"{wide.loc[p, m]}|{cstar.loc[p]}" for p in wide.index],
                               index=wide.index, name="stratum")
            stratified_on = [f"label({m})", "cstar_split (C*'s own train/val/test)"]
            r = build_partition(m, strata, wide, prompts, cstar, cfg, args,
                                stratified_on, labels3, log)
            r["payload"]["note_partition_mode"] = (
                "per-model partition: prompts are assigned differently for each model, "
                "so no cross-model comparison on a shared prompt set is possible.")
            dest = study_dir / f"study_splits_{m}.json"
            if args.verify:
                ok = verify_split(dest, r, log) and ok
            else:
                warn_superseded(dest, stratified_on, log)
                commit_split(dest, study_dir / f"study_splits_{m}.parquet", r,
                             wide[[m]], cstar, [m], log)
            results[m] = r["digest"]
    return {"ok": ok, "split_sha256": results}


def warn_superseded(dest: Path, stratified_on, log) -> None:
    if not dest.exists():
        return
    old = json.loads(dest.read_text(encoding="utf-8"))
    if old.get("stratified_on") != stratified_on:
        log(f"  !! OVERWRITING {dest.name}, which was produced with a DIFFERENT "
            f"stratification:\n     on disk: {old.get('stratified_on')}\n"
            f"     now:     {stratified_on}\n"
            f"     (Anything already computed against the old sha256 "
            f"{str(old.get('sha256', old.get('split_sha256')))[:16]} must be recomputed.)")


def commit_split(dest: Path, parquet_dest: Path, r: dict, wide: pd.DataFrame,
                 cstar: pd.Series, slugs, log) -> None:
    commit_json(dest, r["payload"], log)
    assign = r["assign"]
    flat = pd.DataFrame({"prompt_id": assign.index.astype("int64"),
                         "study_split": assign.to_numpy(),
                         "stratum": r["strata"].reindex(assign.index).to_numpy(),
                         "cstar_split": cstar.reindex(assign.index).to_numpy()})
    for m in slugs:
        flat[f"label_{m}"] = wide[m].reindex(assign.index).to_numpy()
    flat.to_parquet(parquet_dest, index=False)
    log(f"  wrote {parquet_dest}")


def verify_split(dest: Path, r: dict, log) -> bool:
    if not dest.exists():
        log(f"  [VERIFY] {dest} does not exist")
        return False
    old = json.loads(dest.read_text(encoding="utf-8"))
    ok = old.get("split_sha256") == r["digest"]
    log(f"  [VERIFY] {dest.name}: committed "
        f"{str(old.get('split_sha256'))[:16]}  recomputed {r['digest'][:16]}  "
        f"{'MATCH' if ok else 'MISMATCH'}")
    if old.get("stratified_on") != r["payload"]["stratified_on"]:
        log(f"           committed stratified_on={old.get('stratified_on')} differs from "
            f"{r['payload']['stratified_on']}")
        ok = False
    if not verify_json(dest, r["payload"], log):
        ok = False
    return ok


def print_plan(cfg, specs, prompts, args, log) -> None:
    n_p, n_m = len(prompts), len(specs)
    total = n_p * n_m
    layout = cfg["output"].get("hidden_layout", "stacked")
    mnt = cfg["generation"].get("max_new_tokens")
    log("\n=== dry run: the plan ===")
    log(f"  prompts                  {n_p}")
    log(f"  models                   {n_m}  {[s.slug for s in specs]}")
    log(f"  generations (generate_label_split)     {total:,}  = {n_p} x {n_m}, batch 1, greedy, "
        f"max_new_tokens={mnt}")
    log(f"  hidden layout            {layout}  "
        f"({total if layout == 'stacked' else 'n_rows x ' + str(total)} files)")
    for s in specs:
        root, ans_dir, hid_dir = G.model_dirs(cfg["output"]["root"], s.slug)
        done = sum(1 for r in prompts.itertuples(index=False)
                   if G.already_done(ans_dir, hid_dir, int(r.prompt_id), layout))
        log(f"    {s.slug:24s} {done:5d} done / {n_p - done:5d} to run "
            f"-> {root}")
    fr = S.split_fractions(cfg)
    log(f"  study split              " +
        "  ".join(f"{k} {fr[k]:.0%} (~{int(round(n_p * fr[k]))})" for k in SPLIT_ORDER))
    log(f"  labelling                {total:,} answers through C* "
        f"({Path(cfg['study']['classifier_run_dir']).name}) in shards of "
        f"{args.shard_rows}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="generate_label_split: generate, label with C*, and split the study prompts.")
    ap.add_argument("--config", default="configs/study.yaml")
    ap.add_argument("--stage", default="all", choices=STAGES,
                    help="preflight | generate | label | split | all")
    ap.add_argument("--models", default=None,
                    help="comma-separated slugs/hf_ids; default = the whole roster")
    ap.add_argument("--partition", default=None, choices=["shared", "per-model"],
                    help="split mode; default comes from study.partition (shared)")
    ap.add_argument("--seed", type=int, default=None,
                    help="default comes from study.seed (42)")
    ap.add_argument("--limit", type=int, default=None,
                    help="first N prompts (after sort) -- smoke runs only")
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--shard-rows", type=int, default=None,
                    help="rows per scoring shard; default from "
                         "steering.scoring.shard_rows (20000)")
    ap.add_argument("--token-budget", type=int, default=16384,
                    help="max padded tokens per classifier batch")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="generate stage only. 1 (default) is the original one-prompt-per-"
                         "call path. >1 enables batched capture (src/batched_capture.py); "
                         "use it only for models that passed parity_batched_capture.py")
    ap.add_argument("--overwrite", action="store_true",
                    help="regenerate/rescore everything instead of resuming")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny model, few prompts, short generations, from config.smoke")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan (counts) and exit")
    ap.add_argument("--verify", action="store_true",
                    help="recompute every committed hash and exit non-zero on drift")
    ap.add_argument("--strict", action="store_true",
                    help="abort on the first per-prompt failure")
    ap.add_argument("--log-every", type=int, default=25)
    args = ap.parse_args()

    cfg = G.load_config(G.repo_path(args.config))
    S.assert_locked_steering_config(cfg)
    if args.smoke:
        sm = cfg.get("smoke") or {}
        cfg["models"] = sm.get("models", cfg["models"])
        cfg["generation"] = {**cfg["generation"], **(sm.get("generation") or {})}
        S.apply_smoke_paths(cfg)
        if args.limit is None:
            args.limit = int(sm.get("n_prompts", 8))
    if args.max_new_tokens is not None:
        cfg["generation"]["max_new_tokens"] = args.max_new_tokens
    if args.seed is None:
        args.seed = int(cfg["study"]["seed"])
    if args.partition is None:
        args.partition = str(cfg["study"].get("partition", "shared"))
    if args.shard_rows is None:
        args.shard_rows = int(cfg["steering"]["scoring"].get("shard_rows", 20000))

    if cfg["generation"].get("do_sample"):
        raise SystemExit("study.yaml sets generation.do_sample: true; the study is "
                         "greedy-only and every error bar assumes it")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    study_dir = G.repo_path(cfg["output"].get("study_root", "data/study"))
    study_dir.mkdir(parents=True, exist_ok=True)
    log = Tee(study_dir / f"generate_label_split_{args.stage}.log")

    try:
        log(f"\n{'=' * 78}")
        log(f"generate_label_split  stage={args.stage}  "
            f"{datetime.now(timezone.utc).isoformat()}")
        log(f"config      {args.config}")
        log(f"seed        {args.seed}   partition={args.partition}   "
            f"shard_rows={args.shard_rows}")
        log(f"study dir   {study_dir}")
        for k, v in env_info().items():
            log(f"  {k:16s} {v}")

        specs = G.parse_models(cfg, args.models.split(",") if args.models else None)
        log(f"models      {[s.slug for s in specs]}")

        prompts = prompt_table(cfg, args, log)

        if args.dry_run:
            print_plan(cfg, specs, prompts, args, log)
            sys.exit(0)

        want = [args.stage] if args.stage != "all" else list(STAGES[:-1])
        ok = True
        for st in want:
            if st == "preflight":
                ok = stage_preflight(cfg, specs, prompts, args, study_dir, log)["ok"] and ok
            elif st == "generate":
                if args.verify:
                    log("\n=== stage: generate === (--verify: nothing to recompute; the "
                        "per-model manifest.json is the committed artifact)")
                    continue
                ok = stage_generate(cfg, specs, prompts, args, log)["ok"] and ok
            elif st == "label":
                if args.verify:
                    ok = verify_label(study_dir, log)["ok"] and ok
                else:
                    ok = stage_label(cfg, specs, prompts, args, study_dir, log)["ok"] and ok
            elif st == "split":
                ok = stage_split(cfg, specs, prompts, args, study_dir, log)["ok"] and ok
            if not ok and args.stage == "all":
                log(f"\n!! stage {st} failed; not continuing")
                break

        log(f"\n{'OK' if ok else 'FAILED'}  stage={args.stage}")
        sys.exit(0 if ok else 1)
    finally:
        log.close()


if __name__ == "__main__":
    main()
