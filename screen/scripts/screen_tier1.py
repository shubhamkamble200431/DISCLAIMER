"""Tier 1 of the model screen (plan_model_screening.md §2) -- the cheap gate.

Per candidate, in order, stopping at the first failure for THAT model but never aborting
the whole run:

  1. preflight   does transformers recognise the architecture? (config.json only, no weights)
  2. load        AutoModelForCausalLM, with the AutoModelForImageTextToText fallback
  3. hook target dec = model.get_decoder(); assert len(dec.layers) == num_hidden_layers
  4. LAYER MAP   >>> the important one <<< empirically confirm that saved row l is the
                 output of dec.layers[l-1], by hooking that module and comparing its output
                 against hidden_states[l]. The whole steering design rests on this
                 off-by-one, and it has only ever been verified for Qwen3.5 and Gemma-4.
                 A new architecture that indexes differently would silently shift every
                 layer result by one.
  5. no-op hook  register a hook that adds exactly 0.0 and check the logits are unchanged,
                 proving we can write to the residual stream without corrupting the forward
                 pass (tuple-vs-tensor return, in-place aliasing, etc.)
  6. template    render one prompt and dump the exact tail -- this IS the prompt_last capture
                 position, and it differs per family (Qwen ends after an empty </think>,
                 Gemma on '<|turn|>model\\n'). No chat template at all is disqualifying.
  7. norms       n prompts, forward pass only (no generation), median ||h|| per layer,
                 plus the row-0 degeneracy check and peak VRAM.

Forward-pass only: Tier 1 never generates, so it costs seconds per model once weights are
local. Twenty prompts is enough -- a 12-prompt sample already matched a 600-prompt one to
within 1% at every layer on all three incumbents.

Usage:
  python -X utf8 scripts/screen_tier1.py --config configs/screen.yaml --preflight
  python -X utf8 scripts/screen_tier1.py --config configs/screen.yaml --smoke
  python -X utf8 scripts/screen_tier1.py --config configs/screen.yaml
  python -X utf8 scripts/screen_tier1.py --config configs/screen.yaml --models olmo2-1b,falcon3-3b
"""
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
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import generate as G    # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "screen" / "tier1"


class Tee:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def env_info():
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available()}
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except Exception:
        pass
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    return info


def hidden_of(out):
    """A decoder layer returns either a tensor or a tuple whose [0] is the hidden state."""
    return out[0] if isinstance(out, (tuple, list)) else out


def resolve_decoder(model, log):
    """model.get_decoder() walks language_model / text_model / decoder; fall back by hand."""
    dec = None
    try:
        dec = model.get_decoder()
    except Exception as e:
        log(f"      get_decoder() raised {type(e).__name__}: {e}")
    if dec is None or not hasattr(dec, "layers"):
        for path in ("model.language_model", "model.model", "model", "transformer"):
            obj = model
            try:
                for part in path.split("."):
                    obj = getattr(obj, part)
                if hasattr(obj, "layers"):
                    log(f"      get_decoder() gave no .layers; fell back to model.{path}")
                    dec = obj
                    break
            except AttributeError:
                continue
    return dec


def screen_one(spec, prompts, cfg, log):
    rec = {"slug": spec.slug, "hf_id": spec.hf_id, "note": spec.note,
           "stage_reached": "preflight", "ok": False}
    t0 = time.perf_counter()

    # -- 1. preflight ------------------------------------------------------- #
    pf = G.preflight(spec)
    rec["preflight"] = pf
    log(f"    preflight: type={pf['model_type']} supported={pf['supported']} "
        f"H={pf['hidden_size']} blocks={pf['num_hidden_layers']} arch={pf['architectures']}")
    if not pf["supported"]:
        rec["error"] = (f"model_type '{pf['model_type']}' unknown to transformers "
                        f"{env_info().get('transformers')}; needs a newer transformers "
                        f"or trust_remote_code: true")
        log(f"    [FAIL] {rec['error']}")
        return rec

    # -- 2. load ------------------------------------------------------------ #
    rec["stage_reached"] = "load"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(spec.hf_id, trust_remote_code=spec.trust_remote_code)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    kw = dict(dtype=getattr(torch, spec.dtype), device_map=spec.device_map,
              trust_remote_code=spec.trust_remote_code, **(spec.load_kwargs or {}))
    if spec.attn_implementation:
        kw["attn_implementation"] = spec.attn_implementation
    try:
        model = AutoModelForCausalLM.from_pretrained(spec.hf_id, **kw)
        rec["loaded_via"] = "AutoModelForCausalLM"
    except (ValueError, KeyError) as e:
        log(f"      AutoModelForCausalLM refused ({type(e).__name__}); trying ImageTextToText")
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(spec.hf_id, **kw)
        rec["loaded_via"] = "AutoModelForImageTextToText"
    model.eval()
    device = next(model.parameters()).device
    rec["load_seconds"] = round(time.perf_counter() - t0, 1)
    rec["model_class"] = type(model).__name__
    log(f"      loaded {rec['model_class']} via {rec['loaded_via']} in {rec['load_seconds']}s")

    # -- 3. hook target ----------------------------------------------------- #
    rec["stage_reached"] = "hook_target"
    dec = resolve_decoder(model, log)
    if dec is None:
        rec["error"] = "could not resolve a decoder with a .layers ModuleList"
        log(f"    [FAIL] {rec['error']}")
        return rec
    n_layers = len(dec.layers)
    tcfg = getattr(model.config, "text_config", model.config)
    n_blocks = int(getattr(tcfg, "num_hidden_layers", -1))
    rec["decoder_class"] = type(dec).__name__
    rec["n_layer_modules"] = n_layers
    rec["n_blocks_config"] = n_blocks
    rec["layers_match_config"] = bool(n_layers == n_blocks)
    log(f"      decoder={rec['decoder_class']}  len(.layers)={n_layers}  "
        f"config.num_hidden_layers={n_blocks}  match={rec['layers_match_config']}")

    # -- 6. chat template (before the forward pass -- we need the encoding) --- #
    rec["stage_reached"] = "chat_template"
    probe_text = prompts[0]
    msgs = ([{"role": "system", "content": spec.system_prompt}] if spec.system_prompt else [])
    msgs.append({"role": "user", "content": probe_text})
    try:
        rendered = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                           **(spec.chat_template_kwargs or {}))
        rec["has_chat_template"] = True
    except Exception as e:
        rendered = probe_text
        rec["has_chat_template"] = False
        rec["chat_template_error"] = f"{type(e).__name__}: {e}"
        log(f"      [WARN] no usable chat template ({type(e).__name__}) -- DISQUALIFYING "
            f"for a chat-behaviour study")
    rec["rendered_tail"] = rendered[-160:]
    ids = tok(rendered, return_tensors="pt", add_special_tokens=False)["input_ids"]
    rec["capture_token"] = tok.decode(ids[0, -1:])
    log(f"      template tail: {rec['rendered_tail']!r}")
    log(f"      prompt_last capture token: {rec['capture_token']!r}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # -- 4 & 5. layer mapping + no-op hook ---------------------------------- #
    rec["stage_reached"] = "layer_map"
    enc = {k: v.to(device) for k, v in
           tok(rendered, return_tensors="pt", add_special_tokens=False).items()}
    with torch.inference_mode():
        base = model(**enc, output_hidden_states=True)
    hs = base.hidden_states
    rec["n_hidden_rows"] = len(hs)
    rec["rows_equal_blocks_plus_1"] = bool(len(hs) == n_blocks + 1)

    want = [l for l in cfg.get("tier1", {}).get("verify_layers", [1, 4, 8, 16])
            if 1 <= l <= min(n_layers, len(hs) - 1)]
    checks = {}
    for l in want:
        cap = {}

        def grab(_m, _i, out, _cap=cap):
            _cap["h"] = hidden_of(out).detach()

        h = dec.layers[l - 1].register_forward_hook(grab)
        try:
            with torch.inference_mode():
                model(**enc)
        finally:
            h.remove()
        if "h" not in cap:
            checks[l] = {"error": "hook never fired"}
            continue
        a = cap["h"][0, -1, :].float()
        b = hs[l][0, -1, :].float().to(a.device)
        denom = max(float(b.abs().max()), 1e-6)
        checks[l] = {"max_abs_diff": round(float((a - b).abs().max()), 6),
                     "rel": round(float((a - b).abs().max()) / denom, 6),
                     "matches": bool(torch.allclose(a, b, atol=1e-2, rtol=1e-2))}
    rec["layer_map_checks"] = checks
    rec["layer_map_ok"] = bool(checks) and all(c.get("matches") for c in checks.values())
    log(f"      layer map (saved row l == output of layers[l-1]): "
        f"{'CONFIRMED' if rec['layer_map_ok'] else 'MISMATCH'}  {checks}")

    rec["stage_reached"] = "noop_hook"
    tgt = dec.layers[max(0, min(n_layers - 1, (want[-1] if want else 1) - 1))]

    def add_zero(_m, _i, out):
        hh = hidden_of(out)
        new = hh + torch.zeros_like(hh)
        return (new,) + tuple(out[1:]) if isinstance(out, (tuple, list)) else new

    h = tgt.register_forward_hook(add_zero)
    try:
        with torch.inference_mode():
            hooked = model(**enc)
    finally:
        h.remove()
    d = float((hooked.logits[0, -1].float() - base.logits[0, -1].float()).abs().max())
    rec["noop_hook_max_logit_diff"] = round(d, 6)
    rec["noop_hook_ok"] = bool(d < 1e-3)
    log(f"      no-op hook (write to residual, add 0.0): max|dlogit|={d:.2e} "
        f"{'OK' if rec['noop_hook_ok'] else 'PERTURBED -- hook mechanism is unsafe here'}")

    # -- 7. norms over n prompts -------------------------------------------- #
    rec["stage_reached"] = "norms"
    mats, row0 = [], []
    for p in prompts:
        m2 = ([{"role": "system", "content": spec.system_prompt}] if spec.system_prompt else [])
        m2.append({"role": "user", "content": p})
        try:
            txt = tok.apply_chat_template(m2, add_generation_prompt=True, tokenize=False,
                                          **(spec.chat_template_kwargs or {}))
        except Exception:
            txt = p
        e2 = {k: v.to(device) for k, v in
              tok(txt, return_tensors="pt", add_special_tokens=False).items()}
        with torch.inference_mode():
            o = model(**e2, output_hidden_states=True)
        vec = torch.stack([s[0, -1, :].detach().float().cpu() for s in o.hidden_states])
        mats.append(torch.linalg.vector_norm(vec, dim=1).numpy())
        row0.append(vec[0].numpy())
    A = np.stack(mats)
    med = np.median(A, axis=0)

    r0 = np.stack(row0)
    rec["row0_identical_across_prompts"] = bool(np.allclose(r0, r0[0], atol=1e-3))
    steer = med[1:n_blocks] if n_blocks > 1 else med
    rec["norms"] = {
        "median_by_row": [round(float(x), 3) for x in med],
        "row0": round(float(med[0]), 3),
        "row1": round(float(med[1]), 3) if len(med) > 1 else None,
        "row_nblocks_minus_1": round(float(med[n_blocks - 1]), 3) if n_blocks >= 2 else None,
        "last_row": round(float(med[-1]), 3),
        "steerable_min": round(float(steer.min()), 3),
        "steerable_max": round(float(steer.max()), 3),
        "steerable_ratio": round(float(steer.max() / max(steer.min(), 1e-9)), 2),
        "monotone_steps_pct": round(100 * float(np.mean(np.diff(steer) >= 0)), 1),
    }
    rec["peak_vram_gb"] = (round(torch.cuda.max_memory_allocated() / 1e9, 2)
                           if torch.cuda.is_available() else None)
    rec["n_prompts"] = len(prompts)
    rec["total_seconds"] = round(time.perf_counter() - t0, 1)
    rec["stage_reached"] = "done"
    rec["ok"] = bool(rec["layer_map_ok"] and rec["noop_hook_ok"] and rec["has_chat_template"])

    n = rec["norms"]
    log(f"      ||h|| row0={n['row0']} row1={n['row1']} row{n_blocks - 1}="
        f"{n['row_nblocks_minus_1']} last={n['last_row']}  steerable range "
        f"{n['steerable_min']}-{n['steerable_max']} ({n['steerable_ratio']}x)  "
        f"monotone {n['monotone_steps_pct']}%")
    log(f"      row0 identical across prompts: {rec['row0_identical_across_prompts']} "
        f"(expected True -- it means no direction exists at row 0)")
    log(f"      peak VRAM {rec['peak_vram_gb']} GB   total {rec['total_seconds']}s")

    del model, dec
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/screen.yaml")
    ap.add_argument("--models", default=None)
    ap.add_argument("--n-prompts", type=int, default=None)
    ap.add_argument("--preflight", action="store_true", help="arch check only, then exit")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--strict", action="store_true", help="abort on the first model failure")
    args = ap.parse_args()

    cfg = G.load_config(G.repo_path(args.config))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = Tee(OUT_DIR / "tier1.log")
    log(f"\n=== screen tier1 | {datetime.now(timezone.utc).isoformat()} ===")
    for k, v in env_info().items():
        log(f"  {k:16s} {v}")

    if args.smoke:
        sm = cfg.get("smoke") or {}
        cfg["models"] = sm.get("models", cfg["models"])
        if args.n_prompts is None:
            args.n_prompts = int(sm.get("n_prompts", 3))

    specs = G.parse_models(cfg, args.models.split(",") if args.models else None)
    log(f"  candidates: {[s.slug for s in specs]}")

    if args.preflight:
        ok = True
        for s in specs:
            try:
                p = G.preflight(s)
                mark = "ok  " if p["supported"] else "FAIL"
                log(f"  [{mark}] {s.slug:22s} {s.hf_id:44s} type={p['model_type']} "
                    f"H={p['hidden_size']} blocks={p['num_hidden_layers']}")
                ok = ok and p["supported"]
            except Exception as e:
                log(f"  [FAIL] {s.slug:22s} {s.hf_id:44s} {type(e).__name__}: {str(e)[:120]}")
                ok = False
        sys.exit(0 if ok else 1)

    n = args.n_prompts or int(cfg.get("tier1", {}).get("n_prompts", 8))
    sp = pd.read_parquet(G.repo_path(cfg["prompts"]["screen_set"]))
    # tier 1 is a smoke test: the prompts only drive a forward pass for the norm profile,
    # so take a spread across the groups rather than caring which ones they are
    prompts = (sp.sort_values(["group", "prompt_id"])
               .groupby("group", group_keys=False).head(max(1, n // sp.group.nunique()))
               .prompt.tolist())
    log(f"  prompts: {len(prompts)} (smoke test, forward pass only)")

    results = []
    for s in specs:
        log(f"\n--- {s.slug} ({s.hf_id}) ---")
        try:
            rec = screen_one(s, prompts, cfg, log)
        except Exception as e:
            log(f"    [FAIL] {type(e).__name__}: {str(e)[:300]}")
            if args.strict:
                traceback.print_exc()
                raise
            rec = {"slug": s.slug, "hf_id": s.hf_id, "ok": False,
                   "error": f"{type(e).__name__}: {str(e)[:500]}",
                   "traceback": traceback.format_exc()[-1500:]}
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        results.append(rec)
        (OUT_DIR / f"{s.slug}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")

    (OUT_DIR / "tier1_summary.json").write_text(
        json.dumps({"finished_utc": datetime.now(timezone.utc).isoformat(),
                    "env": env_info(), "results": results}, indent=2), encoding="utf-8")

    log("\n" + "=" * 100)
    log(f"{'model':24s} {'ok':4s} {'tmpl':5s} {'map':5s} {'noop':5s} {'blocks':>7} "
        f"{'||h|| range':>16} {'VRAM':>7}  stage/error")
    for r in results:
        nm = r.get("norms") or {}
        rng = (f"{nm.get('steerable_min')}-{nm.get('steerable_max')}" if nm else "-")
        log(f"{r['slug']:24s} {'yes' if r.get('ok') else 'NO ':4s} "
            f"{str(r.get('has_chat_template', '-')):5s} "
            f"{str(r.get('layer_map_ok', '-')):5s} {str(r.get('noop_hook_ok', '-')):5s} "
            f"{str(r.get('n_layer_modules', '-')):>7} {rng:>16} "
            f"{str(r.get('peak_vram_gb', '-')):>7}  "
            f"{r.get('error', r.get('stage_reached', ''))[:70]}")
    log(f"\nwrote {OUT_DIR / 'tier1_summary.json'}")
    log.close()


if __name__ == "__main__":
    main()
