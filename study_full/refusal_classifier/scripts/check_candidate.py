"""Preflight one candidate before committing GPU hours to it. Exits non-zero on failure.

Checks, in order:
  1. decoders: no text contains the pad/EOS id (it would cut the last-token pool short)
  2. pretrained weights load in fp32 with only classification-head keys missing
     (for NeoBERT this is what proves the SwiGLU shim matches the checkpoint), and
     buffers the checkpoint does not carry match an eager build (catches the
     zero-filled RoPE table transformers 5.x leaves in remote-code models)
  3. worst-case micro-batch -- the batch_size longest train rows at max_length --
     survives forward + backward + optimizer step; reports peak VRAM
  4. a few more random optimizer steps stay finite; reports training rows/s
  5. padding parity: probabilities do not change with batch padding
  6. LoRA only: merged weights predict the same as the adapter model
  7. save -> predict.load_classifier() round trip predicts the same

Usage:
  python -X utf8 scripts/check_candidate.py --config configs/neobert.yaml
  # 8 GB card: shrink the probe so it fits (the full-size probe belongs on the 16 GB box)
  python -X utf8 scripts/check_candidate.py --config configs/bge_m3.yaml --max-length 256 --batch-size 2
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data as D       # noqa: E402
from src import train as T      # noqa: E402
import predict as P             # noqa: E402

HEAD_PREFIXES = ("classifier", "score", "pooler", "dense")
MAX_VRAM_FRACTION = 0.92
TOL = 0.02


def fail(msg):
    print(f"FAIL {msg}", flush=True)
    sys.exit(1)


@torch.no_grad()
def class_probs(model, rows, pad_id, device, amp):
    model.eval()
    b = T.collate([{"input_ids": x, "label": 0} for x in rows], pad_id)
    with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
        out = model(input_ids=b["input_ids"].to(device),
                    attention_mask=b["attention_mask"].to(device))
    return torch.softmax(out.logits.float(), -1).cpu().numpy()


def compare(name, a, b):
    """Same near-tie rule as train.padding_parity(): a flipped label on a row whose
    top two classes are within TOL is numeric noise, not a mismatch."""
    diff = float(np.abs(a - b).max())
    top2 = np.sort(a, axis=1)[:, -2:]
    flips = int(((a.argmax(1) != b.argmax(1)) & (top2[:, 1] - top2[:, 0] > TOL)).sum())
    print(f"  {name}: max|dp|={diff:.4f} decisive_label_flips={flips}", flush=True)
    if diff > TOL or flips:
        fail(f"{name} mismatch")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-length", type=int, default=None, help="override (small GPUs)")
    ap.add_argument("--batch-size", type=int, default=None, help="override (small GPUs)")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--sgd", action="store_true",
                    help="small GPUs: SGD instead of AdamW so optimizer state fits "
                         "(peak VRAM then undercounts the real run)")
    args = ap.parse_args()

    cfg = D.load_config(D.repo_path(args.config))
    m, t = cfg["model"], cfg["train"]
    if args.max_length:
        m["max_length"] = args.max_length
    if args.batch_size:
        t["batch_size"] = args.batch_size
    max_length, bs = int(m["max_length"]), int(t["batch_size"])
    device = "cuda"
    # Hard cap: on Windows, overflowing VRAM silently spills into system RAM and
    # crawls instead of raising OOM.
    torch.cuda.set_per_process_memory_fraction(0.97)
    amp = torch.bfloat16 if t.get("bf16", True) and torch.cuda.is_bf16_supported() else None
    print(f"CHECK {cfg['candidate']} ({m['name']}) kind={cfg['kind']} "
          f"max_length={max_length} batch_size={bs} lora={bool(t.get('lora'))}", flush=True)
    T.print_env()
    T.set_seed(0)

    proc = D.repo_path("data/processed")
    splits = {s: pd.read_parquet(proc / f"{s}.parquet") for s in ("train", "val", "test")}
    tr, va = splits["train"], splits["val"]
    tok = T.load_tokenizer(cfg)
    T.check_special_tokens(tok)
    trunc, prefix = t.get("truncation", "head"), m.get("prefix", "")
    ids_tr = T.encode(tr["response_text"].tolist(), tok, max_length, trunc, prefix)
    ids_va = T.encode(va["response_text"].tolist(), tok, max_length, trunc, prefix)
    y_tr = tr["label_id"].to_numpy()

    # 1 ---------------------------------------------------------------------
    if cfg["kind"] == "decoder":
        gen = P.load_generated(D.repo_path(P.DEFAULT_GENERATED))
        texts = [str(x) for s in splits.values() for x in s["response_text"]] \
            + gen["answer"].tolist()
        raw = tok(texts, add_special_tokens=False)["input_ids"]
        hits = sum(tok.pad_token_id in x for x in raw)
        print(f"  pad id {tok.pad_token_id} ({tok.pad_token!r}) occurs inside "
              f"{hits} of {len(texts)} texts", flush=True)
        if hits:
            fail("pad/EOS token appears inside real text")

    # 2 ---------------------------------------------------------------------
    conf = T.load_model_config(cfg, tok)
    model, info = T.load_classifier_model(cfg, conf, loading_info=True)
    missing = sorted(info.get("missing_keys", []))
    body_missing = [k for k in missing if not k.startswith(HEAD_PREFIXES)]
    print(f"  load: missing={missing} unexpected={len(info.get('unexpected_keys', []))} "
          f"mismatched={list(info.get('mismatched_keys', []))}", flush=True)
    if body_missing or info.get("mismatched_keys"):
        fail(f"checkpoint does not cover the model body: {body_missing[:5]}")
    dtypes = {str(p.dtype) for p in model.parameters()}
    if dtypes != {"torch.float32"}:
        fail(f"parameters not fp32: {dtypes}")
    fresh = T.nonpersistent_buffers(T.load_classifier_model(cfg, conf, pretrained=False))
    loaded = T.nonpersistent_buffers(model)
    bad = [n for n, b in fresh.items() if n not in loaded or not torch.equal(b, loaded[n])]
    print(f"  non-checkpoint buffers: {sorted(fresh)[:4]}{'...' if len(fresh) > 4 else ''} "
          f"({len(fresh)}) differing from an eager build: {bad[:4]}", flush=True)
    if bad:
        fail(f"buffers not rebuilt after from_pretrained (e.g. zero-filled RoPE): {bad[:4]}")

    if t.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
    lora = t.get("lora")
    if lora:
        model = T.apply_lora(model, lora)
    model.to(device)
    lr = float(t["lr_sweep"][0])
    opt = (torch.optim.SGD(model.parameters(), lr=lr) if args.sgd else
           T.make_optimizer(model, lr, float(t.get("weight_decay", 0.01)), device))

    def step(rows, k):
        b = T.collate([{"input_ids": ids_tr[i], "label": int(y_tr[i])} for i in rows],
                      tok.pad_token_id)
        model.train()
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            out = model(input_ids=b["input_ids"].to(device),
                        attention_mask=b["attention_mask"].to(device))
        loss = F.cross_entropy(out.logits.float(), b["labels"].to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(t.get("max_grad_norm", 1.0)))
        opt.step()
        opt.zero_grad(set_to_none=True)
        T.check_finite_params(model, k)
        return loss.item(), tuple(b["input_ids"].shape)

    # 3 ---------------------------------------------------------------------
    torch.cuda.reset_peak_memory_stats()
    longest = np.argsort([-len(x) for x in ids_tr], kind="stable")[:bs]
    loss, shape = step(longest, 1)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    total = torch.cuda.get_device_properties(0).total_memory
    print(f"  worst-case micro-batch {shape}: loss={loss:.4f} peak VRAM "
          f"{peak / 1e9:.2f} of {total / 1e9:.2f} GB", flush=True)
    if peak > MAX_VRAM_FRACTION * total:
        fail(f"worst-case batch needs {peak / total:.0%} of VRAM; "
             f"halve batch_size and double grad_accum")

    # 4 ---------------------------------------------------------------------
    rng = np.random.default_rng(0)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    losses = [step(rng.choice(len(ids_tr), bs, replace=False), k + 2)[0]
              for k in range(args.steps)]
    torch.cuda.synchronize()
    rows_s = args.steps * bs / (time.perf_counter() - t0)
    print(f"  {args.steps} random steps: loss {losses[0]:.3f} -> {losses[-1]:.3f}  "
          f"~{rows_s:.1f} train rows/s (~{len(ids_tr) / rows_s / 60:.1f} min/epoch "
          f"before eval)", flush=True)

    # 5 ---------------------------------------------------------------------
    order = np.argsort([len(x) for x in ids_va], kind="stable")
    idx = order[np.linspace(0, len(order) - 1, 8).astype(int)]
    rows = [ids_va[i] for i in idx]
    T.padding_parity(model, rows, tok.pad_token_id, device, amp, log=print, tol=TOL)

    # 6 ---------------------------------------------------------------------
    if lora:
        before = class_probs(model, rows, tok.pad_token_id, device, amp)
        model = model.merge_and_unload()
        compare("LoRA merge parity", before, class_probs(model, rows, tok.pad_token_id,
                                                         device, amp))

    # 7 ---------------------------------------------------------------------
    ref = class_probs(model, rows, tok.pad_token_id, device, amp)
    work = Path(tempfile.mkdtemp(prefix=f"check_{cfg['candidate']}_"))
    try:
        (work / "config_snapshot.json").write_text(json.dumps({"config": cfg}),
                                                   encoding="utf-8")
        tok.save_pretrained(work / "tokenizer")
        torch.save(model.state_dict(), work / "best_state.pt")
        del model, opt
        gc.collect()
        torch.cuda.empty_cache()
        cfg2, tok2, model2, amp2 = P.load_classifier(work, device, print)
        got, _ = P.predict(va["response_text"].iloc[idx].tolist(), cfg2, tok2, model2,
                           device, amp2, token_budget=16384)
        compare("save/reload parity", ref, got)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print(f"PASS {cfg['candidate']}", flush=True)


if __name__ == "__main__":
    main()
