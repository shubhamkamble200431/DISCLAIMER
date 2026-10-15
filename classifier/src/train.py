from __future__ import annotations

import json
import math
import os
import platform
import random
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.metrics import LABELS3, classification_metrics, label_geometry


def print_env():
    print("-" * 78)
    print(f"torch.__version__              : {torch.__version__}")
    print(f"torch.version.cuda             : {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"torch.cuda.get_device_capability(): {torch.cuda.get_device_capability()}")
        print(f"device                         : {torch.cuda.get_device_name(0)}")
        print(f"bf16 supported                 : {torch.cuda.is_bf16_supported()}")
        print(f"total VRAM (GB)                : "
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}")
    else:
        print("torch.cuda.get_device_capability(): <no CUDA device>")
    print(f"platform                       : {platform.platform()}")
    print("-" * 78)


def env_snapshot():
    d = {"torch": torch.__version__, "torch_cuda": torch.version.cuda,
         "cuda_available": torch.cuda.is_available(),
         "platform": platform.platform(), "python": platform.python_version()}
    if torch.cuda.is_available():
        d["device_name"] = torch.cuda.get_device_name(0)
        d["device_capability"] = list(torch.cuda.get_device_capability())
        d["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
        d["total_vram_bytes"] = int(torch.cuda.get_device_properties(0).total_memory)
    try:
        d["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
            text=True).strip()
    except Exception:
        d["git_commit"] = None
    return d


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def install_swiglu_shim():
    import sys
    import types
    if "xformers.ops" in sys.modules:
        return

    class SwiGLU(nn.Module):
        def __init__(self, in_features, hidden_features, out_features=None, bias=True,
                     *, _pack_weights=True):
            super().__init__()
            out_features = out_features or in_features
            hidden_features = hidden_features or in_features
            self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
            self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

        def forward(self, x):
            x1, x2 = self.w12(x).chunk(2, dim=-1)
            return self.w3(F.silu(x1) * x2)

    from importlib.machinery import ModuleSpec
    pkg, ops = types.ModuleType("xformers"), types.ModuleType("xformers.ops")
    pkg.__spec__, ops.__spec__ = ModuleSpec("xformers", None), ModuleSpec("xformers.ops", None)
    ops.SwiGLU = SwiGLU
    pkg.ops = ops
    sys.modules["xformers"], sys.modules["xformers.ops"] = pkg, ops


def _hub_kwargs(m):
    if m.get("xformers_shim"):
        install_swiglu_shim()
    return {"trust_remote_code": bool(m.get("trust_remote_code", False)),
            "revision": m.get("revision")}


def load_tokenizer(cfg, path=None):
    from transformers import AutoTokenizer
    m = cfg["model"]
    kw = _hub_kwargs(m)
    if path is not None:
        kw.pop("revision")
    tok = AutoTokenizer.from_pretrained(path or m["name"], **kw)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model_config(cfg, tok):
    from transformers import AutoConfig
    conf = AutoConfig.from_pretrained(
        cfg["model"]["name"], num_labels=3,
        id2label={i: l for i, l in enumerate(LABELS3)},
        label2id={l: i for i, l in enumerate(LABELS3)}, **_hub_kwargs(cfg["model"]))
    if cfg["kind"] == "decoder" or getattr(conf, "pad_token_id", None) is None:
        conf.pad_token_id = tok.pad_token_id
    return conf


def load_classifier_model(cfg, conf, pretrained=True, loading_info=False):
    from transformers import AutoModelForSequenceClassification
    m = cfg["model"]
    kw = _hub_kwargs(m)
    if not pretrained:
        return AutoModelForSequenceClassification.from_config(
            conf, dtype=torch.float32, trust_remote_code=kw["trust_remote_code"])
    out = AutoModelForSequenceClassification.from_pretrained(
        m["name"], config=conf, dtype=torch.float32, output_loading_info=loading_info, **kw)
    if cfg["train"].get("head_init") == "zeros":
        nn.init.zeros_((out[0] if loading_info else out).score.weight)
    if kw["trust_remote_code"]:
        model = out[0] if loading_info else out
        fresh = load_classifier_model(cfg, conf, pretrained=False)
        for name, buf in nonpersistent_buffers(fresh).items():
            owner, _, attr = name.rpartition(".")
            setattr(model.get_submodule(owner), attr, buf)
        del fresh
    return out


def nonpersistent_buffers(model):
    saved = set(model.state_dict())
    return {n: b for n, b in model.named_buffers() if n not in saved}


def apply_lora(model, spec, log=print):
    from peft import LoraConfig, TaskType, get_peft_model
    if getattr(model, "is_gradient_checkpointing", False):
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.SEQ_CLS, r=int(spec.get("r", 16)),
        lora_alpha=int(spec.get("alpha", 32)), lora_dropout=float(spec.get("dropout", 0.05)),
        target_modules=list(spec["target_modules"])))
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    log(f"    lora: r={spec.get('r', 16)} alpha={spec.get('alpha', 32)} "
        f"trainable={n_train:,} of {n_all:,}")
    return model


def hidden_position(cfg):
    return "last" if cfg["kind"] == "decoder" else "first"


def special_affixes(tokenizer):
    probe = tokenizer("x", add_special_tokens=False)["input_ids"]
    full = tokenizer("x", add_special_tokens=True)["input_ids"]
    if probe:
        for i in range(len(full) - len(probe) + 1):
            if full[i:i + len(probe)] == probe:
                return full[:i], full[i + len(probe):]
    return [], []


def check_special_tokens(tokenizer, log=print):
    pre, suf = special_affixes(tokenizer)
    log(f"    special tokens: prefix={pre} suffix={suf}")
    if not pre and getattr(tokenizer, "cls_token_id", None) is not None:
        raise RuntimeError(
            f"{type(tokenizer).__name__} added no special tokens but declares "
            f"cls_token_id={tokenizer.cls_token_id}. This is the transformers 5.2.0 "
            f"DeBERTa-v3 regression; upgrade to transformers>=5.17.")
    return pre, suf


def encode(texts, tokenizer, max_length, truncation="head", prefix=""):
    texts = [prefix + str(t) for t in texts]
    if truncation == "head":
        return tokenizer(texts, truncation=True, max_length=max_length,
                         padding=False)["input_ids"]

    if truncation != "head_tail":
        raise ValueError(f"unknown truncation mode: {truncation}")

    pre, suf = special_affixes(tokenizer)
    n_special = len(pre) + len(suf)
    budget = max_length - n_special - 1
    head_n = budget // 2
    tail_n = budget - head_n
    raw = tokenizer(texts, add_special_tokens=False, truncation=False,
                    padding=False)["input_ids"]
    sep = tokenizer.sep_token_id
    if sep is None:
        sep = tokenizer.eos_token_id or tokenizer.pad_token_id
    out = []
    for ids in raw:
        if len(ids) <= budget + 1:
            out.append(pre + ids[:max_length - n_special] + suf)
        else:
            out.append(pre + ids[:head_n] + [sep] + ids[-tail_n:] + suf)
    return out


class IdsDataset(torch.utils.data.Dataset):
    def __init__(self, ids, labels):
        self.ids = ids
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return {"input_ids": self.ids[i], "label": int(self.labels[i])}


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        ids[i, :L] = torch.tensor(b["input_ids"], dtype=torch.long)
        att[i, :L] = 1
    return {"input_ids": ids, "attention_mask": att,
            "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long)}


def class_weights(labels, mode, n_classes=3, device="cpu"):
    if mode in (None, "none"):
        return None
    counts = np.bincount(np.asarray(labels), minlength=n_classes).astype(np.float64)
    if mode == "balanced":
        w = counts.sum() / (n_classes * np.maximum(counts, 1))
    else:
        raise ValueError(f"unknown class_weight mode for weights: {mode}")
    return torch.tensor(w, dtype=torch.float32, device=device)


def compute_loss(logits, labels, mode, weights, focal_gamma=2.0):
    if mode == "focal":
        logp = F.log_softmax(logits, dim=-1)
        p = logp.exp()
        tgt = F.one_hot(labels, logits.shape[-1]).float()
        focal = ((1 - p) ** focal_gamma) * logp
        loss = -(tgt * focal).sum(-1)
        if weights is not None:
            loss = loss * weights[labels]
        return loss.mean()
    return F.cross_entropy(logits, labels, weight=weights)


def make_optimizer(model, lr, weight_decay, device, log=print):
    if device == "cuda":
        try:
            opt = torch.optim.AdamW(model.parameters(), lr=lr,
                                    weight_decay=weight_decay, fused=True)
            log("    optimizer: AdamW(fused=True)")
            return opt
        except (RuntimeError, ValueError) as e:
            log(f"    optimizer: fused AdamW unavailable ({e}); falling back")
    log("    optimizer: AdamW(default)")
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def check_finite_params(model, step, where=""):
    bad = [n for n, p in model.named_parameters()
           if p.requires_grad and not torch.isfinite(p).all()]
    if bad:
        raise RuntimeError(
            f"non-finite parameters after optimizer step {step} {where}: "
            f"{len(bad)} tensors, first={bad[:3]}. "
            f"This is the sm_120 non-fused AdamW corruption — check make_optimizer().")


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype):
    model.eval()
    logits_all, labels_all = [], []
    for batch in loader:
        ids = batch["input_ids"].to(device, non_blocking=True)
        att = batch["attention_mask"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(input_ids=ids, attention_mask=att)
        logits_all.append(out.logits.float().cpu())
        labels_all.append(batch["labels"])
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all).numpy()
    probs = torch.softmax(logits, dim=-1).numpy()
    return labels, probs.argmax(1), probs


@torch.no_grad()
def encoder_hidden(model, ids, tokenizer, device, amp_dtype, batch_size=32,
                   position="first"):
    model.eval()
    ds = IdsDataset(ids, np.zeros(len(ids)))
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id))
    out = []
    for batch in loader:
        b_ids = batch["input_ids"].to(device)
        att = batch["attention_mask"].to(device)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            h = model(input_ids=b_ids, attention_mask=att,
                      output_hidden_states=True).hidden_states[-1]
        if position == "last":
            h = h[torch.arange(h.shape[0], device=h.device), att.sum(1) - 1]
        else:
            h = h[:, 0]
        out.append(h.float().cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def padding_parity(model, ids, pad_id, device, amp_dtype, log=print, tol=0.02):
    model.eval()

    def probs(rows):
        b = collate([{"input_ids": x, "label": 0} for x in rows], pad_id)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(input_ids=b["input_ids"].to(device),
                        attention_mask=b["attention_mask"].to(device))
        return torch.softmax(out.logits.float(), -1).cpu()

    batched = probs(ids)
    single = torch.cat([probs([x]) for x in ids])
    diff = float((batched - single).abs().max())
    top2 = single.topk(2, dim=-1).values
    flips = int(((batched.argmax(1) != single.argmax(1))
                 & (top2[:, 0] - top2[:, 1] > tol)).sum())
    lens = sorted(len(x) for x in ids)
    log(f"    padding parity: n={len(ids)} lengths {lens[0]}..{lens[-1]} "
        f"max|dp|={diff:.4f} decisive_label_flips={flips}")
    if diff > tol or flips:
        raise RuntimeError(f"padding parity failed (max|dp|={diff:.4f}, "
                           f"decisive_label_flips={flips}): predictions change with "
                           f"padding")
    return diff


def varied_lengths(ids, n=8):
    order = np.argsort([len(x) for x in ids], kind="stable")
    return [ids[i] for i in order[np.linspace(0, len(ids) - 1, n).astype(int)]]


def throughput_encoder(model, ids, tokenizer, device, amp_dtype, batch_size=32,
                       max_batches=8):
    model.eval()
    ds = IdsDataset(ids, np.zeros(len(ids)))
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    done, nb = 0, 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            b_ids = batch["input_ids"].to(device)
            att = batch["attention_mask"].to(device)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                model(input_ids=b_ids, attention_mask=att)
            done += b_ids.shape[0]
            nb += 1
            if nb >= max_batches:
                break
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return float(done / dt) if dt > 0 else float("nan")


def train_encoder(cfg, data, seed, run_dir, smoke=False, log=print):
    import gc

    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = cfg["model"]
    t = cfg["train"]
    lora = t.get("lora")

    max_length = 64 if smoke else int(m["max_length"])
    epochs = 1 if smoke else int(t["epochs"])
    lrs = [float(t["lr_sweep"][0])] if smoke else [float(x) for x in t["lr_sweep"]]

    tok = load_tokenizer(cfg)
    check_special_tokens(tok, log)
    enc = {k: encode(v[0], tok, max_length, t.get("truncation", "head"),
                     m.get("prefix", "")) for k, v in data.items()}

    amp_dtype = None
    if device == "cuda" and t.get("bf16", True) and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16

    best = None
    for lr in lrs:
        set_seed(seed)
        conf = load_model_config(cfg, tok)
        model = load_classifier_model(cfg, conf)
        if t.get("gradient_checkpointing", False):
            model.gradient_checkpointing_enable()
        if lora:
            model = apply_lora(model, lora, log)
        model.to(device)
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        train_ds = IdsDataset(enc["train"], data["train"][1])
        val_ds = IdsDataset(enc["val"], data["val"][1])
        bs = int(t["batch_size"])
        accum = int(t.get("grad_accum", 1))
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=bs, shuffle=True, drop_last=False,
            collate_fn=lambda b: collate(b, tok.pad_token_id))
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=int(t.get("eval_batch_size", 32)), shuffle=False,
            collate_fn=lambda b: collate(b, tok.pad_token_id))

        w = class_weights(data["train"][1], t.get("class_weight", "none"),
                          device=device) if t.get("class_weight", "none") != "focal" \
            else class_weights(data["train"][1], "balanced", device=device) \
            if t.get("focal_weighted", False) else None
        mode = t.get("class_weight", "none")

        opt = make_optimizer(model, lr, float(t.get("weight_decay", 0.01)), device, log)
        steps_per_epoch = math.ceil(len(train_loader) / accum)
        total_steps = max(1, steps_per_epoch * epochs)
        warmup = int(float(t.get("warmup_ratio", 0.1)) * total_steps)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: s / max(1, warmup) if s < warmup else
            max(0.0, (total_steps - s) / max(1, total_steps - warmup)))

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        patience = int(t.get("early_stopping_patience", 2))
        best_lr_run = None
        bad = 0
        n_opt_steps = 0
        t_start = time.perf_counter()

        for ep in range(epochs):
            model.train()
            opt.zero_grad(set_to_none=True)
            running = 0.0
            for step, batch in enumerate(train_loader):
                ids = batch["input_ids"].to(device, non_blocking=True)
                att = batch["attention_mask"].to(device, non_blocking=True)
                lab = batch["labels"].to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=amp_dtype,
                                    enabled=amp_dtype is not None):
                    out = model(input_ids=ids, attention_mask=att)
                    loss = compute_loss(out.logits.float(), lab, mode, w,
                                        float(t.get("focal_gamma", 2.0))) / accum
                loss.backward()
                running += loss.item() * accum
                if (step + 1) % accum == 0 or step + 1 == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   float(t.get("max_grad_norm", 1.0)))
                    opt.step()
                    sched.step()
                    opt.zero_grad(set_to_none=True)
                    n_opt_steps += 1
                    check_finite_params(model, n_opt_steps, f"(lr={lr:g})")

            y, p, pr = evaluate(model, val_loader, device, amp_dtype)
            mt = classification_metrics(y, p, pr)
            log(f"    lr={lr:g} epoch {ep + 1}/{epochs} "
                f"train_loss={running / max(1, len(train_loader)):.4f} "
                f"val_macroF1={mt['macro_f1']:.4f} val_DISC_F1={mt['disclaimer_f1']:.4f}")

            score = mt["disclaimer_f1"]
            if best_lr_run is None or score > best_lr_run["score"] + 1e-9:
                best_lr_run = {"score": score, "epoch": ep + 1, "metrics": mt,
                               "preds": p, "probs": pr,
                               "state": {k: v.detach().cpu().clone()
                                         for k, v in model.named_parameters()
                                         if v.requires_grad} if lora else
                                        {k: v.detach().cpu().clone()
                                         for k, v in model.state_dict().items()}}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    log(f"    early stop at epoch {ep + 1} "
                        f"(no val DISCLAIMER F1 gain for {patience} epochs)")
                    break

        wall = time.perf_counter() - t_start
        peak = int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0
        loaded = model.load_state_dict(best_lr_run["state"], strict=not lora)
        if lora:
            if loaded.unexpected_keys:
                raise RuntimeError(f"LoRA best state has unexpected keys: "
                                   f"{loaded.unexpected_keys[:3]}")
            model = model.merge_and_unload()
        if smoke:
            padding_parity(model, varied_lengths(enc["val"]), tok.pad_token_id, device,
                           amp_dtype, log)
        tp = throughput_encoder(model, enc["val"], tok, device, amp_dtype,
                                batch_size=int(t.get("throughput_batch_size", 32)))
        position = hidden_position(cfg)
        h_val = encoder_hidden(model, enc["val"], tok, device, amp_dtype,
                               batch_size=int(t.get("eval_batch_size", 32)),
                               position=position)
        geom = label_geometry(h_val, data["val"][1],
                              pooling="cls_last_layer" if position == "first"
                              else "last_token_last_layer")
        log(f"    geometry: " + "  ".join(
            f"{k} d={v['euclidean']:.2f} cos={v['cosine']:.3f}"
            for k, v in geom["pairs"].items() if "euclidean" in v))
        rec = {"lr": lr, "best_epoch": best_lr_run["epoch"],
               "_preds": best_lr_run["preds"].tolist(),
               "_probs": best_lr_run["probs"].tolist(),
               "val_metrics": best_lr_run["metrics"], "geometry": geom,
               "wall_seconds": wall,
               "peak_vram_bytes": peak, "n_trainable_params": n_trainable,
               "throughput_per_sec": tp}
        log(f"    lr={lr:g} BEST epoch={rec['best_epoch']} "
            f"val_DISC_F1={best_lr_run['score']:.4f} wall={wall:.1f}s "
            f"peakVRAM={peak / 1e9:.2f}GB thr={tp:.1f}/s")

        if best is None or best_lr_run["score"] > best["val_metrics"]["disclaimer_f1"] + 1e-9:
            best = rec
            best["_state"] = best_lr_run["state"] if not lora else \
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best["_tok"] = tok
        del model, opt, sched, best_lr_run
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    return best


def train_final(cfg, texts, labels, seed, run_dir, lr, epochs, smoke=False, log=print):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m, t = cfg["model"], cfg["train"]
    max_length = 64 if smoke else int(m["max_length"])

    tok = load_tokenizer(cfg)
    check_special_tokens(tok, log)
    ids = encode(texts, tok, max_length, t.get("truncation", "head"), m.get("prefix", ""))

    amp_dtype = None
    if device == "cuda" and t.get("bf16", True) and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16

    conf = load_model_config(cfg, tok)
    model = load_classifier_model(cfg, conf)
    if t.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
    if t.get("lora"):
        model = apply_lora(model, t["lora"], log)
    model.to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    bs = int(t["batch_size"])
    accum = int(t.get("grad_accum", 1))
    loader = torch.utils.data.DataLoader(
        IdsDataset(ids, labels), batch_size=bs, shuffle=True, drop_last=False,
        collate_fn=lambda b: collate(b, tok.pad_token_id))

    mode = t.get("class_weight", "none")
    w = class_weights(labels, mode, device=device) if mode != "focal"         else class_weights(labels, "balanced", device=device)         if t.get("focal_weighted", False) else None

    opt = make_optimizer(model, lr, float(t.get("weight_decay", 0.01)), device, log)
    steps_per_epoch = math.ceil(len(loader) / accum)
    total_steps = max(1, steps_per_epoch * epochs)
    warmup = int(float(t.get("warmup_ratio", 0.1)) * total_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s_: s_ / max(1, warmup) if s_ < warmup else
        max(0.0, (total_steps - s_) / max(1, total_steps - warmup)))

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    n_opt_steps = 0
    t_start = time.perf_counter()

    for ep in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        running = 0.0
        for step, batch in enumerate(loader):
            b_ids = batch["input_ids"].to(device, non_blocking=True)
            att = batch["attention_mask"].to(device, non_blocking=True)
            lab = batch["labels"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                out = model(input_ids=b_ids, attention_mask=att)
                loss = compute_loss(out.logits.float(), lab, mode, w,
                                    float(t.get("focal_gamma", 2.0))) / accum
            loss.backward()
            running += loss.item() * accum
            if (step + 1) % accum == 0 or step + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               float(t.get("max_grad_norm", 1.0)))
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                n_opt_steps += 1
                check_finite_params(model, n_opt_steps, f"(lr={lr:g})")
        log(f"    epoch {ep + 1}/{epochs} train_loss={running / max(1, len(loader)):.4f}")

    wall = time.perf_counter() - t_start
    peak = int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0
    if t.get("lora"):
        model = model.merge_and_unload()

    eval_loader = torch.utils.data.DataLoader(
        IdsDataset(ids, labels), batch_size=int(t.get("eval_batch_size", 32)),
        shuffle=False, collate_fn=lambda b: collate(b, tok.pad_token_id))
    y, pred, prob = evaluate(model, eval_loader, device, amp_dtype)
    fit_metrics = classification_metrics(y, pred, prob)
    tp = throughput_encoder(model, ids, tok, device, amp_dtype,
                            batch_size=int(t.get("throughput_batch_size", 32)))
    position = hidden_position(cfg)
    h = encoder_hidden(model, ids, tok, device, amp_dtype,
                       batch_size=int(t.get("eval_batch_size", 32)), position=position)
    geom = label_geometry(h, labels, pooling="cls_last_layer" if position == "first"
                          else "last_token_last_layer")
    log("    geometry: " + "  ".join(
        f"{k} d={v['euclidean']:.2f} cos={v['cosine']:.3f}"
        for k, v in geom["pairs"].items() if "euclidean" in v))

    torch.save(model.state_dict(), run_dir / "final_state.pt")
    tok.save_pretrained(run_dir / "tokenizer")
    (run_dir / "label_map.json").write_text(json.dumps(
        {"id2label": {i: l for i, l in enumerate(LABELS3)},
         "label2id": {l: i for i, l in enumerate(LABELS3)},
         "max_length": max_length,
         "truncation": t.get("truncation", "head"),
         "prefix": m.get("prefix", ""),
         "model_name": m["name"]}, indent=2), encoding="utf-8")

    return {"fit_metrics": fit_metrics, "geometry": geom, "lr": lr, "epochs": epochs,
            "wall_seconds": wall, "peak_vram_bytes": peak,
            "n_trainable_params": n_trainable, "throughput_per_sec": tp,
            "n_train_rows": len(texts)}


@torch.no_grad()
def embed_frozen(texts, cfg, device=None, batch_size=16, smoke=False):
    from transformers import AutoModel, AutoTokenizer

    m = cfg["model"]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    max_length = 64 if smoke else int(m["max_length"])
    prefix = m.get("prefix", "")
    if "e5" in m["name"].lower() and not prefix:
        raise ValueError("E5 models require a prefix (e.g. 'query: '); none configured")

    tok = AutoTokenizer.from_pretrained(m["name"])
    model = AutoModel.from_pretrained(m["name"]).to(device).eval()
    amp = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else None

    texts = [prefix + str(t) for t in texts]
    out = []
    for s in range(0, len(texts), batch_size):
        b = tok(texts[s:s + batch_size], truncation=True, max_length=max_length,
                padding=True, return_tensors="pt").to(device)
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            h = model(**b).last_hidden_state
        mask = b["attention_mask"].unsqueeze(-1).float()
        pooled = (h.float() * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        out.append(F.normalize(pooled, p=2, dim=-1).cpu())
    emb = torch.cat(out).numpy()
    n_params = sum(p.numel() for p in model.parameters())
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return emb, n_params


def train_frozen_head(cfg, emb, labels, emb_val, labels_val, seed, log=print):
    h = cfg["head"]
    if h["kind"] == "logreg":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(C=float(h.get("C", 1.0)), max_iter=int(h.get("max_iter", 3000)),
                                 class_weight=None if str(
                                     cfg["train"].get("class_weight", "none")) == "none"
                                 else "balanced",
                                 random_state=seed)
    elif h["kind"] == "mlp":
        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(hidden_layer_sizes=tuple(h.get("hidden", [256])),
                            max_iter=int(h.get("max_iter", 500)), random_state=seed)
    else:
        raise ValueError(f"unknown head kind: {h['kind']}")
    t0 = time.perf_counter()
    clf.fit(emb, labels)
    fit_s = time.perf_counter() - t0
    p = clf.predict(emb_val)
    pr = clf.predict_proba(emb_val)
    mt = classification_metrics(labels_val, p, pr)
    log(f"    head={h['kind']} fit={fit_s:.2f}s val_macroF1={mt['macro_f1']:.4f} "
        f"val_DISC_F1={mt['disclaimer_f1']:.4f}")
    n_head = int(getattr(clf, "coef_", np.zeros((3, emb.shape[1]))).size
                 + getattr(clf, "intercept_", np.zeros(3)).size) if h["kind"] == "logreg" \
        else int(sum(c.size for c in clf.coefs_) + sum(b.size for b in clf.intercepts_))
    return clf, mt, {"fit_seconds": fit_s, "n_head_params": n_head}
