from __future__ import annotations

import json
import re
import sys
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path

import numpy as np
import torch

try:
    from src import generate as G
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src import generate as G

__all__ = [
    "BASELINE_TAG", "DIRECTION_NAMES", "ALLOWED_ATTN",
    "EAGER_MAX_BATCH", "EAGER_MAX_PROMPT_TOKENS",
    "condition_tag", "parse_condition_tag", "is_baseline",
    "resolve_decoder", "n_steerable_layers", "assert_steerable",
    "PrefillLastTokenSteer", "steering_hook",
    "BatchedGenerator", "preflight_hook_matches_capture",
    "write_answers_jsonl", "read_done_ids",
    "unit", "sigma_along",
    "SPLIT_NAMES", "split_fractions",
]

SPLIT_NAMES = ("direction", "probing", "steering")


def split_fractions(cfg: dict) -> dict:
    fr = (cfg.get("study") or {}).get("split_fractions") or {}
    if set(fr) != set(SPLIT_NAMES):
        raise ValueError(f"study.split_fractions has keys {sorted(fr)}; expected exactly "
                         f"{list(SPLIT_NAMES)} (no validation split)")
    out = {k: float(fr[k]) for k in SPLIT_NAMES}
    if any(v <= 0 for v in out.values()) or abs(sum(out.values()) - 1.0) > 1e-9:
        raise ValueError(f"study.split_fractions {out} must be positive and sum to 1")
    return out

BASELINE_TAG = "baseline_ap0"

DIRECTION_NAMES = ("dR", "dD")

ALLOWED_ATTN = ("sdpa", "flash_attention_2")

BATCHED_MIN_COS = 0.998
BATCHED_MAX_REL_L2 = 0.05
BATCHED_T1_MAX_REL = 1e-3
EAGER_MAX_BATCH = 4
EAGER_MAX_PROMPT_TOKENS = 2048

_TAG_RE = re.compile(r"^L(?P<layer>\d{3})_(?P<direction>[A-Za-z][A-Za-z0-9]*)"
                     r"_a(?P<sign>[mp])(?P<mag>\d+(?:p\d+)?)$")


def _fmt_mag(x: float) -> str:
    s = f"{abs(float(x)):.6f}".rstrip("0").rstrip(".")
    return (s or "0").replace(".", "p")


def condition_tag(layer: int, direction: str, alpha: float) -> str:
    layer = int(layer)
    if not 0 <= layer <= 999:
        raise ValueError(f"layer {layer} does not fit the 3-digit tag field")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", str(direction)):
        raise ValueError(f"direction {direction!r} must be alphanumeric, e.g. 'dR' / 'dD'")
    sign = "m" if float(alpha) < 0 else "p"
    return f"L{layer:03d}_{direction}_a{sign}{_fmt_mag(alpha)}"


def parse_condition_tag(tag: str) -> dict:
    if tag == BASELINE_TAG:
        return {"layer": None, "direction": "baseline", "alpha": 0.0}
    m = _TAG_RE.match(str(tag))
    if not m:
        raise ValueError(f"not a condition tag: {tag!r}")
    mag = float(m.group("mag").replace("p", "."))
    return {"layer": int(m.group("layer")),
            "direction": m.group("direction"),
            "alpha": -mag if m.group("sign") == "m" else mag}


def is_baseline(tag: str) -> bool:
    return tag == BASELINE_TAG


def unit(v) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64).ravel()
    n = float(np.linalg.norm(a))
    if not np.isfinite(n) or n < 1e-12:
        raise ValueError(f"cannot normalise a vector of norm {n}")
    return a / n


def sigma_along(X, d) -> float:
    X = np.asarray(X, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64).ravel()
    if X.ndim != 2 or X.shape[1] != d.size:
        raise ValueError(f"X {X.shape} does not match direction of size {d.size}")
    return float(np.std(X @ d, ddof=1))


def resolve_decoder(model) -> torch.nn.Module:
    getter = getattr(model, "get_decoder", None)
    if getter is None:
        raise AttributeError(
            f"{type(model).__name__} has no get_decoder(); cannot locate the decoder "
            f"blocks safely. Do not guess an attribute path -- add an explicit mapping.")
    dec = getter()
    layers = getattr(dec, "layers", None)
    if layers is None or not hasattr(layers, "__len__") or len(layers) == 0:
        raise AttributeError(
            f"get_decoder() returned {type(dec).__name__} with no non-empty .layers "
            f"ModuleList; the steering hook has nothing to attach to.")
    return dec


def n_steerable_layers(model) -> int:
    return len(resolve_decoder(model).layers)


def assert_steerable(layer: int, n_blocks: int) -> None:
    layer = int(layer)
    if layer == 0:
        raise ValueError("layer 0 is the embedding output: byte-identical across prompts, "
                         "no direction exists there and nothing can be hooked")
    if layer >= n_blocks:
        raise ValueError(f"layer {layer} >= n_blocks={n_blocks}: row {n_blocks} is the "
                         f"post-final-norm state, not a residual site")
    if layer < 0:
        raise ValueError(f"layer {layer} is negative")


def _assert_left_padding(tok) -> None:
    side = getattr(tok, "padding_side", None)
    if side != "left":
        raise AssertionError(
            f"tokenizer padding_side is {side!r}, must be 'left'. The hook writes to "
            f"position -1 of the prefill; under right padding that position is a PAD "
            f"token for every sequence but the longest.")


class PrefillLastTokenSteer:

    def __init__(self, vec, alpha: float, tok=None, expect_hidden: int | None = None):
        if tok is not None:
            _assert_left_padding(tok)
        v = vec.detach() if isinstance(vec, torch.Tensor) else torch.as_tensor(np.asarray(vec))
        self.vec = v.reshape(-1).to(torch.float32)
        if expect_hidden is not None and self.vec.numel() != int(expect_hidden):
            raise ValueError(f"direction has {self.vec.numel()} dims, model hidden size "
                             f"is {expect_hidden}")
        self.alpha = float(alpha)
        self.n_fired = 0
        self.n_decode_skipped = 0
        self._delta = None

    def _delta_for(self, hidden: torch.Tensor) -> torch.Tensor:
        key = (hidden.device, hidden.dtype)
        if self._delta is None or self._delta[0] != key:
            d = (self.vec * self.alpha).to(device=hidden.device, dtype=hidden.dtype)
            self._delta = (key, d)
        return self._delta[1]

    def __call__(self, module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(hidden) or hidden.dim() != 3:
            raise TypeError(
                f"hooked module returned {type(output).__name__} whose hidden state is "
                f"{type(hidden).__name__}; expected a [B, T, H] tensor (or a tuple "
                f"starting with one)")
        if hidden.shape[-1] != self.vec.numel():
            raise ValueError(f"hidden size {hidden.shape[-1]} != direction size "
                             f"{self.vec.numel()}")
        if hidden.shape[1] <= 1:
            self.n_decode_skipped += 1
            return output

        if hidden.requires_grad:
            hidden = hidden.clone()
        hidden[:, -1, :] = hidden[:, -1, :] + self._delta_for(hidden)
        self.n_fired += 1
        return ((hidden,) + tuple(output[1:])) if isinstance(output, tuple) else hidden


@contextmanager
def steering_hook(model, layer: int, vec, alpha: float, tok=None, n_blocks: int | None = None):
    dec = resolve_decoder(model)
    nb = int(n_blocks) if n_blocks is not None else len(dec.layers)
    assert_steerable(layer, nb)
    if tok is not None:
        _assert_left_padding(tok)

    hidden_size = getattr(getattr(model.config, "text_config", model.config), "hidden_size", None)
    steer = PrefillLastTokenSteer(vec, alpha, tok=tok, expect_hidden=hidden_size)
    handle = dec.layers[int(layer) - 1].register_forward_hook(steer)
    try:
        yield steer
    finally:
        handle.remove()


class BatchedGenerator:

    def __init__(self, runner, batch_size: int = 8, max_prompt_tokens: int | None = None,
                 log=print):
        self.runner = runner
        self.log = log
        self.requested_batch_size = int(batch_size)
        self.batch_size = int(batch_size)
        self.max_prompt_tokens = int(max_prompt_tokens) if max_prompt_tokens else None
        self.n_padded_repeats = 0
        self.n_oom_halvings = 0
        self._rendered: dict[int, tuple[str, int]] = {}

        attn = getattr(runner.spec, "attn_implementation", None)
        if attn not in ALLOWED_ATTN:
            self.log(f"  !! WARNING: attn_implementation={attn!r} is not one of "
                     f"{list(ALLOWED_ATTN)}. Eager attention materialises the full "
                     f"attention matrix, so batching is memory-unsafe.")
            if self.batch_size > EAGER_MAX_BATCH:
                self.log(f"  !! clamping batch size {self.batch_size} -> {EAGER_MAX_BATCH}")
                self.batch_size = EAGER_MAX_BATCH
            cap = EAGER_MAX_PROMPT_TOKENS if self.max_prompt_tokens is None \
                else min(self.max_prompt_tokens, EAGER_MAX_PROMPT_TOKENS)
            if cap != self.max_prompt_tokens:
                self.log(f"  !! prompt-length cap set to {cap} tokens; longer prompts run "
                         f"one per batch")
            self.max_prompt_tokens = cap
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")

    def _render(self, row) -> tuple[str, int]:
        pid = int(row.prompt_id)
        hit = self._rendered.get(pid)
        if hit is not None:
            return hit

        text, enc = self.runner.render(row.prompt)
        ids = self.runner.tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
        ids = [int(v) for v in ids]

        ref = enc["input_ids"]
        ref = ref[0].tolist() if hasattr(ref, "tolist") else list(ref[0])
        ref = [int(v) for v in ref]
        if ids != ref:
            raise RuntimeError(
                f"prompt {pid}: re-tokenising the rendered chat text with "
                f"add_special_tokens=False gives {len(ids)} ids, the capture path's "
                f"apply_chat_template gave {len(ref)}. The batched path would steer a "
                f"different position than the one the activations were captured at.")
        if len(ids) < 2:
            raise RuntimeError(f"prompt {pid} renders to {len(ids)} token(s); a 1-token "
                               f"prefill is indistinguishable from a decode step")
        self._rendered[pid] = (text, len(ids))
        return self._rendered[pid]

    def plan_batches(self, rows) -> list[list[int]]:
        lengths = [self._render(r)[1] for r in rows]
        order = sorted(range(len(rows)), key=lambda i: (lengths[i], int(rows[i].prompt_id)))

        cap = self.max_prompt_tokens
        short = [i for i in order if cap is None or lengths[i] <= cap]
        long = [i for i in order if cap is not None and lengths[i] > cap]
        if long:
            self.log(f"  {len(long)} prompt(s) exceed the {cap}-token cap; they run one "
                     f"per batch")

        batches: list[list[int]] = []
        for start in range(0, len(short), self.batch_size):
            chunk = short[start:start + self.batch_size]
            if len(chunk) < self.batch_size and len(short) > self.batch_size:
                pad = self.batch_size - len(chunk)
                chunk = chunk + [chunk[-1]] * pad
                self.n_padded_repeats += pad
            batches.append(chunk)
        batches.extend([[i] for i in long])

        batches.sort(key=lambda b: -max(lengths[i] for i in b))
        self.log(f"  batch plan: {len(rows)} rows -> {len(batches)} batches "
                 f"(size {self.batch_size}, {self.n_padded_repeats} padding repeats, "
                 f"tokens {min(lengths)}..{max(lengths)})")
        return batches

    def generate(self, rows, batches, hook_ctx=None, extra: dict | None = None) -> list[dict]:
        if not batches:
            return []
        stack = ExitStack()
        shared = None
        if hook_ctx is not None and not callable(hook_ctx):
            shared = stack.enter_context(hook_ctx)
        out: dict[int, dict] = {}
        t0 = time.perf_counter()
        with stack:
            for bi, batch in enumerate(batches):
                self._run_batch(rows, list(batch), bi, len(batches), hook_ctx, shared,
                                out, extra or {})
        missing = [i for i in range(len(rows)) if i not in out]
        if missing:
            raise RuntimeError(f"{len(missing)} row(s) produced no record: {missing[:10]}")
        self.log(f"  generated {len(rows)} rows in {time.perf_counter() - t0:.1f}s")
        return [out[i] for i in range(len(rows))]

    def _run_batch(self, rows, batch, bi, n_batches, hook_ctx, shared, out, extra):
        work = [batch]
        while work:
            chunk = work.pop(0)
            try:
                out.update(self._forward(rows, chunk, bi, hook_ctx, shared, extra))
            except torch.cuda.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if len(chunk) == 1:
                    raise
                half = len(chunk) // 2
                self.n_oom_halvings += 1
                self.log(f"  !! OOM on batch {bi + 1}/{n_batches} (size {len(chunk)}); "
                         f"halving to {half}/{len(chunk) - half}. These rows leave the "
                         f"static-membership guarantee -- re-run this model at a lower "
                         f"--batch-size for a clean comparison.")
                work[:0] = [chunk[:half], chunk[half:]]

    @torch.inference_mode()
    def _forward(self, rows, chunk, bi, hook_ctx, shared, extra) -> dict[int, dict]:
        runner = self.runner
        tok, model = runner.tok, runner.model
        texts = [self._render(rows[i])[0] for i in chunk]

        prev_side = tok.padding_side
        tok.padding_side = "left"
        try:
            _assert_left_padding(tok)
            enc = tok(texts, padding=True, add_special_tokens=False, return_tensors="pt")

            device = next(model.parameters()).device
            enc = {k: v.to(device) for k, v in dict(enc).items()
                   if k in ("input_ids", "attention_mask")}
            width = int(enc["input_ids"].shape[1])

            gen_kwargs = dict(runner.gen_kwargs)
            gen_kwargs.setdefault("do_sample", False)
            if gen_kwargs.get("do_sample"):
                raise RuntimeError("the study is greedy-only; do_sample must be False")

            with ExitStack() as st:
                steer = shared
                if callable(hook_ctx):
                    steer = st.enter_context(hook_ctx())
                if steer is not None:
                    steer.n_fired = 0
                t0 = time.perf_counter()
                gen = model.generate(**enc, return_dict_in_generate=True, use_cache=True,
                                     pad_token_id=tok.pad_token_id, **gen_kwargs)
                batch_s = time.perf_counter() - t0
                if steer is not None and steer.n_fired != 1:
                    raise RuntimeError(
                        f"steering hook fired {steer.n_fired} times during one generate() "
                        f"call, expected exactly 1 (the prefill). Something re-ran the "
                        f"prefill or chunked it; the intervention is not what the study "
                        f"defines.")

            max_new = int(gen_kwargs.get("max_new_tokens", 0))
            eos_ids = getattr(runner, "_eos_ids", None) or {tok.eos_token_id}
            recs: dict[int, dict] = {}
            for k, i in enumerate(chunk):
                if i in recs:
                    continue
                ids = gen.sequences[k][width:].tolist()
                n_gen, reason = len(ids), "length"
                for pos, t in enumerate(ids):
                    if t in eos_ids:
                        n_gen, reason = pos + 1, "eos"
                        break
                kept = ids[:n_gen]
                recs[i] = {
                    "prompt_id": int(rows[i].prompt_id),
                    "answer": tok.decode(kept, skip_special_tokens=True),
                    "n_prompt_tokens": int(enc["attention_mask"][k].sum()),
                    "n_generated_tokens": int(n_gen),
                    "finish_reason": reason,
                    "truncated": bool(reason == "length" and n_gen >= max_new),
                    "gen_s": round(batch_s / max(1, len(chunk)), 4),
                    "batch_index": int(bi),
                    "batch_size_actual": int(len(chunk)),
                    "batch_seconds": round(batch_s, 3),
                    **extra,
                }
        finally:
            tok.padding_side = prev_side
        return recs


def preflight_hook_matches_capture(runner, rows, hidden_root, atol: float = 1e-2,
                                   rtol: float = 5e-3, min_cos: float = 0.9995,
                                   n_probe: int = 4, point: str = "prompt_last",
                                   log=print, batched_capture: bool = False) -> dict:
    model = runner.model
    dec = resolve_decoder(model)
    n_blocks = len(dec.layers)
    cfg_blocks = int(getattr(getattr(model.config, "text_config", model.config),
                             "num_hidden_layers", -1))
    if n_blocks != cfg_blocks:
        raise RuntimeError(f"decoder has {n_blocks} layers but config says "
                           f"num_hidden_layers={cfg_blocks}; resolve_decoder() found the "
                           f"wrong module")

    root = Path(hidden_root)
    hid = root / "hidden" if (root / "hidden").is_dir() else root
    layers = list(range(1, n_blocks))
    probe = list(rows)[:max(1, int(n_probe))]
    zero = torch.zeros(runner.hidden_size, dtype=torch.float32)

    log(f"  preflight: {len(probe)} prompt(s) x {len(layers)} steerable layers "
        f"(rows 1..{n_blocks - 1}), no-op hooks, batch 1")

    handles, hooks = [], []
    for l in layers:
        h = PrefillLastTokenSteer(zero, 0.0, expect_hidden=runner.hidden_size)
        handles.append(dec.layers[l - 1].register_forward_hook(h))
        hooks.append(h)

    layer_out: dict[int, np.ndarray] = {}
    if batched_capture:
        def _make_reader(l):
            def _read(_mod, _inp, out):
                t = out[0] if isinstance(out, (tuple, list)) else out
                layer_out[l] = t[0, -1, :].detach().float().cpu().numpy().astype(np.float64)
            return _read
        for l in layers:
            handles.append(dec.layers[l - 1].register_forward_hook(_make_reader(l)))

    per_layer = {l: {"max_abs": 0.0, "max_rel": 0.0, "min_cos": 1.0,
                     "max_rel_l2": 0.0, "t1_max_rel": 0.0} for l in layers}
    checked = []
    try:
        device = next(model.parameters()).device
        for row in probe:
            pid = int(row.prompt_id)
            target = G.hidden_target(hid, pid, "per_layer") if (hid / str(pid)).is_dir() \
                else hid / f"{pid}.npz"
            if not Path(target).exists():
                raise FileNotFoundError(
                    f"no stored activations for prompt {pid} at {target}; run the "
                    f"generation stage for this model first")
            stored, stored_layers = G.load_hidden(target, point, None)
            stored_ix = {int(v): k for k, v in enumerate(np.asarray(stored_layers).ravel())}

            _, enc = runner.render(row.prompt)
            enc = {k: v.to(device) for k, v in enc.items()
                   if k in ("input_ids", "attention_mask")}
            for h in hooks:
                h.n_fired = 0
            with torch.inference_mode():
                out = model(**enc, output_hidden_states=True, use_cache=False)
            bad = [l for l, h in zip(layers, hooks) if h.n_fired != 1]
            if bad:
                raise RuntimeError(f"no-op hooks did not fire exactly once at layers {bad}")

            for l in layers:
                if l not in stored_ix:
                    raise RuntimeError(f"layer row {l} is absent from {target}")
                got = out.hidden_states[l][0, -1, :].detach().float().cpu().numpy()
                ref = stored[stored_ix[l]].astype(np.float64)
                got = got.astype(np.float64)
                diff = float(np.abs(got - ref).max())
                scale = max(1e-6, float(np.abs(ref).max()))
                cos = float(np.dot(got, ref) /
                            max(1e-12, np.linalg.norm(got) * np.linalg.norm(ref)))
                s = per_layer[l]
                s["max_abs"] = max(s["max_abs"], diff)
                s["max_rel"] = max(s["max_rel"], diff / scale)
                s["min_cos"] = min(s["min_cos"], cos)
                s["max_rel_l2"] = max(s["max_rel_l2"], float(
                    np.linalg.norm(got - ref) / max(1e-12, np.linalg.norm(ref))))
                if batched_capture:
                    same = got
                    hooked = layer_out[l]
                    s["t1_max_rel"] = max(s["t1_max_rel"], float(
                        np.abs(hooked - same).max() / max(1e-6, np.abs(same).max())))
            del out
            checked.append(pid)
    finally:
        for h in handles:
            h.remove()

    def _failed(s: dict) -> bool:
        if batched_capture:
            return (s["t1_max_rel"] > BATCHED_T1_MAX_REL
                    or s["min_cos"] < BATCHED_MIN_COS
                    or s["max_rel_l2"] > BATCHED_MAX_REL_L2)
        return s["min_cos"] < min_cos or (s["max_abs"] > atol and s["max_rel"] > rtol)

    failed = [l for l, s in per_layer.items() if _failed(s)]
    worst = max(per_layer.items(), key=lambda kv: kv[1]["max_rel"])
    log(f"  preflight: worst layer {worst[0]} max_abs={worst[1]['max_abs']:.4g} "
        f"max_rel={worst[1]['max_rel']:.4g} min_cos={worst[1]['min_cos']:.6f}")
    if failed:
        raise RuntimeError(
            f"hook/capture mismatch at layer row(s) {failed[:10]} over prompts {checked}: "
            f"a NO-OP hook changed the prompt_last vector, or the stored vectors were "
            f"captured from a different module/position/template. Every downstream "
            f"steering number would be wrong. Details: "
            f"{ {l: per_layer[l] for l in failed[:5]} }")

    return {
        "passed": True,
        "model_type": type(model).__name__,
        "decoder_type": type(dec).__name__,
        "n_decoder_layers": n_blocks,
        "config_num_hidden_layers": cfg_blocks,
        "hidden_size": int(runner.hidden_size),
        "steerable_rows": [1, n_blocks - 1],
        "capture_point": point,
        "prompt_ids_checked": checked,
        "mode": "batched_capture_noise_aware" if batched_capture else "strict_fp16",
        "tolerances": ({"t1_max_rel": BATCHED_T1_MAX_REL, "min_cos": BATCHED_MIN_COS,
                        "max_rel_l2": BATCHED_MAX_REL_L2} if batched_capture
                       else {"atol": atol, "rtol": rtol, "min_cos": min_cos}),
        "worst_layer": {"layer": int(worst[0]), **{k: float(v) for k, v in worst[1].items()}},
        "per_layer": {str(l): {k: float(v) for k, v in s.items()}
                      for l, s in per_layer.items()},
    }


def write_answers_jsonl(path, records, mode: str = "a") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, mode, encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()


def read_done_ids(path) -> set[int]:
    p = Path(path)
    if not p.exists():
        return set()
    done: set[int] = set()
    with open(p, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            done.add(int(json.loads(line)["prompt_id"]))
        except Exception:
            if i == len(lines) - 1:
                print(f"  warning: dropping truncated final line of {p}")
            else:
                raise ValueError(f"{p}:{i + 1} is not a valid record")
    return done


class StaleArtifactError(RuntimeError):
    pass


SMOKE_SUFFIX = "_smoke"
SMOKE_PATH_KEYS = ("root", "generated_root", "study_root", "steered_root")


def apply_smoke_paths(cfg: dict, suffix: str = SMOKE_SUFFIX) -> dict:
    out = cfg.setdefault("output", {})
    for key in SMOKE_PATH_KEYS:
        if key in out and not str(out[key]).endswith(suffix):
            out[key] = str(out[key]) + suffix
    return cfg


def assert_study_splits_current(doc: dict, cfg_slugs, path="study_splits.json",
                                partition: str = "shared",
                                fractions: dict | None = None) -> None:
    want = sorted(str(s) for s in cfg_slugs)
    strat = doc.get("stratified_on")
    if not isinstance(strat, list) or not strat:
        raise StaleArtifactError(
            f"{path} has no 'stratified_on' list. It was not written by this pipeline's "
            f"generate_label_split.py; re-run `generate_label_split.py --stage split`.")

    if partition == "shared" and not str(strat[0]).startswith("label_triple"):
        raise StaleArtifactError(
            f"{path} is stratified on {strat[0]!r}, not the label_triple x "
            f"cstar_split rule.\n"
            f"  Re-run `generate_label_split.py --stage split` before any downstream "
            f"stage reads this file.")

    have = sorted(str(s) for s in (doc.get("per_model_class_counts") or {}))
    if have != want:
        raise StaleArtifactError(
            f"{path} was built over roster {have or '(none recorded)'}, not the "
            f"configured roster {want} (missing: {[s for s in want if s not in have]}, "
            f"extra: {[s for s in have if s not in want]}).\n"
            f"  The partition is stratified on the label triple ACROSS the roster, so "
            f"changing the roster invalidates it. Re-run generate_label_split.py's split stage.")

    if fractions is not None:
        got = {str(k): float(v) for k, v in (doc.get("fractions") or {}).items()}
        split_names = sorted(str(k) for k in (doc.get("splits") or {}))
        if (sorted(got) != sorted(fractions)
                or any(abs(got[k] - float(fractions[k])) > 1e-9 for k in fractions)
                or split_names != sorted(fractions)):
            raise StaleArtifactError(
                f"{path} was built with split fractions {got or '(none recorded)'} "
                f"(splits {split_names}), not the configured {dict(fractions)}.\n"
                f"  Re-run generate_label_split.py's split stage.")


def assert_selected_layers_current(doc: dict, criterion: str, top_k: int,
                                   study_sha: str | None, slugs=(),
                                   path="selected_layers.json") -> None:
    got = doc.get("criterion")
    if got is None:
        raise StaleArtifactError(
            f"{path} carries no 'criterion'; the expected criterion is "
            f"{criterion!r}. Re-run `directions_steering.py --stage probe`.")
    if str(got) != str(criterion):
        raise StaleArtifactError(
            f"{path} was selected by {got!r}, not the configured {criterion!r}. "
            f"Re-run --stage probe.")
    if int(doc.get("top_k", -1)) != int(top_k):
        raise StaleArtifactError(
            f"{path} holds top_k={doc.get('top_k')}, configured top_k={top_k}. "
            f"Re-run --stage probe.")
    if study_sha and doc.get("study_split_sha256") and \
            str(doc["study_split_sha256"]) != str(study_sha):
        raise StaleArtifactError(
            f"{path} was selected on study split "
            f"{str(doc['study_split_sha256'])[:16]} but the loaded partition is "
            f"{str(study_sha)[:16]}. The layers were chosen on a different Probing "
            f"split. Re-run --stage probe.")
    have = doc.get("selected") or {}
    missing = [s for s in slugs if s not in have]
    if missing:
        raise StaleArtifactError(
            f"{path} has no selected layers for {missing}; run --stage probe with the "
            f"full roster.")


def assert_locked_steering_config(cfg: dict) -> None:
    st = cfg.get("steering") or {}
    checks = [
        ("steering.positions", st.get("positions"), "prefill_last"),
        ("steering.output_format", st.get("output_format"), "answers_jsonl"),
        ("steering.save_activations", st.get("save_activations"), False),
        ("steering.static_batch_membership", st.get("static_batch_membership"), True),
    ]
    bad = [f"{k}={v!r} (only {exp!r} is implemented)"
           for k, v, exp in checks if v is not None and v != exp]

    names = (cfg.get("directions") or {}).get("names")
    dirs = st.get("directions")
    if names is not None and dirs is not None and list(names) != list(dirs):
        bad.append(f"directions.names={list(names)} != steering.directions={list(dirs)}; "
                   f"only steering.directions is read")

    probe_rows = (cfg.get("probe") or {}).get("steerable_rows_only")
    if probe_rows is False:
        bad.append("probe.steerable_rows_only=false, but rows 0 and n_blocks are always "
                   "excluded (row 0 is degenerate, row n_blocks is post-final-norm)")

    if bad:
        raise StaleArtifactError(
            "configs/study.yaml sets keys that describe LOCKED, hardcoded behaviour:\n  "
            + "\n  ".join(bad)
            + "\nThese are not switches. Change the code, or restore the locked values.")


def _selfcheck() -> None:
    from dataclasses import dataclass

    alphas = [-20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20, 2.5, -0.5]
    n = 0
    for layer in (1, 7, 24, 999):
        for d in DIRECTION_NAMES + ("rand",):
            for a in alphas:
                tag = condition_tag(layer, d, a)
                got = parse_condition_tag(tag)
                assert got["layer"] == layer, (tag, got)
                assert got["direction"] == d, (tag, got)
                assert float(got["alpha"]) == float(a), (tag, got)
                assert condition_tag(got["layer"], got["direction"], got["alpha"]) == tag
                n += 1
    assert condition_tag(24, "dR", -5) == "L024_dR_am5"
    assert condition_tag(24, "dD", 0) == "L024_dD_ap0"
    assert condition_tag(7, "dR", 2.5) == "L007_dR_ap2p5"
    assert parse_condition_tag(BASELINE_TAG)["alpha"] == 0.0 and is_baseline(BASELINE_TAG)
    assert sorted(["L100_dR_ap1", "L024_dR_ap1", "L007_dR_ap1"])[0] == "L007_dR_ap1"
    for bad in ("L24_dR_ap1", "L024_dR_a1", "L024_dR_ap1p", "garbage"):
        try:
            parse_condition_tag(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad!r} should not parse")
    print(f"[ok] condition tags round-trip ({n} tags) and sort correctly")

    class FakeBlock(torch.nn.Module):
        def forward(self, x):
            return (x, "kv-cache-placeholder")

    H, B, T = 4, 3, 5
    vec = np.array([1.0, 0.0, 0.0, 0.0])
    blk = FakeBlock()
    steer = PrefillLastTokenSteer(vec, 2.0, expect_hidden=H)
    handle = blk.register_forward_hook(steer)
    try:
        x = torch.zeros(B, T, H)
        out = blk(x)
        assert isinstance(out, tuple) and out[1] == "kv-cache-placeholder"
        h = out[0]
        assert h.shape == (B, T, H)
        assert torch.allclose(h[:, -1, :], torch.tensor([2.0, 0.0, 0.0, 0.0]).expand(B, H))
        assert torch.allclose(h[:, :-1, :], torch.zeros(B, T - 1, H)), "non-final positions moved"
        assert steer.n_fired == 1 and steer.n_decode_skipped == 0

        d = blk(torch.zeros(B, 1, H))
        assert torch.allclose(d[0], torch.zeros(B, 1, H))
        assert steer.n_fired == 1 and steer.n_decode_skipped == 1

        alpha0 = PrefillLastTokenSteer(vec, 0.0, expect_hidden=H)
        handle.remove()
        handle = blk.register_forward_hook(alpha0)
        y = blk(torch.ones(B, T, H))[0]
        assert torch.allclose(y, torch.ones(B, T, H)), "alpha=0 must be a no-op"
        assert alpha0.n_fired == 1
    finally:
        handle.remove()
    assert blk(torch.zeros(B, T, H))[0].abs().max().item() == 0.0, "hook leaked past remove()"

    plain = torch.nn.Identity()
    s3 = PrefillLastTokenSteer(vec, -1.0, expect_hidden=H)
    hp = plain.register_forward_hook(s3)
    try:
        z = plain(torch.zeros(1, 3, H))
        assert torch.is_tensor(z) and float(z[0, -1, 0]) == -1.0
    finally:
        hp.remove()
    print("[ok] PrefillLastTokenSteer: last prefill position only, tuple and tensor "
          "outputs, decode steps skipped, alpha=0 is a no-op, handle removal is clean")

    try:
        PrefillLastTokenSteer(vec, 1.0, expect_hidden=H + 1)
    except ValueError:
        pass
    else:
        raise AssertionError("hidden-size mismatch must raise")

    class RightPadTok:
        padding_side = "right"
    try:
        PrefillLastTokenSteer(vec, 1.0, tok=RightPadTok())
    except AssertionError:
        pass
    else:
        raise AssertionError("right padding must be rejected")
    print("[ok] guards: hidden-size mismatch and right padding both raise")

    @dataclass
    class Row:
        prompt_id: int
        prompt: str

    class FakeTok:
        padding_side = "left"
        pad_token_id = 0

        def __call__(self, text, add_special_tokens=True, **kw):
            ids = [1] + [ord(c) % 97 for c in text]
            return {"input_ids": ids}

    class FakeSpec:
        attn_implementation = "sdpa"

    class FakeRunner:
        spec = FakeSpec()
        tok = FakeTok()
        gen_kwargs = {"max_new_tokens": 1024, "do_sample": False}
        hidden_size = H

        def render(self, prompt):
            text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            ids = FakeTok()(text, add_special_tokens=False)["input_ids"]
            return text, {"input_ids": torch.tensor([ids])}

    rows = [Row(prompt_id=1000 + i, prompt="q" * (i % 17) + f" item {i}") for i in range(53)]
    quiet = lambda *a: None
    g_a = BatchedGenerator(FakeRunner(), batch_size=8, log=quiet)
    g_b = BatchedGenerator(FakeRunner(), batch_size=8, log=quiet)
    p_a = g_a.plan_batches(rows)
    p_b = g_b.plan_batches(rows)
    assert p_a == p_b, "batch plan differs between conditions"
    assert all(len(b) == 8 for b in p_a), [len(b) for b in p_a]
    assert sorted(set(i for b in p_a for i in b)) == list(range(len(rows))), "rows lost"
    assert g_a.n_padded_repeats == (8 - len(rows) % 8) % 8 == 3
    assert p_a == BatchedGenerator(FakeRunner(), batch_size=8, log=quiet).plan_batches(
        list(rows)), "plan is not a pure function of the rows"
    p1 = BatchedGenerator(FakeRunner(), batch_size=1, log=quiet).plan_batches(rows)
    assert len(p1) == len(rows) and all(len(b) == 1 for b in p1)
    print(f"[ok] plan_batches: static membership across alphas "
          f"({len(p_a)} batches of 8, {g_a.n_padded_repeats} padding repeats, "
          f"all {len(rows)} rows covered)")

    class EagerSpec:
        attn_implementation = "eager"

    class EagerRunner(FakeRunner):
        spec = EagerSpec()
    warned = []
    ge = BatchedGenerator(EagerRunner(), batch_size=8, log=warned.append)
    assert ge.batch_size == EAGER_MAX_BATCH and ge.max_prompt_tokens == EAGER_MAX_PROMPT_TOKENS
    assert any("WARNING" in w for w in warned)
    print(f"[ok] eager attention clamps batch 8 -> {ge.batch_size} with a "
          f"{ge.max_prompt_tokens}-token prompt cap, loudly")

    for bad_layer in (0, 32, 33, -1):
        try:
            assert_steerable(bad_layer, 32)
        except ValueError:
            pass
        else:
            raise AssertionError(f"layer {bad_layer} must be rejected for n_blocks=32")
    assert_steerable(1, 32)
    assert_steerable(31, 32)
    print("[ok] assert_steerable: rows 1..n_blocks-1 only (0 and n_blocks rejected)")

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "answers.jsonl"
        assert read_done_ids(f) == set()
        write_answers_jsonl(f, [{"prompt_id": 7, "answer": "café — ok"},
                                {"prompt_id": 9, "answer": "x"}])
        write_answers_jsonl(f, [{"prompt_id": 11, "answer": "y"}])
        assert read_done_ids(f) == {7, 9, 11}
        assert f.read_bytes().count(b"\r") == 0, "CRLF leaked into a jsonl"
        with open(f, "a", encoding="utf-8", newline="\n") as fh:
            fh.write('{"prompt_id": 13, "ans')
        assert read_done_ids(f) == {7, 9, 11}
    print("[ok] answers.jsonl: utf-8, LF-only, resume ids, truncated tail tolerated")

    assert abs(unit([3.0, 4.0])[0] - 0.6) < 1e-12
    try:
        unit([0.0, 0.0])
    except ValueError:
        pass
    else:
        raise AssertionError("unit() of a zero vector must raise")
    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, 8))
    d = unit(rng.normal(size=8))
    assert abs(sigma_along(X, d) - float(np.std(X @ d, ddof=1))) < 1e-12
    print("[ok] unit / sigma_along")

    per_model = 700 * 10 * 2 * 10 + 700
    total = 2 * per_model
    assert per_model == 140_700 and total == 281_400
    print(f"[ok] budget: {per_model:,}/model, {total:,} total")
    print("\nall self-checks passed")


if __name__ == "__main__":
    _selfcheck()
