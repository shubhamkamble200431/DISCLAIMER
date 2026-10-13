"""Generation + hidden-state capture for open-weight chat models.

For every unique prompt in the processed dataset this produces, per model:
  answers/<prompt_id>.json          the decoded answer + generation metadata
  hidden/<prompt_id>/<point>_layer_<NN>.npy
                                    residual-stream vector at every layer, one file each
                                    (output.hidden_layout: stacked gives one .npz instead)

The configured capture point is `prompt_last`: the final token of the fully
chat-formatted prompt, read from the prefill pass immediately before the model emits
its first response token. It is taken from hidden_states[0][l][0, -1, :] and so is
independent of anything that happens during generation.

Design notes
------------
* batch size is fixed at 1. Padding perturbs hidden states (and pooled means), and
  the whole point of this pass is clean activations, so we never batch.
* one forward pass: generate(output_hidden_states=True) returns the prefill states
  (all prompt positions) plus one state per generated token, for every layer.
* hidden_states[0][l]      -> prefill, [1, n_prompt_tokens, H]
  hidden_states[t][l], t>0 -> state AT generated token t, [1, 1, H]
  l runs 0..n_blocks, where l=0 is the embedding output and l=n_blocks the final
  hidden state after the last block (a 32-block model therefore gives 33 rows).
* generate() stops after emitting token N without ever forwarding it, so the state
  at the final generated token is missing from hidden_states. We recover it with one
  extra 1-token forward on the returned KV cache; without that, gen_last and gen_mean
  are silently off by one position. `gen_last_exact` in the record says whether that
  succeeded. Note the final token is usually EOS (`final_token_is_eos`) -- if you want
  the last *content* token instead, that is the position generate() itself reported.
* stored arrays are [n_layer_rows, H], float16 by default, automatically promoted
  to float32 if a value would overflow fp16 (some models carry massive outlier
  activations in late layers).
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
FP16_MAX = 65504.0

CAPTURE_POINTS = ("prompt_last", "prompt_mean", "gen_first", "gen_mean", "gen_last")


# --------------------------------------------------------------------------- #
# config / paths
# --------------------------------------------------------------------------- #
def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def repo_path(rel):
    p = Path(rel)
    return p if p.is_absolute() else REPO_ROOT / p


def slugify(s):
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")


def sha1(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# prompt table
# --------------------------------------------------------------------------- #
def build_prompt_table(cfg, splits=None):
    """Union of the requested split parquets, deduplicated to one row per prompt_id.

    If prompts.screen_set is set (the model-screening config), that committed parquet is
    used verbatim instead and the split names are ignored -- the screen set is its own
    frozen sample, carrying a `group` column of 'random' / 'enriched'.
    """
    pc = cfg["prompts"]
    id_col, text_col = pc["id_column"], pc["text_column"]

    if pc.get("screen_set"):
        tbl = pd.read_parquet(repo_path(pc["screen_set"]))
        tbl = tbl.rename(columns={id_col: "prompt_id", text_col: "prompt"})
        tbl["prompt"] = tbl["prompt"].astype(str)
        # `splits` is what the runner writes into answers.jsonl; carry the screen group
        # there so downstream code can tell the unbiased rows from the enriched ones.
        tbl["splits"] = "screen_" + tbl["group"].astype(str)
        tbl["prompt_sha1"] = [sha1(p) for p in tbl["prompt"]]
        if "n_chars" not in tbl:
            tbl["n_chars"] = tbl["prompt"].str.len()
        return (tbl[["prompt_id", "prompt", "splits", "prompt_sha1", "n_chars"]]
                .sort_values("prompt_id").reset_index(drop=True))

    wanted = list(splits) if splits else list(pc["splits"].keys())

    frames = []
    for name in wanted:
        if name not in pc["splits"]:
            raise KeyError(f"unknown split '{name}'; config has {sorted(pc['splits'])}")
        df = pd.read_parquet(repo_path(pc["splits"][name]), columns=[id_col, text_col])
        frames.append(df.assign(split=name))

    allrows = pd.concat(frames, ignore_index=True)
    allrows[text_col] = allrows[text_col].astype(str)

    # remember every split a prompt appeared in, then keep one row per prompt_id
    split_map = (allrows.groupby(id_col)["split"]
                 .agg(lambda s: ",".join(sorted(set(s)))).rename("splits"))
    tbl = (allrows.drop_duplicates(subset=[id_col])
           .drop(columns=["split"]).merge(split_map, on=id_col)
           .sort_values(id_col).reset_index(drop=True))
    tbl = tbl.rename(columns={id_col: "prompt_id", text_col: "prompt"})

    # a prompt_id must map to exactly one prompt string, otherwise it is not a key
    n_pairs = allrows.drop_duplicates(subset=[id_col, text_col]).shape[0]
    if n_pairs != len(tbl):
        raise ValueError(f"{n_pairs - len(tbl)} prompt_id(s) carry more than one prompt text")

    tbl["prompt_sha1"] = [sha1(p) for p in tbl["prompt"]]
    tbl["n_chars"] = tbl["prompt"].str.len()
    return tbl


# --------------------------------------------------------------------------- #
# capture helpers
# --------------------------------------------------------------------------- #
def _stack_layers(step_states, index):
    """[n_layer_rows, H] float32 cpu, taking position `index` from each layer tensor."""
    return torch.stack([s[0, index, :].detach().float().cpu() for s in step_states])


def _mean_layers(step_list):
    """Mean over per-step layer tuples -> [n_layer_rows, H] float32 cpu."""
    n_rows = len(step_list[0])
    acc = None
    for step in step_list:
        cur = torch.stack([step[l][0, -1, :].detach().float().cpu() for l in range(n_rows)])
        acc = cur if acc is None else acc + cur
    return acc / len(step_list)


def _to_storage(arr, dtype):
    """numpy array in `dtype`, promoted to float32 when fp16 cannot hold it."""
    a = arr.numpy() if isinstance(arr, torch.Tensor) else np.asarray(arr)
    if dtype == "float16" and (not np.isfinite(a).all() or np.abs(a).max() > FP16_MAX):
        return a.astype(np.float32), True
    return a.astype(dtype), False


def select_layers(n_rows, spec):
    """spec: 'all' | 'every:k' | list[int] -> sorted list of layer-row indices."""
    if spec in (None, "all"):
        return list(range(n_rows))
    if isinstance(spec, str) and spec.startswith("every:"):
        k = int(spec.split(":", 1)[1])
        idx = list(range(0, n_rows, k))
        if idx[-1] != n_rows - 1:
            idx.append(n_rows - 1)
        return idx
    return sorted({int(i) % n_rows for i in spec})


# --------------------------------------------------------------------------- #
# model spec
# --------------------------------------------------------------------------- #
@dataclass
class ModelSpec:
    name: str
    hf_id: str
    dtype: str = "bfloat16"
    device_map: str = "auto"
    attn_implementation: str | None = "sdpa"
    system_prompt: str | None = None
    chat_template_kwargs: dict = field(default_factory=dict)
    generation: dict = field(default_factory=dict)
    load_kwargs: dict = field(default_factory=dict)
    trust_remote_code: bool = False
    note: str | None = None

    @property
    def slug(self):
        return slugify(self.name or self.hf_id.split("/")[-1])


def parse_models(cfg, only=None):
    specs = [ModelSpec(**m) for m in cfg["models"]]
    if only:
        want = {slugify(x) for x in only} | set(only)
        picked = [s for s in specs if s.slug in want or s.name in want or s.hf_id in want]
        if not picked:
            raise SystemExit(f"--models matched nothing; available: {[s.slug for s in specs]}")
        specs = picked
    return specs


def preflight(spec):
    """Cheap check that the installed transformers can actually build this arch."""
    from transformers import AutoConfig
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
    cfg = AutoConfig.from_pretrained(spec.hf_id, trust_remote_code=spec.trust_remote_code)
    mt = getattr(cfg, "model_type", None)
    text_cfg = getattr(cfg, "text_config", cfg)
    return {
        "hf_id": spec.hf_id,
        "model_type": mt,
        "supported": bool(spec.trust_remote_code or mt in CONFIG_MAPPING_NAMES),
        "hidden_size": getattr(text_cfg, "hidden_size", None),
        "num_hidden_layers": getattr(text_cfg, "num_hidden_layers", None),
        "architectures": getattr(cfg, "architectures", None),
    }


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #
class Runner:
    """Loads one model and turns a prompt into (answer, per-layer activations)."""

    def __init__(self, spec: ModelSpec, gen_defaults: dict, capture: dict, log=print):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.spec, self.capture, self.log = spec, capture, log
        self.gen_kwargs = {**(gen_defaults or {}), **(spec.generation or {})}
        self.gen_kwargs.pop("seed", None)

        t0 = time.perf_counter()
        self.tok = AutoTokenizer.from_pretrained(
            spec.hf_id, trust_remote_code=spec.trust_remote_code)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

        kw = dict(dtype=getattr(torch, spec.dtype), device_map=spec.device_map,
                  trust_remote_code=spec.trust_remote_code, **(spec.load_kwargs or {}))
        if spec.attn_implementation:
            kw["attn_implementation"] = spec.attn_implementation
        try:
            self.model = AutoModelForCausalLM.from_pretrained(spec.hf_id, **kw)
        except (ValueError, KeyError) as e:
            self.log(f"  AutoModelForCausalLM refused ({type(e).__name__}: {e}); "
                     f"falling back to AutoModelForImageTextToText")
            from transformers import AutoModelForImageTextToText
            self.model = AutoModelForImageTextToText.from_pretrained(spec.hf_id, **kw)
        self.model.eval()

        # most instruct checkpoints ship sampling defaults in generation_config; under
        # greedy decoding transformers warns about each one, so clear them explicitly
        if self.gen_kwargs.get("do_sample") is False:
            for k in ("temperature", "top_p", "top_k", "min_p", "typical_p"):
                if getattr(self.model.generation_config, k, None) is not None:
                    setattr(self.model.generation_config, k, None)

        eos = self.model.generation_config.eos_token_id
        if eos is None:
            eos = self.tok.eos_token_id
        self._eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos]) - {None}

        tcfg = getattr(self.model.config, "text_config", self.model.config)
        self.hidden_size = int(tcfg.hidden_size)
        self.n_blocks = int(tcfg.num_hidden_layers)
        self.load_s = time.perf_counter() - t0
        self.log(f"  loaded {spec.hf_id} in {self.load_s:.1f}s | hidden={self.hidden_size} "
                 f"blocks={self.n_blocks} dtype={spec.dtype} device_map={spec.device_map}")

    # -- prompt formatting -------------------------------------------------- #
    def render(self, prompt):
        msgs = []
        if self.spec.system_prompt:
            msgs.append({"role": "system", "content": self.spec.system_prompt})
        msgs.append({"role": "user", "content": prompt})
        ctk = self.spec.chat_template_kwargs or {}
        try:
            enc = self.tok.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                return_tensors="pt", **ctk)
            text = self.tok.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False, **ctk)
        except Exception as e:                      # tokenizer carries no chat template
            self.log(f"  chat template unavailable ({type(e).__name__}), using raw prompt")
            text = prompt
            enc = self.tok(text, return_tensors="pt")
        keep = ("input_ids", "attention_mask")
        return text, {k: v for k, v in dict(enc).items() if k in keep}

    # -- one prompt --------------------------------------------------------- #
    @torch.inference_mode()
    def run(self, prompt):
        rendered, enc = self.render(prompt)
        device = next(self.model.parameters()).device
        enc = {k: v.to(device) for k, v in enc.items()}
        n_prompt = int(enc["input_ids"].shape[1])

        t0 = time.perf_counter()
        out = self.model.generate(**enc, output_hidden_states=True,
                                  return_dict_in_generate=True, use_cache=True,
                                  **self.gen_kwargs)
        gen_s = time.perf_counter() - t0

        gen_ids = out.sequences[0][n_prompt:]
        answer = self.tok.decode(gen_ids, skip_special_tokens=True)
        n_gen = int(gen_ids.shape[0])

        hs = out.hidden_states                      # tuple(steps) of tuple(layer rows)
        n_rows = len(hs[0])                         # n_blocks + 1
        prefill, gen_steps = hs[0], list(hs[1:])

        # generate() runs no forward pass at the position of the LAST generated token
        # (nothing has to be predicted from it), so that residual stream is absent from
        # out.hidden_states -- hs[-1] sits at token N-1. One extra 1-token forward on the
        # returned cache recovers the real final state; without it every gen_* point is
        # silently off by one.
        pkv, final_step = getattr(out, "past_key_values", None), None
        if n_gen > 0 and pkv is not None:
            try:
                cache_len = pkv.get_seq_length()
                fwd = self.model(
                    input_ids=out.sequences[:, -1:],
                    attention_mask=torch.ones((1, cache_len + 1), dtype=torch.long,
                                              device=device),
                    past_key_values=pkv, output_hidden_states=True, use_cache=True)
                final_step = fwd.hidden_states
                del fwd
            except Exception as e:
                self.log(f"    final-token forward failed ({type(e).__name__}); "
                         f"gen_last falls back to token N-1")

        gen_pos = gen_steps + ([final_step] if final_step is not None else [])

        want = set(self.capture.get("points") or CAPTURE_POINTS)
        vecs = {}
        if "prompt_last" in want:
            vecs["prompt_last"] = _stack_layers(prefill, -1)
        if "prompt_mean" in want:
            vecs["prompt_mean"] = torch.stack(
                [s[0].detach().float().mean(dim=0).cpu() for s in prefill])
        if gen_pos:
            if "gen_first" in want:
                vecs["gen_first"] = _stack_layers(gen_pos[0], -1)
            if "gen_last" in want:
                vecs["gen_last"] = _stack_layers(gen_pos[-1], -1)
            if "gen_mean" in want:
                vecs["gen_mean"] = _mean_layers(gen_pos)

        k = int(self.capture.get("last_k_prompt_tokens") or 0)
        if k > 0:
            take = min(k, n_prompt)
            vecs["prompt_tail"] = torch.stack(
                [s[0, -take:, :].detach().float().cpu() for s in prefill])

        layers = select_layers(n_rows, self.capture.get("layers", "all"))
        if len(layers) != n_rows:
            vecs = {name: v[layers] for name, v in vecs.items()}

        n_gen_positions = len(gen_pos)
        gen_last_exact = final_step is not None
        last_is_eos = bool(n_gen and int(gen_ids[-1]) in self._eos_ids)

        del out, hs, prefill, gen_steps, gen_pos, final_step, pkv
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        meta = {
            "n_prompt_tokens": n_prompt,
            "n_generated_tokens": n_gen,
            "hit_max_new_tokens": n_gen >= int(self.gen_kwargs.get("max_new_tokens", 0)),
            "n_layer_rows_total": n_rows,
            "layers_saved": layers,
            "hidden_size": self.hidden_size,
            "n_gen_positions_captured": n_gen_positions,
            "gen_last_exact": gen_last_exact,   # False -> gen_last sits at token N-1
            "final_token_is_eos": last_is_eos,
            "missing_gen_points": [] if n_gen_positions else
                                  sorted(want & {"gen_first", "gen_mean", "gen_last"}),
            "generate_seconds": round(gen_s, 3),
            "tokens_per_second": round(n_gen / gen_s, 2) if gen_s > 0 else None,
            "rendered_prompt": rendered,
        }
        return answer, vecs, meta


# --------------------------------------------------------------------------- #
# writers / readers
# --------------------------------------------------------------------------- #
def model_dirs(out_root, slug):
    root = repo_path(out_root) / slug
    return root, root / "answers", root / "hidden"


def hidden_target(hid_dir, pid, layout):
    """Where this prompt's activations live: a folder (per_layer) or a file (stacked)."""
    return hid_dir / (str(pid) if layout == "per_layer" else f"{pid}.npz")


def already_done(ans_dir, hid_dir, pid, layout="per_layer"):
    # the answer json is written last, so its presence means the hidden side finished
    return ((ans_dir / f"{pid}.json").exists()
            and hidden_target(hid_dir, pid, layout).exists())


def _write_hidden(hid_dir, pid, arrays, layers, layout, compress):
    """per_layer -> <pid>/<point>_layer_<NN>.npy, one file per layer per capture point.
       stacked   -> <pid>.npz holding [n_layer_rows, H] per point."""
    target = hidden_target(hid_dir, pid, layout)
    if layout == "stacked":
        saver = np.savez_compressed if compress else np.savez
        saver(target, layers=np.asarray(layers, np.int32), **arrays)
        return [target.name]

    target.mkdir(parents=True, exist_ok=True)
    written = []
    width = max(2, len(str(max(layers))))
    for name, a in arrays.items():
        for row, layer_ix in enumerate(layers):
            f = target / f"{name}_layer_{layer_ix:0{width}d}.npy"
            np.save(f, a[row])
            written.append(f.name)
    return written


def write_prompt(ans_dir, hid_dir, row, answer, vecs, meta, spec, store_dtype, compress,
                 layout="per_layer"):
    pid = int(row.prompt_id)
    arrays, promoted = {}, []
    for name, t in vecs.items():
        a, was = _to_storage(t, store_dtype)
        arrays[name] = a
        if was:
            promoted.append(name)

    layers = meta["layers_saved"]
    files = _write_hidden(hid_dir, pid, arrays, layers, layout, compress)

    record = {
        "prompt_id": pid,
        "prompt": row.prompt,
        "prompt_sha1": row.prompt_sha1,
        "splits": row.splits,
        "model": {"name": spec.name, "hf_id": spec.hf_id, "dtype": spec.dtype,
                  "system_prompt": spec.system_prompt,
                  "chat_template_kwargs": spec.chat_template_kwargs},
        "answer": answer,
        **{k: v for k, v in meta.items() if k != "rendered_prompt"},
        "rendered_prompt": meta["rendered_prompt"],
        "hidden_layout": layout,
        "hidden_path": (f"hidden/{pid}" if layout == "per_layer" else f"hidden/{pid}.npz"),
        "hidden_n_files": len(files),
        # shape of ONE layer file under per_layer; of the stacked array under stacked
        "hidden_shapes": {k: list(v.shape[1:] if layout == "per_layer" else v.shape)
                          for k, v in arrays.items()},
        "hidden_dtypes": {k: str(v.dtype) for k, v in arrays.items()},
        "fp16_overflow_promoted": promoted,
    }
    with open(ans_dir / f"{pid}.json", "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return record


def load_hidden(path, point="prompt_last", layer=None):
    """Read one prompt's activations, whichever layout was written.

    path   a hidden/<pid> folder (per_layer) or a hidden/<pid>.npz file (stacked)
    layer  None -> every layer stacked, [n_layer_rows, H]
           int  -> just that layer index, [H]

    Returns (array float32, layer index array).
    """
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob(f"{point}_layer_*.npy"))
        if not files:
            raise FileNotFoundError(f"no '{point}' layer files in {path}")
        layers = np.array([int(f.stem.rsplit("_", 1)[1]) for f in files], dtype=np.int32)
        if layer is not None:
            i = int(np.where(layers == layer)[0][0])
            return np.load(files[i]).astype(np.float32), layers[i:i + 1]
        return np.stack([np.load(f) for f in files]).astype(np.float32), layers

    with np.load(path) as z:
        arr, layers = z[point].astype(np.float32), z["layers"]
    if layer is not None:
        i = int(np.where(layers == layer)[0][0])
        return arr[i], layers[i:i + 1]
    return arr, layers


def load_layer_matrix(model_root, layer, point="prompt_last", prompt_ids=None):
    """Stack one layer across prompts -> ([n_prompts, H] float32, list of prompt_ids).

    This is the shape a per-layer probe wants. Under the per_layer layout it opens one
    small file per prompt, so expect it to be I/O bound on the first pass.
    """
    model_root = Path(model_root)
    hid = model_root / "hidden"
    ids = ([int(p) for p in prompt_ids] if prompt_ids is not None else
           sorted(int(p.stem if p.suffix == ".npz" else p.name) for p in hid.iterdir()))
    rows = [load_hidden(hidden_target(hid, pid, "per_layer") if (hid / str(pid)).is_dir()
                        else hid / f"{pid}.npz", point, layer)[0] for pid in ids]
    return np.stack(rows), ids
