"""Batched step-1 generation + prompt_last capture (opt-in; batch 1 stays the default).

Step 1 originally ran one prompt per generate() call because padding can perturb the
captured hidden states. This module makes batching *checkable* instead of forbidden:
it left-pads (so index -1 is the real last token for every row), captures the prefill
state at that position for every layer, and scripts/parity_batched_capture.py compares
the result against batch-1 capture before the real run is allowed to use it.

The capture is a separate prefill-only generate(max_new_tokens=1, output_hidden_states=True)
so the ~1000 decode steps never hold all-layer hidden states in memory. Position ids
and masking are therefore whatever generate() derives from the attention mask, the same
as the batch-1 path.
"""
from __future__ import annotations

import torch

from src import steer as S


def prefill_last_states(runner, texts) -> torch.Tensor:
    """[B, n_layer_rows, H] float32 cpu: hidden state at the last real token of each prompt."""
    tok, model = runner.tok, runner.model
    prev_side = tok.padding_side
    tok.padding_side = "left"
    try:
        enc = tok(texts, padding=True, add_special_tokens=False, return_tensors="pt")
        device = next(model.parameters()).device
        enc = {k: v.to(device) for k, v in dict(enc).items()
               if k in ("input_ids", "attention_mask")}
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=1, do_sample=False,
                                 output_hidden_states=True, return_dict_in_generate=True,
                                 use_cache=True, pad_token_id=tok.pad_token_id)
        hs = out.hidden_states[0]
        vec = torch.stack([h[:, -1, :].detach().float().cpu() for h in hs], dim=1)
        del out, hs
        return vec
    finally:
        tok.padding_side = prev_side


def single_prompt_states(runner, prompt) -> torch.Tensor:
    """Batch-1 reference: exactly what Runner.run captures, prefill only. [n_rows, H]."""
    rendered, enc = runner.render(prompt)
    device = next(runner.model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.inference_mode():
        out = runner.model.generate(**enc, max_new_tokens=1, do_sample=False,
                                    output_hidden_states=True, return_dict_in_generate=True,
                                    use_cache=True, pad_token_id=runner.tok.pad_token_id)
    hs = out.hidden_states[0]
    vec = torch.stack([h[0, -1, :].detach().float().cpu() for h in hs])
    del out, hs
    return vec


class CapturingGenerator(S.BatchedGenerator):
    """BatchedGenerator that also hands each finished row's prompt_last vector to a callback.

    on_record(i, rec, vec, text): i indexes into `rows`, rec is the answer record
    _forward already builds, vec is [n_layer_rows, H] float32 cpu, text the rendered prompt.
    Rows repeated to pad a chunk are reported once.
    """

    def __init__(self, *a, on_record=None, **kw):
        super().__init__(*a, **kw)
        self.on_record = on_record

    def _forward(self, rows, chunk, bi, hook_ctx, shared, extra):
        texts = [self._render(rows[i])[0] for i in chunk]
        # capture first: on a long-prompt batch the all-layer prefill is the likeliest
        # OOM, and _run_batch halves the chunk and retries both passes.
        vec = prefill_last_states(self.runner, texts)
        recs = super()._forward(rows, chunk, bi, hook_ctx, shared, extra)
        seen = set()
        for k, i in enumerate(chunk):
            if i in seen:
                continue
            seen.add(i)
            if self.on_record is not None:
                self.on_record(i, recs[i], vec[k], texts[k])
        return recs
