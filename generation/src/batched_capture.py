from __future__ import annotations

import torch

from src import steer as S


def prefill_last_states(runner, texts) -> torch.Tensor:
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

    def __init__(self, *a, on_record=None, **kw):
        super().__init__(*a, **kw)
        self.on_record = on_record

    def _forward(self, rows, chunk, bi, hook_ctx, shared, extra):
        texts = [self._render(rows[i])[0] for i in chunk]
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
