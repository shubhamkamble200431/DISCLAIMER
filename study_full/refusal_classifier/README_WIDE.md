# Wide comparison — six more classifiers on one RTX 5060 Ti (16 GB, Linux)

Adds RoBERTa-large, DeBERTa-v3-large, BGE-M3, NeoBERT, GPT-2 medium and TinyLlama-1.1B
(LoRA) next to the existing ModernBERT-large and Longformer runs. Same data, same splits,
same training loop, effective batch 32, seed 42. Each classifier is scored on its own.

## Run

```bash
unzip refusal_wide.zip && cd refusal/refusal_classifier
bash scripts/setup_linux.sh          # skip if you already have a suitable env
source .venv/bin/activate
tmux new -s wide                     # or nohup; the full run takes hours
bash scripts/run_wide.sh
```

Needs internet on first run: about 9 GB of checkpoints download into the HuggingFace
cache (set `HF_HOME` to put them elsewhere).

Rough time on a 5060 Ti: **4–6 hours** for all six, estimated rather than measured.
The `check` stage logs measured rows/s and minutes per epoch for each model.

## What it does, per model

| stage | what | output |
|---|---|---|
| check | weights load, worst-case VRAM at full size, padding and save/reload parity | `results/wide/logs/check_<model>.log` |
| smoke | 200 rows, 1 epoch (README rule) | `results/runs/smoke_<model>_*` |
| train | LR sweep, early stopping on val DISCLAIMER F1 | `results/runs/<model>_42_*` |
| predict | val parity, test split, 5,250 generated answers | `results/predictions_wide/<model>/` |
| compare | all 8 classifiers side by side | `results/wide/comparison.md` |

A failure skips that model only. If a model runs out of VRAM, the script retries once
with half the micro-batch and twice the gradient accumulation (same effective batch).

- Progress: `tail -f results/wide/status.tsv`
- Subset: `bash scripts/run_wide.sh bge_m3 neobert`
- Resume later stages: `STAGES="predict compare" bash scripts/run_wide.sh`

## Send back

`results/wide/` (comparison + logs), plus `results/runs/<model>_42_*/{metrics.json,train.log}`
and `results/predictions_wide/`. The `best_state.pt` checkpoints (0.6–4.4 GB each) are
only needed if you want to re-run predictions elsewhere.

## Changes from the 2026-09-16 handoff (why DeBERTa and friends now train)

- **fp32 master weights.** transformers 5.x `from_pretrained` keeps a checkpoint's own
  dtype. DeBERTa-v3 ships fp16, so AdamW was updating fp16 weights, and every
  `deberta_v3*` run diverged. Models now always load in fp32; bf16 comes from autocast.
- **Remote-code buffers.** The same loader leaves NeoBERT's RoPE table zero-filled (no
  position information at all). It is now rebuilt, and `check_candidate.py` compares
  every non-checkpoint buffer against an eager build.
- **NeoBERT without xformers.** Its code imports `xformers.ops.SwiGLU`, which has no
  build for this torch/GPU. A pure-PyTorch module with the same weight names stands in;
  the check stage verifies the checkpoint loads with only the classifier head missing.
- **Decoders** (`kind: decoder`): EOS doubles as pad (checked absent from every text), the
  head reads the last real token, head starts at zero (GPT-2's huge final activations
  otherwise start the loss near 13), geometry uses the last token.
- **LoRA** (`train.lora`): adapters + full head; merged into `best_state.pt`, so
  `predict.py` loads it without peft.
- `predict.py` reports all-classifier and pairwise agreement for any number of runs.
