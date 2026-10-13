# Refusal classifier — training handoff

Classifies a model response into one of **three** classes:
`NORMAL` (0), `DISCLAIMER` (1), `REFUSAL` (2).

## Setup

```
pip install -r requirements.txt
```

`transformers>=5.17` is a **hard requirement**, not a preference. Version 5.2.0 ships a
DeBERTa-v3 tokenizer that silently adds no `[CLS]`/`[SEP]`, which degrades training with
no error. The script refuses to start on the broken version — if you see a
`check_special_tokens` error, upgrade transformers rather than working around it.

Install torch separately for your CUDA version. Every run prints `torch.__version__`,
`torch.version.cuda` and the device capability at startup.

The 7 encoder checkpoints (~7 GB total) download themselves from HuggingFace on first
use. All are ungated — no token needed. The dataset is NOT downloaded: it ships in
`data/processed/` (3,500 labelled responses, already split train/val/test).

## Run

Smoke-test first (200 rows, 1 epoch, ~1 min) — pre-registered rule, every candidate:

```
python -X utf8 scripts/train_transformer.py --config configs/modernbert.yaml --smoke
```

Then the real run:

```
python -X utf8 scripts/train_transformer.py --config configs/modernbert.yaml --seed 42
```

All seven, **strictly one at a time** — never two GPU jobs in parallel:

```
for c in frozen_embed modernbert modernbert_large deberta_v3 deberta_v3_large deberta_v3_headtail longformer; do
  python -X utf8 scripts/train_transformer.py --config configs/$c.yaml --seed 42
done
```

## What to send back

One `results/runs/<candidate>_42_<timestamp>/metrics.json` per candidate, plus
`train.log` from each. The headline numbers are `val_metrics.macro_f1` and
`val_metrics.disclaimer_f1`.

## Read the numbers before reporting them

A candidate reporting **macro-F1 ≈ 0.17 with DISCLAIMER F1 = 0.000** has collapsed to
predicting a single class — that is a failed run, not a weak model. It shows up as a
training loss stuck near 1.0986 (= ln 3, the chance floor). Flag those separately rather
than reporting them as scores. `deberta_v3` has done this at lr 2e-5; the config sweeps
1e-5 and 3e-5 too, so it may well train fine.

For reference, the non-neural baselines to beat: TF-IDF + logistic regression reaches
**0.807–0.833** macro-F1, frozen e5 embeddings **0.789**.

## Batch size

Every encoder candidate uses an **effective batch size of 32** (`batch_size * grad_accum`),
so the optimiser sees the same batch everywhere and the candidates are comparable. The
per-device `batch_size` still varies by model because their memory footprints differ --
that is a memory detail, not an optimisation one. On a large GPU you can raise
`batch_size` and divide `grad_accum` by the same factor for speed; the math is identical.

## Label geometry

Every run also reports the geometry of the learned representation, alongside the usual
classification metrics, in `metrics.json` under `geometry`:

- `pairs[<A>|<B>].euclidean` -- distance between the two class centroids
- `pairs[<A>|<B>].cosine` -- cosine similarity between the two class centroids
- `pairs[<A>|<B>].separation` -- centroid distance divided by the mean intra-class
  spread. **Read this one first.** Raw distance is not comparable across models, since
  each has its own embedding scale; the ratio is. Above 1 means the centroids sit further
  apart than the classes are wide.
- `per_label[<L>]` -- support, centroid norm, and intra-class spread for each class.

Vectors are the final-layer `[CLS]` state for fine-tuned encoders, and the masked
mean-pooled L2-normalised embedding for `frozen_embed`; the `pooling` field records which.
Do not compare raw distances between those two pooling schemes -- compare `separation`.

## Notes

- `--split-mode cv` is not implemented and will raise. Use the default (`holdout`).
- `--split-mode all` fits on all 3,500 rows with no validation split, for a final
  deployable model. It needs frozen hyperparameters (`--from-run <holdout run>`, or
  explicit `--lr`/`--epochs`) and its metrics are in-sample only.
- The `label_5` column in the parquets (technical/ethical detail) is unused by training.
