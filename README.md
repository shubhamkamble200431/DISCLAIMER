# DISCLAIMER

Code-only snapshot of the refusal/disclaimer activation-steering study: Python
scripts, source modules, and YAML configs, collected from
`~/workspace/refusal/` into one repo. No SLURM/shell launch
scripts, no logs, no data, no result files, no figures -- repo structure and
implementation only.

## Layout

- `generation/` -- main study pipeline (steps 1-10: response generation,
  labelling, probing, direction extraction, steering, geometric analysis,
  token-overlap analysis, random-direction control, custom-layer steering).
  See `generation/README_STUDY.md`.
- `classifier/` -- C* classifier training (the frozen Longformer reference
  classifier and the other candidates it was chosen against). See
  `classifier/README.md`.
- `screen/` -- the tier-1 refusal screening pass. See `screen/SCREEN_MANUAL.md`.
- `study_full/` -- an earlier/parallel snapshot of `generation/` and
  `classifier/`, kept separate rather than merged since the two snapshots
  differ (e.g. `study_full/refusal_classifier/configs/` has several
  classifier candidates -- `neobert`, `tinyllama_lora`, `bge_m3`,
  `roberta_large`, `gpt2_medium` -- not present in the current
  `classifier/configs/`).

Each project subfolder keeps its own internal `scripts/`, `src/`, `configs/`
layout from the source repo.

## What's deliberately excluded

SLURM/shell launch scripts (`scripts/slurm/*`, `*.sh`), all `data/` and
`results/` directories, logs, `__pycache__`, and every `.txt`/figure/results
artifact from the source repo.
