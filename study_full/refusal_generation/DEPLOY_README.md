# Refusal / disclaimer steering study — DGX deployment

This archive is the complete, runnable study: the four step scripts, the single config, the
SLURM launchers, both subprojects' code, the processed prompt tables, and the frozen C\*
(Longformer) checkpoint. It contains **no generation-model weights** (those come from the
DGX HF cache) and **no activations** — everything else needed to reproduce steps 1–4 is here.

Every member is prefixed `refusal/`, so unpack it from `~/workspace`:

```bash
unzip -d ~/workspace /path/to/<archive>.zip
```

Depth lives in `refusal/refusal_generation/README_STUDY.md` — the long operating manual.
This file is the deployment front page and does not replace it.

---

## 1. Contents

| what | where (inside `refusal/`) | notes |
|---|---|---|
| the four steps | `refusal_generation/scripts/step{1,2,3,4}_*.py` | each organised as `--stage` |
| shared steering module | `refusal_generation/src/steer.py`, `src/generate.py` | hook + batched generator + the locked-config asserts |
| single config | `refusal_generation/configs/study.yaml` | roster, alpha grid, splits, C\* run dir. `generate.yaml` / `screen.yaml` are phase-1 leftovers |
| SLURM | `refusal_generation/scripts/slurm/{10..45}_*.slurm`, `run_study.sh`, `study_config.sh` | `study_config.sh` is **sourced, not run**; `0*`/`a5000_*`/`config.sh` are legacy phase-1 launchers |
| manuals | `refusal_generation/README_STUDY.md`, `MANUAL.md`, `SCREEN_MANUAL.md`, `plan.md` | |
| classifier code | `refusal_classifier/{src,scripts,configs}/` | the scoring stages import `src.train`, `src.metrics` and `scripts/predict.py` from here |
| data/processed | `refusal_generation/data/processed/*.parquet`, `refusal_classifier/data/processed/*.parquet` | train/val/test; this is C\*'s own split, **not** the 3-way study split |
| **C\* checkpoint** | `refusal_classifier/results/runs/longformer_42_20260916-160022/` | `best_state.pt` (567 MiB), `tokenizer/`, `config_snapshot.json`, `metrics.json`, `val_preds.npy`, `val_probs.npy` |

Not shipped, by construction: generation-model weights, activations (`data/generated/*/hidden/**`,
`*.npz`), phase-1 `answers.jsonl` from the discarded roster, and `data/study/*.json` (the retired
partition and layer selection — their frozen hashes no longer match the roster being run).
Steps 1–2 regenerate those on the DGX.

The code side of the archive is an allowlist re-checked against a deny list with an 8 MB
per-file cap; the C\* run directory is the one deliberate exception, added on top.

---

## 2. Unpack and verify

Container integrity first:

```bash
unzip -t /path/to/<archive>.zip
```

Then three checks that need **no GPU and no model download**. Run them from the generation repo:

```bash
cd ~/workspace/refusal/refusal_generation
```

**(a) The steering contract self-check.** Exercises the hook, static batch membership, the
eager clamp, `assert_steerable`, the resume reader and the budget arithmetic:

```bash
python -X utf8 src/steer.py
```

Expected tail: `[ok] budget: 140,700/model, 422,100 total, 2,106 GPU-h at batch-1 / 526 GPU-h
at batch-8 (5.5 days on 4x A100-40GB)` followed by `all self-checks passed`.

**(b) The config parses:**

```bash
python -X utf8 -c "import yaml,sys; c=yaml.safe_load(open('configs/study.yaml',encoding='utf-8')); print(len(c['models']),'models:',[m['name'] for m in c['models']])"
```

Expected: `3 models: ['qwen3.5-4b', 'qwen3.5-9b', 'olmo3-7b']` — the final roster, see
§8. (`qwen3.5-4b` / `qwen3.5-9b` slugify to the on-disk `qwen3-5-4b` / `qwen3-5-9b`.)

**(c) The plan-vs-inventory dry run.** Prints the locked 3-model budget and what this config
actually contains, side by side, and exits without doing work:

```bash
python -X utf8 scripts/step2_directions_steering.py --config configs/study.yaml --stage steer --dry-run
```

On a fresh unpack this stops with
`FATAL: no data/study/study_splits.json. Run step 1 (the split stage) first; step 2 never
invents a partition of its own.` **That is the correct result** — it proves the imports, the
config load, the per-model block/row geometry and the stale-artifact guard are all live. The
full `PLANNED BUDGET` / `ACTUAL INVENTORY` table appears once `12_step1_split_dgx.slurm` has run.

---

## 3. Environment split

The DGX has two conda envs and **one process cannot do both jobs**. This is why every script is
staged and why generation and C\*-scoring are separate SLURM jobs.

| stage | conda env | transformers | why |
|---|---|---|---|
| `step1 preflight`, `step1 generate` | `base` | 5.14.1 | loads roster models |
| `step1 label` | `refusal_clf` | 5.17.0 | imports the classifier repo's `src.train` / `src.metrics` |
| `step1 split` | `base` (either works) | — | pandas/numpy only |
| `step2 directions`, `probe`, `preflight`, `steer` | `base` | 5.14.1 | activations, hooks, generation |
| `step2 score` | `refusal_clf` | 5.17.0 | C\* |
| `step2 metrics` | `refusal_clf` | 5.17.0 | imports `M.LABELS3` / `M.MIN_N` rather than reimplementing them |
| `step3` (`compute`, `plot`) | `base` (either works) | — | numpy/matplotlib only |
| `step4 prepare`, `generate`, `steer` | `base` | 5.14.1 | download + generation |
| `step4 score`, `score-steered`, `metrics` | `refusal_clf` | 5.17.0 | C\* |

The working directory is `refusal_generation/` for **every** stage, including the `refusal_clf`
ones: the step scripts and `study.yaml` live there, and the classifier repo is reached by path
(`study.classifier_repo`). Do **not** put it on `PYTHONPATH` — both repos have a top-level `src`
package and they would collide. `predict.py` is never shelled out; it is exec'd under the module
name `cstar_predict`, and the run directory is always passed explicitly because its
`DEFAULT_RUNS` lists ModernBERT first.

Each `NN_*.slurm` exports the right `CONDA_ENV` and `exec`s `run_study.sh`, which does the
conda-activate, the banner, the torch/CUDA/transformers sanity print, and the python invoke.

---

## 4. sbatch sequence

```bash
cd ~/workspace/refusal/refusal_generation/scripts/slurm
```

### Step 1 — generate, label, split

| job | file | note |
|---|---|---|
| `refusal_study_gen_step1` | `10_step1_generate_dgx.slurm` | batch-1 (mandatory: activations are captured), 36 h, one GPU |
| `refusal_study_score_step1_label` | `11_step1_label_dgx.slurm` | `refusal_clf`; turns 3 x 1,750 answers into the label triple |
| `refusal_study_gen_step1_split` | `12_step1_split_dgx.slurm` | CPU, no `--gres`; writes the ONE shared 40/20/40 partition (no validation split) |

Submit one model per job so they run in parallel on separate GPUs:

```bash
sbatch --export=ALL,STUDY_MODELS=qwen3-5-4b 10_step1_generate_dgx.slurm
```

```bash
sbatch --export=ALL,STUDY_MODELS=qwen3-5-9b 10_step1_generate_dgx.slurm
```

```bash
sbatch --export=ALL,STUDY_MODELS=olmo3-7b 10_step1_generate_dgx.slurm
```

With `STUDY_MODELS` unset the whole roster runs sequentially in one job. Then:

```bash
sbatch 11_step1_label_dgx.slurm
```

```bash
sbatch 12_step1_split_dgx.slurm
```

### Step 2 — directions, layers, the gate, the sweep, scoring, metrics

| job | file | note |
|---|---|---|
| `refusal_study_gen_step2_dirs` | `20_step2_directions_dgx.slurm` | `d_R`, `d_D`, `sigma_l`; CPU, no `--gres` |
| `refusal_study_gen_step2_probe` | `21_step2_probe_dgx.slurm` | linear probe, top-10 layers per model; CPU |
| `refusal_study_gen_step2_preflight` | `22_step2_preflight_dgx.slurm` | **HARD GATE** — see below |
| `refusal_study_gen_step2_steer` | `23_step2_steer_dgx.slurm` | array `0-15%8`, 48 h/task, the 5.5-day job |
| `refusal_study_score_step2` | `24_step2_score_dgx.slurm` | `refusal_clf`; 422k answers, one checkpoint load |
| `refusal_study_score_step2_metrics` | `25_step2_metrics_dgx.slurm` | paired bootstrap + transition table and bar plots, CPU |

```bash
sbatch 20_step2_directions_dgx.slurm
```

```bash
sbatch 21_step2_probe_dgx.slurm
```

```bash
sbatch 22_step2_preflight_dgx.slurm
```

**`22_step2_preflight_dgx.slurm` is a hard gate.** It registers no-op (alpha=0) hooks on every
steerable layer, does a batch-1 forward, and compares `prompt_last` against the stored
`hidden/<pid>.npz` at every layer — proving the HF off-by-one is honoured (saved row `l` is the
output of `layers[l-1]`). Getting that wrong does not crash; it silently steers the wrong layer
for five days. The steer stage refuses to start without a passing preflight record unless
`--skip-preflight` is passed. **Do not pass it.** Read the job's output, confirm
`max_abs=0` / `min_cos=1.000000` at every steerable layer, and only then:

```bash
sbatch 23_step2_steer_dgx.slurm
```

```bash
sbatch 24_step2_score_dgx.slurm
```

```bash
sbatch 25_step2_metrics_dgx.slurm
```

Step 2 metrics writes, under `results/reports/step2/` next to `metrics.json`:

| file | content |
|---|---|
| `transitions_per_prompt.parquet` | one row per (model, steered condition, prompt) over the full Steering split; columns `prompt_id`, `model`, `layer`, `direction`, `alpha`, `alpha_over_sigma`, `before`, `after`, `changed`, `cstar_split`; `before` = the label at the shared `alpha=0` baseline |
| `transitions_table.parquet` + `transitions_table.csv` | long format, all 9 transition cells per condition and subset, with `count`, `denom`, `rate_pct`, `below_min_n` |
| `transitions_bar_<model>_<direction>_<subset>.png` | appendix: one panel per steered layer (all 10), x = alpha labelled with `alpha/sigma_l`, grouped bars for N->D, N->R, D->N, D->R, R->N, R->D; hatched = baseline n < 30 |
| `transitions_bar_main_<model>_<subset>.png` | **main-paper figure**: rows `dR` / `dD` x that direction's top 5 of the 10 steered layers, ranked by peak target-class pp (`dD` by DISCLAIMER pp, `dR` by REFUSAL pp) on the full Steering split; panels drawn exactly like the per-layer figure. Figure panels only: every table covers all 10 layers |
| `main_figure_layers.json` | the full per-direction ranking of all 10 steered layers by peak pp behind the main figure (peak pp, CI, alpha and alpha/sigma at the peak) |

`rate_pct = 100 x count / denom`, i.e. `T(i->j) = #(i->j) / #(i at baseline) x 100`, and
`below_min_n` flags a baseline n under `study.min_n` (30). `metrics.json` keeps its rates as
fractions 0-1 for compatibility; only the transition table is in percent.

### Step 3 — geometry (CPU-minutes)

`30_step3_geometry.slurm` (`refusal_study_gen_step3_geometry`) requests `--partition=gpu` with no
`--gres` only because no CPU partition has been confirmed on this cluster. If `sinfo` shows one,
change that line; otherwise skip sbatch entirely and run it on the login node:

```bash
bash run_study.sh step3 all
```

### Step 4 — Mozilla OOD transfer

| job | file | note |
|---|---|---|
| `refusal_study_gen_step4_prepare` | `40_step4_prepare_dgx.slurm` | needs outbound HTTPS to huggingface.co; pinned revision, 290 rows |
| `refusal_study_gen_step4_generate` | `41_step4_generate_dgx.slurm` | batch-1, OOD activations (for `sigma_l` only; `d_hat` stays frozen) |
| `refusal_study_gen_step4_steer` | `42_step4_steer_dgx.slurm` | array `0-7%8`; default = in-domain parity: each model's own top-10 layers x 2 directions x the full alpha grid |
| `refusal_study_score_step4` | `43_step4_score_dgx.slurm` | `refusal_clf`; scores the **unsteered** OOD answers (`--stage score`) |
| — | `44_step4_score_steered_dgx.slurm` | `refusal_clf`; labels the **steered** OOD answers (`--stage score-steered`) |
| `refusal_study_score_step4_metrics` | `45_step4_metrics_dgx.slurm` | sensitive and benign reported separately, never pooled |

```bash
sbatch 40_step4_prepare_dgx.slurm
```

```bash
sbatch 41_step4_generate_dgx.slurm
```

```bash
sbatch 42_step4_steer_dgx.slurm
```

**Step 4 steer defaults to in-domain parity**: each model's own top-10 selected layers
(`ood.layers_top: 10`) x the full steering alpha grid `[-20 .. 20]` (`ood.alphas: inherit`)
x `dR`/`dD`, so an OOD pp is read against the step-2 pp at the same layer and `alpha/sigma_l`.
That is **~8x the old step-4 default** (§7). `--reduced` gives the old cheap check
(`ood.reduced`: top-3 layers, alphas `[-10, -5, 5, 10]`); `--full-parity` is kept as a no-op
alias of the default. The steer stage takes `--shard i/n`, and `42_step4_steer_dgx.slurm` is an
**8-task array** (`--array=0-7%8`); its range must agree with `STUDY_OOD_SHARDS=8` in
`study_config.sh`, exactly as `23_`'s range must agree with `STUDY_STEER_SHARDS=16`. By hand:

```bash
bash run_study.sh step4 steer --shard 3/8
```

```bash
bash run_study.sh step4 steer --reduced
```

```bash
sbatch 43_step4_score_dgx.slurm
```

```bash
sbatch 44_step4_score_steered_dgx.slurm   # conda env refusal_clf; label the STEERED answers
```

```bash
sbatch 45_step4_metrics_dgx.slurm
```

Anything can also be driven by hand without sbatch, e.g.:

```bash
CONDA_ENV=refusal_clf bash run_study.sh step2 score
```

---

## 5. Monitoring progress

Every job writes a Tee log. For a one-screen summary of how far each stage has got — no package,
no account, no network, read-only — run from `refusal_generation/` on the login node:

```bash
bash scripts/study_status.sh
```

`--smoke` reads the `_smoke` output roots, `--watch N` re-renders every N seconds, and
`--tail N` sets how many lines of the most recent stage log to show.

---

## 6. Before the first real job

1. **Preflight the roster** (final, three models, §8; reads `config.json` only, no weights
   downloaded):

   ```bash
   python -X utf8 scripts/step1_generate_label_split.py --config configs/study.yaml --stage preflight
   ```

   Any later roster change alters the label-triple strata, so **step 1's `split` stage must be
   re-run** and steps 2–4 re-run, not patched.

2. **Run the preflight gate** (`22_step2_preflight_dgx.slurm`) and read its output before
   `23_step2_steer_dgx.slurm`.

3. **Check the queue before planning wall-clock.** All six legacy phase-1 launchers used
   `--time=UNLIMITED`; Slurm cannot backfill a job with no declared limit, so such a job waits
   for a full-width drain. Every job in this study family declares a real limit instead. Eight
   concurrent array tasks at `--mem=96G` is 768 GB of host RAM — confirm the node and the QOS
   can hold it, and lower the `%8` throttle if not.

   ```bash
   sinfo -o "%P %m %G %D %t"
   ```

   ```bash
   sacctmgr show qos
   ```

   ```bash
   squeue -u "$USER"
   ```

4. **Four wrapper defects were found and are ALREADY FIXED in this archive.** Listed so you can
   recognise them if you compare against an older copy of the tree:

   | file | was | now |
   |---|---|---|
   | `23_step2_steer_dgx.slurm` | `--shard N --n-shards 16`; `step2` has no `--n-shards` and its `--shard` takes one `i/n` argument | `--shard "$((SHARD + 1))/$STUDY_STEER_SHARDS"` |
   | `23_step2_steer_dgx.slurm` | `SLURM_ARRAY_TASK_ID` is 0-based but `--shard` validates `1 <= i <= n`, so shard 0 aborted | the `+ 1` above |
   | `30_step3_geometry.slurm` | `--stage geometry`; step 3's stages are `compute`, `plot`, `all` | `step3 all` |
   | *(missing job)* | nothing ran `step4 --stage score-steered`, so `metrics` silently reported "no conditions" over unscored steered answers | new `44_step4_score_steered_dgx.slurm`; metrics renumbered to `45_` |

   `README_STUDY.md` has been updated to match.

---

## 7. Budget

Measured anchor: **17.96 s/generation** at batch-1, 1024-token cap, one A100-40GB. Batch-8 is
assumed to buy ~4x.

| item | per model | x3 models |
|---|---|---|
| steering prompts | 700 | 2,100 |
| conditions (10 layers x 2 directions x 10 non-zero alphas) | 200 | 600 |
| steered generations | 140,000 | 420,000 |
| shared `alpha=0` baseline | 700 | 2,100 |
| **total generations** | **140,700** | **422,100** |
| GPU-h at batch-1 | 702 | 2,106 |
| **GPU-h at batch-8** | **176** | **526** |

**~527 GPU-h ≈ 5.5 days on 4x A100-40GB.** The whole study, stage by stage (batch-8 is the ~4x
assumption, unmeasured):

| stage | generations | GPU-h batch-1 | GPU-h batch-8 |
|---|---|---|---|
| step 1 generate (batch-1 mandatory: activations captured) | 5,250 | ~26 | — |
| step 2 steer | 422,100 (140,700/model) | ~2,106 | ~527 |
| step 4 generate (batch-1 mandatory) | 870 | ~4.3 | — |
| **step 4 steer, default** | **174,870 (58,290/model)** | **~872** | **~218** |
| step 4 steer, `--reduced` | 21,750 | ~108 | ~27 |

Step 4's default (290 prompts x 10 layers x 2 directions x 10 non-zero alphas + the 290-prompt
baseline, per model) is **~8x its old default**, which is what `--reduced` now gives; it takes
steps 1+2+4 from ~585 to ~775 GPU-h. C\*-scoring all 422k step-2 answers ≈ 3 GPU-h, and steps 3
and the bootstraps are CPU-minutes. Do not widen the alpha grid — it is the
multiplier on everything above. `--dry-run` on any stage prints its own estimate; trust that
over this table if they ever disagree.

---

## 8. Locked decisions

Set by the user. The code does not re-litigate them and neither should a reviewer's patch.

- **C\* = Longformer only**, run dir `refusal_classifier/results/runs/longformer_42_20260916-160022`,
  `max_length 1024`, `attention_window 512`, truncation `head`. Always passed explicitly.
- **Roster of 3 generation models, one per family, final:**

  | slug | `hf_id` | family |
  |---|---|---|
  | `qwen3-5-4b` | `Qwen/Qwen3.5-4B` | Qwen (`enable_thinking: false`) |
  | `qwen3-5-9b` | `Qwen/Qwen3.5-9B` | Qwen (`enable_thinking: false`) |
  | `olmo3-7b` | `allenai/Olmo-3-7B-Instruct` | OLMo |

  Set 2026-09-22: Nemotron is out, Qwen3.5-4B is back. 4B vs 9B is a **scale** pair within
  one family; OLMo-3 is the only cross-family contrast.

- **System-prompt rule, one rule for all three:** `system_prompt: null`, and each model's
  chat-template default applies. Measured 2026-09-21: qwen has no system turn with
  `enable_thinking=false`;
  the olmo3 template injects `You are a helpful function-calling AI assistant. You do not
  currently have access to any functions. <functions></functions>`.
  **OLMo-3 therefore runs under a default persona the other two never see**, which may raise
  technical refusals ("I don't have access to ...") — report it as a caveat. Overriding it means
  setting an explicit, non-empty `system_prompt` on that entry; **an empty string does NOT work**
  (`src/generate.py` adds a system turn only when the value is truthy), and any override means
  re-running step 1 onward.
- **Exactly two directions**: `d_R = unit(mu_R - mu_N)`, `d_D = unit(mu_D - mu_N)`. No `d_D_perp`.
- **Balanced difference-of-means**: subsample NORMAL to `min(n_R, n_D)`, K=50 seeded draws,
  average the **unit** vectors, renormalise.
- **Layer selection**: top 10 per model by `mean(AUROC(R vs rest), AUROC(D vs rest))` from a
  linear probe on the 20% Probing split.
- **Steering position**: additive injection at the **last prefill token only**, before the first
  generated token. Generated tokens are not steered; the effect propagates via the KV cache.
- **Alpha grid** `[-20,-10,-5,-2,-1,0,1,2,5,10,20]`. `alpha=0` is **one shared baseline per model**,
  generated through the identical batched path with identical batch membership.
- **Every plot and table uses `alpha / sigma_l`**, never raw alpha.
- **`max_new_tokens = 1024`, greedy.** Batch size 8; requires `sdpa` or `flash_attention_2`
  (eager clamps to 4).
- **One shared prompt partition** 40/20/40 (Direction / Probing / Steering), **no validation
  split**, stratified on the label triple crossed with C\*'s own train/val/test split. Per-model class counts inside a split differ by construction, and
  every output says so.
- **Bootstrap**: paired over prompt ids, B=10,000, percentile 95% CI, one shared resample index.
- **Random-direction control**: implemented, defaults OFF, generates nothing.
- **Step 3 geometry uses the full Direction split** (no cross-fitting), compared against a
  label-permutation null band built with the same estimator.
- **Activations** written as `hidden_layout: stacked` — one `.npz` per prompt.
- **Steered output is `answers.jsonl` only**, never per-prompt JSON.
- **All metrics are defined on the argmax label only**: dynamic batch composition in the
  classifier shifts probabilities by up to 0.023.

---

## 9. Smoke test

`--smoke` swaps in `allenai/OLMo-2-0425-1B-Instruct`, 256 prompts, a 128-token cap, alphas
`[-5, 0, 5]` and `top_k: 2`, and redirects **every** output root to a parallel `_smoke` tree, so
it can never touch real artifacts. One command runs all four steps:

```bash
for s in step1_generate_label_split step2_directions_steering step3_geometry step4_ood_mozilla; do python -X utf8 scripts/$s.py --config configs/study.yaml --stage all --smoke || break; done
```

It proves the full chain: generation and activation capture, C\* labelling and the val-parity
gate, the shared partition, direction estimation, probe-based layer selection, the hook/capture
parity gate, batched steered generation, scoring, the paired bootstrap, the geometry pass, and
the Mozilla OOD path.

Measured end to end on a laptop RTX 5070, 2026-09-20, all four steps passed:

| check | result |
|---|---|
| C\* val parity | 1.0000, max abs prob delta 0.0230 |
| label mix | 179 NORMAL / 40 DISCLAIMER / 37 REFUSAL |
| splits (direction/probing/steering), 40/20/40, re-run 2026-09-22 | 104 / 52 / 100 |
| step-2 hook preflight | `max_abs=0`, `min_cos=1.000000` at all 15 steerable layers |
| causal effect | `d_R` at alpha=+5, layer 10: **+11.8 pp REFUSAL**, 95% CI [+4.9, +18.6] |

---

## 10. Troubleshooting

**`attn_implementation: eager` warning, batch clamped to 4.** `S.BatchedGenerator` refuses to
trust eager attention at batch 8 — eager materialises the full `[B, H, L, L]` score tensor. It
clamps to batch 4 with a 2048-token prompt cap and says so loudly. That is a 2x budget hit;
prefer a model with `sdpa` or `flash_attention_2` (this is why Gemma was discarded). If you must
keep an eager model, re-run `--dry-run` for the real GPU-h before submitting.

**CUDA OOM during the array.** `generate()` halves the batch and retries, logging it;
`.n_oom_halvings` is in the record stream. If it halves repeatedly, resubmit that shard with
`--batch-size 4` (or 2) rather than letting it thrash. **Changing the batch size changes batch
membership, and batched greedy decoding is not bitwise invariant to batch size** — so if you
re-run a condition at a different batch size, re-run its `alpha=0` baseline the same way or the
comparison is no longer paired. `plan_batches()` is called once per model and the same plan is
reused for every condition including the baseline; that invariant is what makes the grid
comparable.

**Resume.** Everything is resume-safe. Step 1 skips prompts that already have
`hidden/<pid>.npz`; the steering stages skip prompt ids already in the condition's
`answers.jsonl` (the reader tolerates a truncated final line, which is exactly what a job killed
at its `--time` limit leaves behind); scoring skips parquet shards already written. The correct
response to a job hitting its wall-clock limit is to `sbatch` it again — **never `rm -rf` the
output directory.**

**A stale-artifact refusal.** The scripts refuse to run against a `study_splits.json` or
`selected_layers.json` from a superseded design or roster, e.g.:

- `study_splits.json is stratified on 'majority_label ...', not the locked label_triple x
  cstar_split rule` — that is `make_study_splits.py`'s superseded partition.
- `study_splits.json was built over roster [...], which does not cover the configured roster` —
  the roster changed; the partition is stratified across it.
- `selected_layers.json carries no 'criterion'` — that is `layer_probe.py`'s superseded selection.
- `selected_layers.json was selected on study split <sha> but the loaded partition is <sha>`.

**The fix is always to re-run the stage that writes the file** (`step1 --stage split`,
`step2 --stage probe`), never to delete the guard or hand-edit the hash. These are not warnings;
they are the difference between a five-day run that measures something and one that does not.

**`--verify` fails.** That is drift, not noise: an input changed under a committed artifact.
Every artifact carries its seed, an input inventory and a sha256. Find the input, decide whether
the change was intended, and re-run the stage. Do not regenerate the hash to make the check pass.

**`ImportError: cannot import name 'train' from 'src'`.** Both repos ship a top-level `src`
package. The step script owns the `sys.path` ordering and exec's `predict.py` under a distinct
module name. Do **not** "fix" this by exporting `PYTHONPATH=.../refusal_classifier`.

---

## 11. Provenance

Built from the working tree at `D:/IITIndore/refusal/refusal_intern/` on 2026-09-21: the
`refusal_study.zip` allowlist (93 code/config/data members, 4.81 MB uncompressed, `unzip -t`
clean, per-member sha256 in `refusal_study.manifest.json`) combined with the 8-file C\* run
directory `longformer_42_20260916-160022`. Smoke-tested end to end on 2026-09-20.

**`longformer_ckpt.zip` at the old repo root is CORRUPT** — no end-of-central-directory, i.e. a
truncated upload. It must never be used. The valid checkpoint is the working-tree run directory,
which this archive carries.

**Correction to `plan.md` §1.13.** It describes `Mozilla/sensitive-topic-disclaimer-eval` as
"17 sensitive topics (170 rows) and 12 benign control topics (120 rows)". That is **wrong**:
`is_sensitive` is a per-**row** property, not per-topic, and 13 of the 29 topics are mixed. The
real split is **126 sensitive / 164 benign** rows out of 290. `step4_ood_mozilla.py` pins the
measured values and says so at runtime, and `README_STUDY.md` §1 now carries them; the step-4
SLURM comments may still carry the old 170/120 figures.
