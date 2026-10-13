# Refusal / disclaimer direction + causal steering study

Four steps, one config (`configs/study.yaml`), one shared prompt partition, one frozen
classifier. This file is the operating manual: what each step produces, the exact command
sequence on a laptop and on the DGX, which conda env each stage needs, the budget, and the
decisions and measured facts that the code is not allowed to drift from.

Companion files: `../../plan.md` (the analysis this is built from), `MANUAL.md` (the
unsteered generation runner), `src/steer.py` (the shared steering/batching contract —
`python -X utf8 src/steer.py` runs its self-checks with no GPU and no download).

---

## 1. The four steps

| step | script | stages (`--stage`) | produces |
|---|---|---|---|
| 1 | `scripts/step1_generate_label_split.py` | `generate` · `label` · `split` | `data/generated/<slug>/answers.jsonl` + `hidden/<pid>.npz` (prompt_last, all rows); C\* labels per model; `data/study/splits.parquet` — the ONE shared 3-way 40/20/40 partition (no validation split) |
| 2 | `scripts/step2_directions_steering.py` | `directions` · `probe` · `preflight` · `steer` · `score` · `metrics` | `d_R`, `d_D` and `sigma_l` per model/layer; top-10 layers per model; the hook/capture parity manifest; `data/steered/<slug>/<condition>/answers.jsonl`; scored parquet shards; pp curves with paired-bootstrap CIs; the transition outputs listed below |
| 3 | `scripts/step3_geometry.py` | `geometry` | `S_l = cos(d_D_l, d_R_l)` per layer on the full Direction split (no cross-fitting) with its permutation null band, `cos(d_*, d_len)`, `\|mu_D-mu_N\| / \|mu_R-mu_N\|` |
| 4 | `scripts/step4_ood_mozilla.py` | `prepare` · `generate` · `steer` · `score` · `metrics` | Mozilla OOD prompt table (290 rows, pinned revision), OOD activations, steered answers on the in-domain-parity grid (each model's own top-10 layers x the full alpha grid; `--reduced` = the old top-3 x 4-alpha check), transfer metrics reported separately for the 126 sensitive and 164 benign rows (measured per row; plan.md 1.13 says 170/120, which is wrong) |

**Step-2 transition outputs** (written by `--stage metrics` under `results/reports/step2/`,
next to `metrics.json`). They exist because a net pp curve hides churn: +5 pp REFUSAL can be
+10 pp N->R and -5 pp R->N at once.

| file | content |
|---|---|
| `transitions_per_prompt.parquet` | one row per (model, steered condition, prompt) over the full Steering split; columns `prompt_id`, `model`, `layer`, `direction`, `alpha`, `alpha_over_sigma`, `before`, `after`, `changed`, `cstar_split`. `before` is the label at the shared `alpha=0` baseline |
| `transitions_table.parquet` + `transitions_table.csv` | long format, all 9 transition cells (N/D/R x N/D/R) per condition and subset, with `count`, `denom`, `rate_pct` and `below_min_n` |
| `transitions_bar_<model>_<direction>_<subset>.png` | appendix: one panel per steered layer (all 10); x = alpha, tick-labelled with `alpha/sigma_l`; grouped bars for N->D, N->R, D->N, D->R, R->N, R->D; a hatched bar means baseline n < 30 |
| `transitions_bar_main_<model>_<subset>.png` | **main-paper figure**: rows `dR` / `dD` x that direction's top 5 of the 10 steered layers, ranked by peak target-class pp (`dD` by DISCLAIMER pp, `dR` by REFUSAL pp) on the full Steering split; panels drawn exactly like the per-layer figure. Figure panels only: every table covers all 10 layers |
| `main_figure_layers.json` | the full per-direction ranking of all 10 steered layers by peak pp behind the main figure (peak pp, CI, alpha and alpha/sigma at the peak) |

`rate_pct = 100 x count / denom`, where `denom` is the number of prompts with label `i` at
the baseline, i.e. `T(i->j) = #(i->j) / #(i at baseline) x 100`. `below_min_n` is true when
that baseline n is under `study.min_n` (30); the number is still written, but flagged, never
silently dropped. `metrics.json` keeps its rates as **fractions 0-1** for compatibility with
existing readers — only the transition table is in percent.

Every script also accepts `--stage all` (laptop), `--dry-run` (prints cell counts,
estimated generations and estimated GPU-h, then exits), `--limit N` (smoke), `--verify`
(recompute the committed sha256 and exit non-zero on drift) and `--config`.

There is no held-out validation split any more (user decision 2026-09-22): the three study
splits cover all 1,750 prompts.

---

## 2. Environment split (why every script has `--stage`)

The DGX has two conda envs and **one process cannot do both jobs**:

| stage | conda env | transformers | why |
|---|---|---|---|
| `generate`, `directions`, `probe`, `preflight`, `steer` | `base` | 5.14.1 | the generation stack; loads the roster models |
| `label`, `score`, `metrics` | `refusal_clf` | 5.17.0 | `predict.py` imports the classifier repo's `src.train` / `src.metrics`, which need 5.17 |
| `split`, `geometry` | either (`base` used) | — | pandas/numpy only |

That split is the whole reason the GPU stages and the C\*-scoring stages are separate
`--stage` subcommands and separate SLURM jobs. The job families are named for it:
`refusal_study_gen_*` and `refusal_study_score_*`.

The working directory stays `refusal_generation/` for **every** stage, including the
`refusal_clf` ones: the step scripts and `configs/study.yaml` live here, and the classifier
repo is reached by path (`study.classifier_repo`), not by `cd`. It is deliberately **not**
put on `PYTHONPATH` — both repos have a top-level `src` package and they would collide.

---

## 3. Command sequence

### 3.1 Laptop (smoke, no SLURM)

`--smoke` swaps in the tiny roster from `study.yaml`'s `smoke:` block (one ungated 1.5B
model, 8 prompts, 24 new tokens, alphas `[-5, 0, 5]`, batch 2). Run the whole pipeline end
to end before queueing anything.

```bash
cd D:/IITIndore/refusal/refusal_intern/refusal_generation
PY=D:/IITIndore/refusal/.venv-gen/Scripts/python.exe

$PY -X utf8 src/steer.py                                     # contract self-checks, no GPU

$PY -X utf8 scripts/step1_generate_label_split.py --config configs/study.yaml --stage all --smoke
$PY -X utf8 scripts/step2_directions_steering.py  --config configs/study.yaml --stage all --smoke
$PY -X utf8 scripts/step3_geometry.py             --config configs/study.yaml --stage all --smoke
$PY -X utf8 scripts/step4_ood_mozilla.py          --config configs/study.yaml --stage all --smoke
```

Plan-only, no work done (prints cells, generations, GPU-h):

```bash
$PY -X utf8 scripts/step2_directions_steering.py --config configs/study.yaml --stage steer --dry-run
```

Re-verify every committed artifact:

```bash
for s in step1_generate_label_split step2_directions_steering step3_geometry step4_ood_mozilla; do
  $PY -X utf8 scripts/$s.py --config configs/study.yaml --stage all --verify
done
```

### 3.2 DGX (SLURM)

All launchers live in `scripts/slurm/`. `study_config.sh` is **sourced, not run**;
`run_study.sh <SCRIPT> <STAGE> [args]` is the single place that does conda-activate + banner
+ torch/transformers sanity check + the python invoke; each `NN_*.slurm` is a thin
`exec bash` wrapper that exports the right `CONDA_ENV` first.

```bash
cd ~/workspace/refusal/refusal_generation/scripts/slurm

# ---- step 1 -------------------------------------------------------------
# One model per submission runs them in parallel on separate GPUs:
sbatch --export=ALL,STUDY_MODELS=qwen3-5-4b 10_step1_generate_dgx.slurm
sbatch --export=ALL,STUDY_MODELS=qwen3-5-9b           10_step1_generate_dgx.slurm
sbatch --export=ALL,STUDY_MODELS=olmo3-7b             10_step1_generate_dgx.slurm
# With STUDY_MODELS unset the whole roster runs sequentially in one job.
sbatch 11_step1_label_dgx.slurm        # conda env refusal_clf
sbatch 12_step1_split_dgx.slurm        # CPU; writes the ONE shared partition

# ---- step 2 -------------------------------------------------------------
sbatch 20_step2_directions_dgx.slurm   # d_R, d_D, sigma_l
sbatch 21_step2_probe_dgx.slurm        # top-10 layers per model
sbatch 22_step2_preflight_dgx.slurm    # hook/capture parity gate -- MUST pass first
sbatch 23_step2_steer_dgx.slurm        # ARRAY 0-15%8, the 5.5-day job
sbatch 24_step2_score_dgx.slurm        # conda env refusal_clf
sbatch 25_step2_metrics_dgx.slurm      # paired bootstrap + transitions table/bars, CPU

# ---- step 3 -------------------------------------------------------------
sbatch 30_step3_geometry.slurm         # or just: bash run_study.sh step3 all

# ---- step 4 -------------------------------------------------------------
sbatch 40_step4_prepare_dgx.slurm      # needs outbound HTTPS to huggingface.co
sbatch 41_step4_generate_dgx.slurm     # batch-1, OOD activations
sbatch 42_step4_steer_dgx.slurm        # ARRAY 0-7%8, default in-domain-parity grid
sbatch 43_step4_score_dgx.slurm        # conda env refusal_clf
sbatch 44_step4_score_steered_dgx.slurm   # conda env refusal_clf; label the STEERED answers
sbatch 45_step4_metrics_dgx.slurm      # conda env refusal_clf; sensitive / benign separately
```

Anything can also be driven by hand without sbatch:

```bash
CONDA_ENV=refusal_clf bash run_study.sh step2 score
bash run_study.sh step2 steer --shard 4/16
bash run_study.sh step4 steer --shard 3/8
bash run_study.sh step4 steer --reduced            # the old cheap check, one process
```

**Step 4 grid and sharding.** `step4 --stage steer` now **defaults to in-domain parity**:
each model's own top-10 selected layers (`ood.layers_top: 10`) x the full steering alpha grid
(`ood.alphas: inherit` = `steering.alphas`, `[-20 .. 20]`) x `dR`/`dD`, so an OOD pp can be
read against the step-2 pp at the same layer and the same `alpha/sigma_l`. That is ~8x the
old default. `--reduced` restores the old cheap check (`ood.reduced`: top-3 layers, alphas
`[-10, -5, 5, 10]`); `--full-parity` is kept as a no-op alias of the default so old command
lines still run. The steer stage takes `--shard i/n` (one argument, `1 <= i <= n`, as in
step 2), and `42_step4_steer_dgx.slurm` is an **8-task array** (`--array=0-7%8`); its range
and `STUDY_OOD_SHARDS=8` in `study_config.sh` must agree, exactly like `23_`'s range and
`STUDY_STEER_SHARDS`.

**Do not chain these blindly.** `22_step2_preflight` is a gate: if it fails, the hook is on
the wrong module and the 5.5-day array would steer the wrong layer without crashing.

**Watching a multi-day job.** `bash scripts/study_status.sh` prints a one-screen progress
report from the Tee logs and the output directories; it needs no package and no network.
See **`DEPLOY_README.md` §5**.

#### SLURM conventions in this family

- **Every job declares an explicit `--time`.** The six pre-existing jobs in this repo use
  `--time=UNLIMITED`. Slurm cannot backfill a job with no declared limit, so an UNLIMITED
  job waits for a full-width drain and its queue time is worst-case by construction — and
  for a 16-task array that penalty is paid 16 times. Resume is safe, so a task killed at
  its limit is re-submitted rather than restarted.
- `--partition=gpu`, `--cpus-per-task=8`, `--mem=96G`, `--gres=gpu:1` on GPU stages.
  `--gres` is **omitted** on the pure-numpy stages (`split`, `directions`, `probe`,
  `metrics`, `geometry`, `prepare`) so they do not idle an A100 in a queue.
- **8 concurrent array tasks at `--mem=96G` is 768 GB of host RAM.** Before planning
  wall-clock, run `sinfo -o "%P %m %G %D %t"`, `sacctmgr show qos` and `squeue`; lower the
  `%8` throttle if `MaxTRESPerUser` or node memory says otherwise.
- `23_step2_steer_dgx.slurm`'s `--array=0-15` and `study_config.sh`'s
  `STUDY_STEER_SHARDS=16` must agree; the wrapper aborts on a mismatch rather than
  generating a partial grid that looks complete. `42_step4_steer_dgx.slurm`'s
  `--array=0-7%8` and `STUDY_OOD_SHARDS=8` must agree for the same reason.
- `30_step3_geometry.slurm` requests `--partition=gpu` because `gpu` is the only partition
  this repo has ever used and no CPU partition has been confirmed. If `sinfo` shows one,
  change that line; if not, the cheapest option is not sbatch at all — it is minutes of
  numpy on the login node.

---

## 4. Budget

Measured anchor: **17.96 s/generation** at batch-1, 1024-token cap, one A100-40GB
(`S.SEC_PER_GEN_A100_40GB`). Batch-8 is assumed to buy ~4x throughput.

| item | per model | x3 models |
|---|---|---|
| steering prompts | 700 | 2,100 |
| layers x directions x non-zero alphas | 10 x 2 x 10 = 200 conditions | 600 |
| steered generations | 140,000 | 420,000 |
| shared alpha=0 baseline | 700 | 2,100 |
| **total generations** | **140,700** | **422,100** |
| GPU-h at batch-1 | 702 | 2,106 |
| **GPU-h at batch-8** | **176** | **526** |

**~527 GPU-h ≈ 5.5 days on 4x A100-40GB.** Do not buy more resolution by widening the
alpha grid — it is the multiplier on everything above.

The whole study, stage by stage (batch-8 throughput is the ~4x assumption, unmeasured):

| stage | generations | GPU-h batch-1 | GPU-h batch-8 |
|---|---|---|---|
| step 1 generate (batch-1 mandatory) | 5,250 (1,750 x 3) | ~26 | — |
| step 2 steer | 422,100 (140,700/model) | ~2,106 | ~527 |
| step 4 generate (batch-1 mandatory) | 870 (290 x 3) | ~4.3 | — |
| **step 4 steer, default** | **174,870 (58,290/model)** | **~872** | **~218** |
| step 4 steer, `--reduced` | 21,750 (7,250/model) | ~108 | ~27 |

Step 4's default is 290 prompts x 10 layers x 2 directions x 10 non-zero alphas + the
290-prompt baseline = 58,290 per model. **That is ~8x the old step-4 default** (the old
default was what `--reduced` now gives) and it takes steps 1+2+4 from ~585 to ~775 GPU-h
(batch 8 where legal, batch-1 where activations are captured). Use `--reduced` if the queue cannot absorb it; the paper's OOD claim then rests on
3 layers and 4 alphas, and should say so. C\*-scoring is ~3 GPU-h for 422k answers (plus a
little over one more for step 4), and steps 3 and the bootstraps are CPU-minutes.

`--dry-run` on any stage prints its own cell counts and GPU-h estimate; trust that over
this table if they ever disagree, and fix the table.

---

## 5. Locked decisions

These were decided by the user. The code does not re-litigate them and neither should a
reviewer's patch.

- **C\* = Longformer only**, run dir
  `../refusal_classifier/results/runs/longformer_42_20260916-160022`. `predict.py`'s
  `DEFAULT_RUNS` lists ModernBERT first, so the run dir is **always** passed explicitly.
  ModernBERT-large was dropped 2026-09-19. See `../../results/reports/decision_log.md`.
- **Roster: Qwen3.5-4B, Qwen3.5-9B, OLMo-3-7B (final, 2026-09-22).** See §7 for the table
  and the system-prompt rule. Nemotron and Gemma are out.
- **Exactly two directions**: `d_R = unit(mu_R - mu_N)`, `d_D = unit(mu_D - mu_N)`.
  No `d_D_perp` (despite plan.md §D6 — that predates this decision).
- **Balanced difference-of-means**: subsample NORMAL to `min(n_R, n_D)`, K=50 seeded draws,
  average the resulting **unit** vectors, renormalise. Averaging unit vectors, not raw
  differences, is what stops one large-norm draw dominating.
- **Layer selection**: top 10 per model by `mean(AUROC(R vs rest), AUROC(D vs rest))` from a
  **linear** probe (standardised logistic regression, 5-fold CV) on the 20% Probing split.
  Steering, every metric and every table cover all 10; only the main-paper transition
  figure shows the top 5 of them per direction by peak target-class pp.
- **Steering position**: additive injection at the **last prefill token only**,
  `h' = h + alpha * d_hat`, applied once before the first generated token. Generated tokens
  are not steered; the effect propagates through the KV cache.
- **Alpha grid** `[-20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20]`. `alpha=0` is generated
  **once per model** as the shared baseline, through the identical batched code path with
  identical static batch membership.
- **Every plot and table uses `alpha / sigma_l`**, never raw alpha. `sigma_l` is the sd of
  the Direction-split activations at layer `l` projected onto the direction. Raw alpha is
  the sampling grid only.
- **`max_new_tokens = 1024`, greedy** (`do_sample: false`).
- **Batching**: left padding, length-bucketed, **static batch membership** (the tail shard
  is padded, never shrunk). `--batch-size` defaults to 8 and goes down to 1. Requires
  `sdpa` or `flash_attention_2`.
- **Random-direction control: OFF.** The flag exists
  (`--control random-unit --control-draws K`, a random vector L2-matched to the found
  direction) and generates nothing by default.
- **One shared prompt partition for all models**: Direction 40% / Probing 20% /
  Steering 40%, no validation split, stratified on the **label triple** crossed with C\*'s own
  train/val/test split, largest-remainder with seeded tie-breaking, seed 42, sha256 over
  sorted `(prompt_id, split)` pairs. `--partition per-model` is the alternative mode.
  **Per-model class counts inside each split will differ — that is expected and every
  output states it.**
- **Bootstrap**: paired over prompt ids, B=10,000, percentile 95% CI, **one shared
  `[B, n_prompts]` resample index** reused across all conditions (a fresh
  `np.random.choice` per condition is ~4 CPU-hours per model).
- **Step 3 geometry uses the full Direction split** — no cross-fitting. `S_l` is compared
  against a label-permutation null band built with the same shared-`mu_N` estimator.
- **Activations layout `stacked`** (one `hidden/<pid>.npz` per prompt). `per_layer` writes
  ~58k files per model and makes the tree unscannable on Windows.
- **Steered output is `answers.jsonl` only.** Per-prompt JSON would be 420k files.
- **Scoring**: import `load_classifier` / `predict` once per process, score in ~20,000-row
  shards appended to parquet, val-parity gate once at startup. **All metrics are defined on
  the ARGMAX LABEL only** — dynamic batch composition shifts probabilities by up to 0.023.

---

## 6. Hard facts (measured in this repo — violating one silently corrupts results)

1. **The HF off-by-one.** Saved activation row `l` is the **output of decoder module
   `layers[l-1]`**. To steer study layer `l` you hook `model.get_decoder().layers[l-1]`.
   Never hardcode a module path: `get_decoder()` resolves correctly for wrapped and
   multimodal models. `S.steering_hook()` already does this; use it.
2. **Steerable rows are `1 .. n_blocks-1` inclusive.** Row 0 is `inputs_embeds` and is
   byte-identical across prompts (no direction exists there); row `n_blocks` is
   post-final-norm and not a residual site. A loop over "all layers" that includes row 0
   emits NaN. Use `S.assert_steerable()` rather than re-deriving the rule.
3. **The capture point is `prompt_last`**: `out.hidden_states[0][l][0, -1, :]` from the
   prefill — the final token of the fully chat-formatted prompt, immediately before the
   first generated token. It is independent of `max_new_tokens`.
4. **Batch-1 is mandatory wherever activations are captured** (padding perturbs hidden
   states). Batching is legal only in the steering stages, where nothing is saved.
5. **Greedy decoding means no within-cell variance.** Error bars come from the paired
   bootstrap over prompt ids, never from reruns.
6. **1,223 of the 1,750 prompts are in C\*'s own training split.** Every headline is
   reported **twice**: on the full split and on the C\*-test-clean subset. The `split`
   column in `data/processed` is C\*'s train/val/test split — never confuse it with the new
   3-way study split.

`22_step2_preflight_dgx.slurm` exists to prove (1)–(3) empirically per model before the
expensive array runs: no-op hooks on every steerable layer, batch-1 forward, compare
`prompt_last` against the stored `.npz` at every layer. Its return dict goes straight into
the run manifest.

---

## 7. The roster (final)

Three models: a Qwen3.5 scale pair (4B, 9B) plus OLMo-3 as the cross-family contrast:

| slug (on disk) | `hf_id` | family | notes |
|---|---|---|---|
| `qwen3-5-4b` | `Qwen/Qwen3.5-4B` | Qwen | `sdpa`; `enable_thinking: false`; hidden 2560, 32 blocks; the config `name` is `qwen3.5-4b` |
| `qwen3-5-9b` | `Qwen/Qwen3.5-9B` | Qwen | `sdpa`; `enable_thinking: false`; the config `name` is `qwen3.5-9b`, which slugifies to `qwen3-5-9b` |
| `olmo3-7b` | `allenai/Olmo-3-7B-Instruct` | OLMo (Allen AI) | `sdpa`; hidden 4096, 32 blocks, steerable rows 1..31 |

**System-prompt rule, one rule for all three:** `system_prompt: null`, i.e. no user-supplied
system prompt, and each model's **chat-template default** applies. Measured 2026-09-21 by
rendering each template with no system message:

| model | what the template does with no system message |
|---|---|
| qwen | no system turn; `enable_thinking=false` keeps the think block empty |
| olmo3 | injects `You are a helpful function-calling AI assistant. You do not currently have access to any functions. <functions></functions>` |

**Flag for the paper: OLMo-3 runs under its template's default persona** ("helpful
function-calling assistant ... no access to any functions"). The other two models never see
that text, and it may raise *technical* refusals ("I don't have access to ...") that C\*
would label REFUSAL for reasons unrelated to content. It is kept because it is the model's
own default and the rule is one rule for all three; report it as a caveat, and check OLMo's
REFUSAL rate on benign prompts against the other two. Overriding it means setting an
**explicit, non-empty** `system_prompt` on the `olmo3-7b` entry — an **empty string does NOT
work**, because `src/generate.py` only adds a system turn when the value is truthy, so `""`
silently falls back to the template default. An override changes the captured activations,
so it means re-running step 1 (and the `split`) and everything after it, not patching.

Changing the roster in any way changes the label-triple strata and therefore the shared
partition: re-run step 1 including `split`, then steps 2–4. Preflight any roster change
first — it reads `config.json` only and downloads no weights:

```bash
python -X utf8 scripts/step1_generate_label_split.py --config configs/study.yaml --stage preflight
```

---

## 8. Troubleshooting

**"attn_implementation eager" warning / batch clamped to 4.**
`S.BatchedGenerator` refuses to trust eager attention at batch 8: eager materialises the
full `[B, H, L, L]` score tensor, and the p99 prompt here is 581 tokens against a max of
3,602. It clamps to `EAGER_MAX_BATCH=4` and `EAGER_MAX_PROMPT_TOKENS=2048` and says so
loudly. This is a 2x budget hit — prefer a model with sdpa or flash_attention_2 (this is
why Gemma was discarded). If you must keep an eager model, re-run `--dry-run` to get the
real GPU-h before submitting.

**CUDA OOM during the array.**
`BatchedGenerator.generate()` halves the batch and retries, logging it; `.n_oom_halvings`
is in the record stream. If it halves repeatedly, submit that shard with
`--batch-size 4` (or 2) rather than letting it thrash. Note that changing the batch size
changes batch membership, and **batched greedy decoding is not bitwise invariant to batch
size** — so if you re-run a shard at a different batch size, re-run its `alpha=0` baseline
the same way, or the comparison is no longer paired. `plan_batches()` is called **once per
model** and the same plan is passed to every condition including the baseline; that
invariant is what makes the whole grid comparable.

**Resume.**
Everything is resume-safe and re-running skips completed work: step 1 skips prompts that
already have `hidden/<pid>.npz`; the steering stages skip prompt ids already present in the
condition's `answers.jsonl` (`S.read_done_ids()` tolerates a truncated final line, which is
what a job killed at its `--time` limit leaves behind); the scoring stage skips parquet
shards already written. So the correct response to a job hitting its wall-clock limit is
`sbatch` again — never `rm -rf` the output directory.

**A `--verify` that fails.**
That is drift, not noise: an input changed under a committed artifact. Find the input
(every artifact carries its `seed`, an input inventory and a sha256), decide whether the
change was intended, and re-run the stage. Do not regenerate the hash to make the check
pass.

**`ImportError: cannot import name 'train' from 'src'` in a scoring stage.**
Both repos ship a top-level `src` package. The step script inserts the generation repo at
`sys.path[0]` and must insert the classifier repo explicitly for `predict.py`'s
`import src.train`. Do **not** "fix" this by exporting `PYTHONPATH=.../refusal_classifier`
— that makes the collision worse and non-deterministic.

**Scores look shifted by a hair between runs.**
Expected, and the reason **all metrics are defined on the argmax label only**: dynamic
batch composition in the classifier moves probabilities by up to 0.023. If a number moved
and it was computed from probabilities, the number is wrong, not the run.

---

## 9. Shipping to the DGX

`scripts/make_study_bundle.py` writes `../../refusal_study.zip` — code, configs, SLURM
launchers and `data/processed` for **both** subprojects (the scoring stages import the
classifier's `src.train` / `src.metrics` / `scripts/predict.py`, so it has to ship too),
with **no weights and no activations**. It CRC-tests the archive before returning, and
prints the manifest, per-file sizes and a sha256.

```bash
python -X utf8 scripts/make_study_bundle.py --dry-run   # print the manifest, write nothing
python -X utf8 scripts/make_study_bundle.py             # writes D:/IITIndore/refusal/refusal_study.zip
python -X utf8 scripts/make_study_bundle.py --verify    # re-test an existing archive, no rebuild
```

Current bundle: **93 files, 4.78 MB uncompressed / 3.75 MB zipped**, `unzip -t` clean.
Alongside it, `refusal_study.manifest.json` records every member with its own sha256;
`--verify` re-checks each member's content against that manifest and exits non-zero on
drift, so it catches an edited source file, not just a damaged container. Inclusion is an
**allowlist** re-checked against a deny list (`*.pt`, `*.safetensors`, `*.npz`, `hidden/`,
`runs/`, `__pycache__`, and an 8 MB per-file cap) — widening an include glob still cannot
smuggle a 567 MiB checkpoint or an activation tree into the bundle. Deliberately excluded:
phase-1 `data/generated/*/answers.jsonl` from the discarded roster, and the retired
`data/study/*.json` (their frozen hashes no longer match the models being run).

**`D:/IITIndore/refusal/longformer_ckpt.zip` is CORRUPT** — `unzip -l` reports "End-of-
central-directory signature not found", i.e. it is a truncated upload, not an archive. Do
not ship it. A valid copy of the run directory is in the working tree at
`refusal_intern/refusal_classifier/results/runs/longformer_42_20260916-160022/`
(`best_state.pt` 567 MiB, plus `tokenizer/`, `config_snapshot.json`, `metrics.json`,
`val_preds.npy`, `val_probs.npy`). Either re-zip that directory and `unzip -t` it before
sending, or confirm the DGX already holds
`~/workspace/refusal/refusal_classifier/results/runs/longformer_42_20260916-160022/`
— C\* must be that exact checkpoint, and a re-trained stand-in is not it.
