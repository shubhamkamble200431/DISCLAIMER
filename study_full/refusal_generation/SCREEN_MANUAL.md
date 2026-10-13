# Model screening — instruction manual

Your job: run a list of **candidate chat models** through two checks on the DGX and send
back the results. That is the whole task. **No training, no labelling, no analysis, and no
interpreting the numbers** — someone else does that part.

There are two stages and **you stop between them and wait**. Stage 1 is a smoke test that
costs almost nothing. Stage 2 generates text and is only run on the models that survive
Stage 1.

---

## 1. What this is for (one paragraph, so the numbers mean something)

We have a study that needs chat models which sometimes **refuse** ("I can't help with that")
and sometimes **add a disclaimer** ("I'm not a doctor, but…"). Our current three models do
this only 14–19% of the time, which is barely enough for the statistics to work. This screen
measures whether some other model would do it more often, and whether we can hook into its
internals the way the study requires. Stage 1 is a smoke test answering "can we use this model
at all"; Stage 2 runs 30 prompts and answers "does it actually refuse and hedge".

The 30 prompts are 10 we know produce a normal answer, 10 that produce a disclaimer and 10
that produce a refusal — picked so that two different classifiers AND all three of our current
models agreed on every one of them. So a candidate's answers compare directly against a
known-good reference label, which rides along in the output as `splits=screen_<LABEL>`.

---

## 2. What you received

Unzip **over the existing checkout** at
`/raid/chandresh/KPdubey/refusal/refusal_generation` — this is an add-on, not a replacement.

```
configs/screen.yaml                    the 15 candidate models, in the order to try them
scripts/screen_tier1.py                Stage 1
scripts/build_probe30.py               already run — do NOT re-run it (see §7)
scripts/slurm/run_screen.sh            launcher, same pattern as run_generate.sh
scripts/slurm/04_screen_tier1_dgx.slurm
scripts/slurm/05_screen_tier2_dgx.slurm
src/generate.py                        MODIFIED — overwrite the existing file
data/screen/probe30.parquet            the frozen 30-prompt set (10 N / 10 D / 10 R)
```

⚠️ **`src/generate.py` replaces the one already there.** It is the old file plus one small
addition. Everything that already works keeps working — verified before shipping, the
original 1,750-prompt config still resolves 1,750 prompts exactly as before.

Run everything **from `refusal_generation/`** (the folder with `src/` and `scripts/` in it).
The scripts do `from src import generate`, so they will not find their library otherwise.

---

## 3. Before you start

Disk: the candidates are about **40 GB** of weights in total. `HF_HOME` already points at
`$REPO_ROOT/.hf_cache` via `scripts/slurm/config.sh`, so they land on `/raid`, not your home
directory. Check there is room.

No Hugging Face token is needed. Every model on the list is ungated — that was checked
deliberately. If something asks you to log in or accept a licence, **stop and report it**;
it means the list is wrong, not that you should accept.

---

## 4. Stage 1 — the cheap gate

### Step 1: preflight (10 seconds, downloads no weights)

```
bash scripts/slurm/run_screen.sh tier1 --preflight
```

This only reads each model's `config.json` and asks whether the installed transformers
recognises the architecture. **Report the output before going further.** Anything printing
`[FAIL]` means our transformers is too old for that model — that is a useful result, not a
problem you need to fix.

This matters because the check was done on a laptop with transformers 5.17.0 and the DGX
runs 5.14.1. Some models may pass there and fail here. That is exactly what this step finds.

### Step 2: the real Stage 1

```
sbatch scripts/slurm/04_screen_tier1_dgx.slurm
```

Downloads the weights, then for each model: loads it, finds the internal layers we need to
attach to, verifies our layer numbering is right, checks we can write into the model without
breaking it, prints its chat template, and measures the size of its internal activations over
a handful of prompts.

**It never generates any text**, so once the weights are downloaded it takes **seconds per
model**. Nearly all the wall-clock is downloading. Expect well under the 4-hour limit.

A model that fails is caught, logged, and the run continues to the next one. **One model
failing is normal and is not something to fix** — it is a result.

### Step 3: send back

```
data/screen/tier1/                     the whole folder
scripts/slurm/logs/refusal_screen_t1_*.out
```

Then **stop and wait.** Do not start Stage 2 on your own.

---

## 5. Stage 2 — only after you are told which models passed

One job per model, using the slug from the left-hand column of `configs/screen.yaml`:

```
sbatch --export=ALL,SCREEN_MODEL=olmo2-1b   scripts/slurm/05_screen_tier2_dgx.slurm
sbatch --export=ALL,SCREEN_MODEL=falcon3-3b scripts/slurm/05_screen_tier2_dgx.slurm
```

This generates 30 answers per model and saves the internal activations. It is quick --
**a few minutes of GPU per model** once the weights are local. Run them one after another;
do not run two on the same GPU.

Send back `data/screen/generated/` and the logs.

Resumable: if a job dies, resubmit the identical command. Finished prompts are detected and
skipped.

---

## 6. Useful flags (Stage 1)

| flag | what it does |
|---|---|
| `--preflight` | architecture check only, then exit |
| `--models a,b` | only these slugs |
| `--n-prompts N` | fewer/more prompts (default 6) |
| `--smoke` | tiny model, 3 prompts — tests the harness itself |
| `--strict` | stop at the first failure instead of continuing |

Try one model on its own first if you want to see it work:

```
bash scripts/slurm/run_screen.sh tier1 --models olmo2-1b
```

---

## 7. Do not change these

They are deliberate, and changing them silently invalidates the comparison:

* **`build_probe30.py` must not be re-run.** The 30 prompts are frozen. Re-running it would
  mean different models get screened on different prompts, which makes them incomparable.
  It is shipped for reference only.
* **`max_new_tokens: 256`** — chosen from a measurement, and our existing models are already
  scored at this exact setting so the comparison is like-for-like.
* **`do_sample: false`** — greedy decoding. Reruns give identical answers. Do not add
  sampling or a temperature.
* **No system prompt** — we are measuring each model's own default behaviour.
* **Batch size 1** — padding corrupts the activations this exercise exists to collect.
* **`capture.points: [prompt_last]`, `layers: all`.**

If you think one of these needs to change, **ask first**.

---

## 8. If something goes wrong

**"Transformers does not recognize this architecture"** — expected for some candidates.
Record which, move on. Do not upgrade transformers: that would change the environment the
existing results were produced in.

**Asks for a token / licence** — stop and report. Do not accept anything.

**CUDA out of memory** — shouldn't happen in Stage 1 (forward pass only, the biggest
candidate peaked at ~3 GB on an 8 GB laptop card). If it does, report which model.

**Disk full** — the weights are large. Report it rather than deleting things from `.hf_cache`.

**A model reports `map=False` or `noop=False`** — that is a genuine, interesting finding
about that model, not a bug for you to chase. Report it as-is.

**The job died / SSH dropped** — resubmit the same command.

---

## 9. What to report when you are done

For Stage 1, the summary table at the bottom of the log says it all. For each model:

* `ok` — yes/no
* `tmpl` / `map` / `noop` — the three structural checks
* `blocks` — how many layers it has
* `||h|| range` and `VRAM`
* the `stage/error` column for anything that failed, with the error text

Send the whole `data/screen/tier1/` folder rather than retyping numbers.

---

## 10. Running it on the 5060 Ti instead of the DGX

The 16 GB RTX 5060 Ti box is an alternative to the SLURM route above. No `sbatch`, no
`config.sh` — one script runs everything sequentially.

```
bash scripts/setup_5060.sh          # one time; creates .venv with torch cu128 + transformers>=5.17
source .venv/bin/activate
bash scripts/run_screen_5060.sh     # tier1 then tier2, every model that fits in 16 GB
```

Subsets and resuming work the same way as `run_wide.sh`:

```
bash scripts/run_screen_5060.sh olmo2-1b falcon3-3b     # just these
STAGES="tier2" bash scripts/run_screen_5060.sh          # skip tier1, resume tier2
```

Progress lands in `data/screen/status.tsv`, per-stage logs in `data/screen/logs/`.
A model that fails is logged and skipped; the rest carry on. tier2 is resumable —
re-running after a crash skips finished prompts.

**Two differences from the DGX run, both deliberate.**

*Four models are excluded* because they do not fit in 16 GB at bf16 (peak VRAM measured
during tier1 on the A100): `deepseek-r1-distill-llama-8b` 16.1 GB,
`qwen3-5-9b-control` 18.3 GB, `ministral-3-8b` 19.2 GB, and `phi-tiny-moe` which needs
`flash_attn` (no Blackwell/sm_120 wheel exists). Run those on the DGX if they are wanted.

*The control changes* from `qwen3-5-9b-control` to `qwen3-5-4b-control`. Both are
incumbents and both were part of the agreement that selected the 30 probe prompts, but
the 4B fits. The control runs first on purpose: if it does not return a near-clean
10/10/10 diagonal, stop and fix the harness before reading any candidate's numbers.

**One thing to write down:** this box runs transformers 5.17 while the DGX runs 5.14.1,
which is where all the existing incumbent data came from. For a screen that is acceptable,
but record it. It also means `nemotron3-nano-4b` — which died on the DGX with
`KeyError: 'mlp'` — should load here, because 5.17 has that layer type. The runner prints
whether it does at startup.
