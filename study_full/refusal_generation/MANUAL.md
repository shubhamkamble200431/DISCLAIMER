# Generation + hidden-state capture — instruction manual

Your job: run three chat models over 1750 prompts and save, for each prompt, the
model's answer and its internal hidden state at every layer. That is the whole task.
No training, no labelling, no analysis.

---

## 1. What you received

```
scripts/generate_activations.py   the command you run
src/generate.py                   the library it uses
src/__init__.py                   empty file, but the import breaks without it
configs/generate.yaml             which models to run and how
data/processed/*.parquet          the dataset the prompts come from
MANUAL.md                         this file
```

Keep this exact folder layout and run everything from the top-level folder (the one
containing `src/` and `scripts/`). The script does `from src import generate`, so it
will not find its library if you run it from inside `scripts/`.

---

## 2. Install

You need Python 3.10+ and an NVIDIA GPU.

```
pip install "transformers>=5.5" accelerate torch pandas pyarrow numpy PyYAML huggingface_hub
```

**`transformers>=5.5` is not optional.** Support for `google/gemma-4-E2B-it` (model
type `gemma4`) was added in 5.5. On anything older it fails instantly with
"Transformers does not recognize this architecture". If your torch needs a specific
CUDA build, install torch separately first, then the rest.

Check it worked:

```
python -c "import torch, transformers; print(transformers.__version__, torch.cuda.is_available())"
```

You want a version >= 5.5 and `True`.

---

## 3. The models

All three are public — no licence acceptance, no HF token needed.

| slug | Hugging Face page | size on disk |
|------|-------------------|--------------|
| `qwen3-5-4b` | https://huggingface.co/Qwen/Qwen3.5-4B | 4.7B params |
| `gemma-4-e2b` | https://huggingface.co/google/gemma-4-E2B-it | 5.1B params |
| `qwen3-5-9b` | https://huggingface.co/Qwen/Qwen3.5-9B | 9.7B params |

Roughly 40 GB of downloads at bf16. Download them before starting so a slow network
does not stall the run:

```
hf download Qwen/Qwen3.5-4B
hf download google/gemma-4-E2B-it
hf download Qwen/Qwen3.5-9B
```

Set `HF_HOME` to a disk with space if your home directory is small.

Note: "E2B" in the Gemma name is its *effective* size. The actual checkpoint is 5.1B
params, larger than the Qwen 4B. Do not size your disk assuming 2B.

---

## 4. Run it — in this order

### Step 1: preflight (seconds, downloads nothing but config files)

```
python -X utf8 scripts/generate_activations.py --config configs/generate.yaml --preflight
```

Every model must print `[ok  ]`. If one prints `[FAIL]`, fix that before continuing —
almost always it means your transformers is too old. This step exists so you find out
in 10 seconds instead of after a 40 GB download.

### Step 2: smoke test (about a minute)

```
python -X utf8 scripts/generate_activations.py --config configs/generate.yaml --smoke
```

Runs a small model on 3 prompts and writes to `data/generated_smoke/`. Confirms the
whole pipeline works on your machine. Should end with `3 ok, 0 failed`.

### Step 3: the real run, one model at a time

```
python -X utf8 scripts/generate_activations.py --config configs/generate.yaml --models qwen3-5-4b
python -X utf8 scripts/generate_activations.py --config configs/generate.yaml --models gemma-4-e2b
python -X utf8 scripts/generate_activations.py --config configs/generate.yaml --models qwen3-5-9b
```

Run these **sequentially, never in parallel** — two models on one GPU will fight for
memory and both will slow down or crash.

Omitting `--models` entirely runs all three back to back, which is fine if you can
leave it unattended.

Expect roughly 5-7 hours per model on a server GPU at the default 512 new tokens.

---

## 5. Choosing models

See what is available:

```
python -X utf8 scripts/generate_activations.py --config configs/generate.yaml --list-models
```

`--models` accepts:

| form | example |
|------|---------|
| one slug | `--models qwen3-5-9b` |
| several, comma-separated | `--models qwen3-5-4b,gemma-4-e2b` |
| the full Hugging Face id | `--models Qwen/Qwen3.5-9B` |
| omitted | runs every model in the config |

A name that matches nothing is an error listing the valid slugs, so a typo stops you
immediately rather than silently running the wrong thing.

---

## 6. All the flags

| flag | what it does |
|------|--------------|
| `--config PATH` | which config file to use (always `configs/generate.yaml`) |
| `--models A,B` | which models to run (see above) |
| `--list-models` | print the roster and exit |
| `--preflight` | check model support, print sizes, exit |
| `--smoke` | tiny model, 3 prompts, short answers |
| `--dry-run` | show what would run and how much is already done, then exit |
| `--limit N` | only the first N prompts — good for a quick trial |
| `--splits train,val,test` | which dataset splits to pull prompts from (default: all) |
| `--max-new-tokens N` | override answer length cap (default 512) |
| `--overwrite` | redo prompts that are already finished |
| `--strict` | stop at the first failed prompt instead of continuing |
| `--log-every N` | progress line every N prompts (default 25) |

A good trial before committing to the full run:

```
python -X utf8 scripts/generate_activations.py --config configs/generate.yaml --models qwen3-5-4b --limit 20
```

---

## 7. What it produces

```
data/generated/
├── prompts.parquet                    the 1750 prompts, shared by all models
├── run_summary.json
└── qwen3-5-4b/                        one folder per model
    ├── answers/
    │   ├── 1404102.json               one file per prompt: the answer + metadata
    │   └── ...
    ├── hidden/
    │   ├── 1404102/                      one FOLDER per prompt
    │   │   ├── prompt_last_layer_00.npy  one file per layer
    │   │   ├── prompt_last_layer_01.npy
    │   │   └── ...                       33 files (Qwen) or 36 (Gemma)
    │   └── ...
    ├── answers.jsonl                  every answer in one file, easy to skim
    ├── manifest.json                  what ran, how long, what failed
    └── run.log
```

Each `.npy` is one layer's hidden state at **the last token of the chat-formatted
prompt** — the position immediately before the model starts answering. The answers are
still generated and saved; they are a deliverable in their own right, and every record
stores the exact `rendered_prompt` the vector came from. About 1 GB and
~178,000 files in total across the three models.

**Because of the file count, always tar or zip before transferring.** Copying 178k
small files directly over scp or rsync is enormously slower than moving one archive.

```
tar -czf generated.tar.gz data/generated
```

---

## 8. If something goes wrong

**"Transformers does not recognize this architecture"** — your transformers is older
than 5.5. Upgrade it. This is by far the most common problem.

**CUDA out of memory** — the script logs the failing prompt and keeps going, so a few
OOMs will not kill the run. If it happens constantly, lower `--max-new-tokens` to 256,
or check nothing else is using the GPU.

**The run died / SSH dropped / the machine rebooted** — just run the exact same
command again. Finished prompts are detected and skipped; it picks up where it stopped.
Nothing is lost and nothing is duplicated.

**It is extremely slow (under 5 tokens/sec)** — the model does not fit in VRAM and is
being offloaded to CPU. Use a bigger GPU. This is expected on small cards and is not a
bug.

**Some prompts failed** — normal, do not panic. `manifest.json` for that model lists
every failure with its reason under `"failures"`. Report the count when you hand over.

---

## 9. When you are finished

For each of the three models, check `manifest.json` and report:

* `prompts.n_run` — how many succeeded
* `prompts.n_failed` — how many did not
* `totals.wall_seconds` — how long it took

Then archive `data/generated/` and send it back.

---

## 10. Do not change these

They are deliberate, and changing them silently invalidates the results:

* **`do_sample: false`** — greedy decoding, i.e. temperature 0. Reruns produce
  byte-identical answers. Do not add sampling or a temperature.
* **Batch size 1** — batching requires padding, and padding corrupts the hidden states
  this whole exercise exists to collect.
* **`capture.points: [prompt_last]`** and **`layers: all`** — the chosen measurement:
  the last token of the chat-formatted prompt, at every layer.
* **No system prompt** — we are measuring each model's own default behaviour.

If you think one of these needs to change, ask first.
