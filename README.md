# To Refusal and Beyond: Representation and Steering of Disclaimer Behavior in LLMs


## Repo structure

```
DISCLAIMER/
├── README.md
├── requirements.txt
|── classifier/                       
│   ├── configs/
|   │   └── longformer.yaml           
|   ├── data/
|   │   └── processed/
|   │       ├── train.parquet         
|   │       ├── val.parquet           
|   │       └── test.parquet          
|   ├── scripts/
|   │   ├── train_transformer.py      
|   │   └── predict.py                
|   └── src/
|        ├── data.py                   
|        ├── train.py                  
|        └── metrics.py  
|
└── generation/                       
    ├── configs/
    │   ├── study.yaml                
    │   └── generate.yaml             
    ├── scripts/
    │   ├── generate_label_split.py   
    │   ├── layer_probe.py            
    │   ├── directions_steering.py    
    │   ├── splithalf_geometry_cross.py  
    │   ├── token_overlap.py           
    │   ├── random_control.py        
    │   ├── generate_activations.py   
    │   └── parity_batched_capture.py 
    └── src/
        ├── generate.py               
        ├── batched_capture.py       
        └── steer.py                 
             
```

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

- **Original Dataset** — [`Human-CentricAI/chatbot-arena-llm-refusal`](https://huggingface.co/datasets/Human-CentricAI/chatbot-arena-llm-refusal)


## CLI commands
All commands run from inside `classifier/`.

```bash

python scripts/train_transformer.py --config configs/longformer.yaml --smoke

python scripts/train_transformer.py --config configs/longformer.yaml --seed 42

python scripts/predict.py --generated ../generation/data/generated
```

`classifier/src/data.py`, `train.py`, and `metrics.py` are library
modules only (data loading/label collapse, the training harness, and
shared metrics), imported by the two scripts above and with no CLI of
their own.

Training writes the checkpoint to
`classifier/results/runs/longformer_<seed>_<timestamp>/`. Point
`study.classifier_run_dir` in `generation/configs/study.yaml` at that
directory, and pass it to `predict.py` with `--run`.



All commands run from inside `generation/`.

```bash

python scripts/generate_label_split.py --config configs/study.yaml --stage all

python scripts/generate_label_split.py --config configs/study.yaml --stage split

python scripts/generate_activations.py --config configs/generate.yaml --models qwen3-5-4b,olmo3-7b

python scripts/parity_batched_capture.py --config configs/study.yaml --n-prompts 50

python scripts/layer_probe.py --models qwen3-5-4b,olmo3-7b --top-k 10

python scripts/directions_steering.py --config configs/study.yaml --stage all

python scripts/directions_steering.py --config configs/study.yaml --stage probe

python scripts/splithalf_geometry_cross.py --config configs/study.yaml --stage all

python scripts/token_overlap.py --config configs/study.yaml --models qwen3-5-4b --stage all

python scripts/random_control.py --config configs/study.yaml --models qwen3-5-4b --stage all
```

`generation/src/steer.py` is a library module (steering hook,
`BatchedGenerator`, shared helpers) imported by the scripts above, not
run directly -- except its own self-test suite:

```bash
python src/steer.py
```

`generation/src/generate.py` and `generation/src/batched_capture.py`
are library modules only, with no CLI of their own.

### Batching

`steering.batch_size` in `configs/study.yaml` defaults to `1`
(one prompt at a time, no padding). Every script that steers or
generates also takes `--batch-size N` directly, e.g.:

```bash
python scripts/directions_steering.py --config configs/study.yaml --stage steer --batch-size 8
```





