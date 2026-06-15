# DPLoRA: A Dual-Pruning Framework based on ILP Optimization and Progressive Pruning for Parameter-Efficient LoRA Fine-Tuning

This repository is the official implementation of our paper
"DPLoRA: A Dual-Pruning Framework based on ILP Optimization and Progressive
Pruning for Parameter-Efficient LoRA Fine-Tuning", accepted to the Findings
of the Association for Computational Linguistics: ACL 2026.

**Authors:** Changjun Park¹, Sejong Yoon², Jaekwang Kim¹,³ (✉ corresponding author)
¹ Department of Applied Data Science, Sungkyunkwan University ·
² Department of Computer Science, The College of New Jersey ·
³ Department of Applied Artificial Intelligence, Sungkyunkwan University

DPLoRA is a parameter-budget-aware LoRA fine-tuning method. It allocates
LoRA ranks across layers using an Integer Linear Program (ILP) and then
progressively reduces the budget along a Bézier-shaped schedule, recovering
performance between prune steps. The pipeline is two stages:

1. **Stage 1 — Initial rank allocation (ILP).** Given a per-layer set of
   candidate ranks `r ∈ R` and a total parameter budget `B`, solve an ILP
   that picks exactly one rank per layer subject to ∑ cost(r_l) ≤ B and an
   aggregate rank ≥ 1 per layer type (Eq. 5). The objective maximises the
   total estimated performance gain (Eq. 3), where each layer's gain (Eq. 2)
   is driven by a Fisher-style importance signal — the mean of squared
   gradients (Eq. 1) — collected from a small calibration loop.

2. **Stage 2 — Progressive pruning.** During training, periodically tighten
   the parameter budget along a cubic Bézier schedule (Eq. 12/13/14),
   re-solve the ILP for each new budget under a non-increasing-rank
   constraint (Eq. 11) using EMA-smoothed importance (Eq. 6/7), then resize
   each LoRA adapter by top-k slicing on weight-magnitude (L2-norm)
   importance (Eq. 15/16), and continue training so the model recovers
   between prune steps.

This yields the two methods reported below:

- **OPLoRA** — Stage 1 only (optimal ILP rank placement, no progressive
  pruning). Launched by the `*_oplora.sh` scripts.
- **DPLoRA (p)** — Stage 1 + Stage 2 with target reduction `p`. Launched by
  the `*_dp0.4 … dp0.8.sh` scripts.

The codebase covers two settings:

- **GLUE classification** (`training/train_glue.py`) — RoBERTa-base,
  eight GLUE tasks.
- **Instruction tuning on Alpaca** (`training/train_alpaca.py`) —
  Llama-3 (3.2 1B/3B, 3 8B), with MT-Bench evaluation through
  FastChat.

Entry points are invoked via `python3 -m training.train_<task>`; the shell
launchers in `scripts/` set hyperparameters per task and call the right
module.

## Repository layout

```
DPLoRA/
├── data/                     # dataset preprocessing
│   ├── glue.py               # GLUE column conventions + tokenization
│   └── alpaca.py             # Stanford Alpaca prompt + masking
├── modeling/                 # model + LoRA construction
│   └── model_setup.py        # prepare_model_for_glue / _for_alpaca
├── utils/                    # cross-cutting, task-agnostic helpers
│   ├── determinism.py        # set_deterministic_environment, seed_worker
│   ├── common.py             # runtime / memory / model utility functions
│   ├── logging_utils.py      # CSV / JSON logging of optimisation results
│   └── save_resume.py        # full train-state checkpoint save / resume
├── pruning/                  # rank decision + layer surgery
│   ├── pruning_config.py            # default hyperparameters (shared)
│   ├── pruning_utils.py             # importance, masking, optimizer resync;
│   │                                #   estimate_parameter_cost + ILPNotSolvedError (shared)
│   ├── stage1/                      # Stage 1 — initial rank allocation
│   │   ├── initial_rank_allocation.py   # ILP solver
│   │   ├── lora_optimizer.py            # LoRAOptimizer wrapper around Stage 1
│   │   └── lora_setup.py                # optimize_lora_config + validation helpers
│   └── stage2/                      # Stage 2 — progressive pruning
│       ├── progressive_pruning.py       # manager + ILP per step
│       └── pruning_scheduler.py         # Bézier schedule (Eq. 12/13/14)
├── training/                 # entry points + argparse only
│   ├── train_glue.py         # ENTRY: GLUE training loop
│   ├── train_alpaca.py       # ENTRY: Alpaca training + MT-Bench
│   ├── args_glue.py          # argparse for GLUE
│   └── args_alpaca.py        # argparse for Alpaca
├── evaluation/               # eval logic, factored out of train loops
│   ├── glue.py               # GLUE eval pass
│   ├── alpaca_generation.py  # token-weighted perplexity + samples
│   └── mt_bench.py           # FastChat MT-Bench harness
├── scripts/                  # shell launchers only
│   ├── train_<task>_<method>.sh                  # GLUE: per task × method (dp0.4–0.8, oplora)
│   ├── train_alpaca_<size>_<method>.sh           # Alpaca: 1B/3B/8B × {dp0.6, oplora}; seeds 42/2025/777 loop inside
│   └── run_mt_bench.sh                           # MT-Bench evaluation
├── loralib/                  # vendored Microsoft loralib (official, unmodified)
└── FastChat/                 # MT-Bench harness (third-party, required) — must use the DPLoRA conv templates (alpaca_dplora_llama / alpaca_dplora_qwen)
```

## Requirements

This project uses two separate requirement files, one for the GLUE benchmark
and one for the Alpaca benchmark.

The pinned `torch`/`torchvision` wheels are CUDA 12.6 builds (`+cu126`), served
from PyTorch's wheel index rather than the default PyPI — pass
`--extra-index-url https://download.pytorch.org/whl/cu126` when installing.

**For GLUE experiments:**
```setup
pip install -r requirements_glue.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

**For Alpaca experiments:**
```setup
pip install -r requirements_alpaca.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

You will also need the following system dependencies:
```setup
sudo apt update
sudo apt install dos2unix
sudo apt-get install coinor-cbc
```

### Storage Requirements

Full experiments require substantial storage:

- **GLUE Benchmark**: ~35GB
- **Alpaca Benchmark**: ~250GB

## Training and Evaluation

All launchers live in `scripts/`. Each GLUE launcher trains RoBERTa-base on
one task across three seeds (42, 2025, 777); hyperparameters follow the paper.

### GLUE

```bash
# DPLoRA with target reduction p=0.6 on CoLA (runs seeds 42, 2025, 777)
bash scripts/train_cola_dp0.6.sh

# OPLoRA (Stage 1 only, no progressive pruning) on SST-2
bash scripts/train_sst2_oplora.sh
```

Available methods per task: `dp0.4`, `dp0.5`, `dp0.6`, `dp0.7`, `dp0.8`
(DPLoRA at target reduction `p`) and `oplora` (OPLoRA). Tasks: `cola`,
`mnli`, `mrpc`, `qnli`, `qqp`, `rte`, `sst2`, `stsb`.

### Alpaca

```bash
# DPLoRA p=0.6 on the 8B model (runs seeds 42, 2025, 777)
bash scripts/train_alpaca_8B_dp0.6.sh

# OPLoRA (Stage 1 only) on the 1B model
bash scripts/train_alpaca_1B_oplora.sh
```

Available Alpaca launchers: `1B`/`3B`/`8B` × `dp0.6` (DPLoRA) and `oplora`
(OPLoRA); each script runs seeds 42, 2025, 777 internally.

To invoke the entry points directly without a launcher:

```bash
python3 -m training.train_glue   --task_name cola --seed 42 ...
python3 -m training.train_alpaca --model_name_or_path meta-llama/Llama-3.2-1B --seed 42 ...
```

The Alpaca launcher writes to `output/alpaca_<model>_<method>_seed<seed>_<timestamp>/`.

### MT-Bench evaluation

```bash
bash scripts/run_mt_bench.sh path/to/checkpoint/final_model
# answer generation must use the DPLoRA conversation template:
#   --conv-template alpaca_dplora_llama   (Llama-3 models)
#   --conv-template alpaca_dplora_qwen    (Qwen models)
```

Requires FastChat (cloned under `./FastChat`) and an `OPENAI_API_KEY` in the
environment if you want to run the GPT-4 judgment step. Pass `true` as the
second argument to skip judgment. The bundled `DPLoRAAdapter` auto-selects the
DPLoRA template when the model path contains `dplora`; otherwise set it
explicitly via `--conv-template` (`alpaca_dplora_llama` / `alpaca_dplora_qwen`).

## Note on Experimental Setup

DPLoRA's contribution is ILP-based layer-wise rank allocation for LoRA adapters,
so its trainable-parameter budget is defined over the adapter scope.
Therefore, GLUE experiments freeze the classification head, 
and the # Params we report for our methods are adapter-only.

## Results

We report the overall (matched and mismatched) accuracy for MNLI, Matthews
correlation for CoLA, Pearson correlation for STS-B, and accuracy for the
other tasks. Higher is better for all metrics.

### GLUE Benchmark

| Method | # Params(M) | CoLA | MNLI | MRPC | QNLI | QQP | RTE | SST-2 | STS-B | Avg. |
| :----- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Full FT | 125.0 | 60.7 | 87.5 | 88.6 | 92.7 | **91.7** | 75.8 | 93.8 | 90.3 | 85.1 |
| LoRA (r=8) | 1.3 | 63.4 | 87.2 | 87.8 | 92.5 | 90.5 | 76.3 | 94.4 | 90.7 | 85.3 |
| SoRA | 0.9 | 58.4 | **87.7** | 87.3 | 92.9 | 91.6 | 76.8 | 93.8 | 90.3 | 84.8 |
| AdaLoRA+ | 0.8 | 61.7 | 87.5 | 87.4 | 92.7 | 90.0 | 78.9 | 94.4 | 91.0 | 85.5 |
| AdaLoRA | 0.3 | 61.4 | 87.3 | 86.5 | 92.7 | 89.8 | 78.8 | 93.9 | **91.0** | 85.2 |
| Adapter* | 0.9 | 62.6 | 87.3 | 88.4 | **93.0** | 90.6 | 75.9 | 94.7 | 90.3 | 85.4 |
| BitFit* | 0.1 | 62.0 | 84.7 | **92.7** | 91.8 | 84.0 | **81.5** | 93.7 | 90.8 | 85.2 |
| OPLoRA (ours) | 1.2 | **63.7** | 87.5 | 89.8 | **93.0** | 90.9 | 77.6 | **94.8** | 90.0 | 85.9 |
| DPLoRA (ours, p=0.4) | 0.7 | 63.6 | 87.5 | 89.2 | **93.0** | 90.9 | 80.0 | 94.6 | 90.2 | **86.1** |
| DPLoRA (ours, p=0.5) | 0.6 | 61.4 | 87.2 | 89.5 | 92.8 | 90.7 | 79.3 | 94.0 | 90.0 | 85.6 |
| DPLoRA (ours, p=0.6) | 0.5 | 61.2 | 86.8 | 89.5 | 92.8 | 90.5 | 78.3 | 94.7 | 89.8 | 85.4 |
| DPLoRA (ours, p=0.7) | 0.4 | 59.5 | 86.4 | 89.6 | 92.8 | 90.1 | 77.1 | 93.5 | 89.2 | 84.8 |
| DPLoRA (ours, p=0.8) | 0.2 | 60.5 | 85.5 | 89.1 | 92.3 | 89.2 | 77.6 | 94.0 | 89.2 | 84.7 |

AdaLoRA+ denotes AdaLoRA run with its official default settings while only increasing the target parameter budget.

### Alpaca Benchmark (MT-BENCH)

Evaluated by GPT-4; higher is better.

| Model | Method | # Params(M) | MT-BENCH |
| :---: | :----- | :---: | :---: |
| **LLaMA 3.2 1B** | LoRA (r=8)† | 5.6 | 3.07 |
| | DoRA | 6.0 | 2.83 |
| | OPLoRA (ours) | 5.5 | **3.12** |
| | DPLoRA (ours, p=0.6) | **2.2** | 2.51 |
| **LLaMA 3.2 3B** | LoRA (r=8)† | 12.2 | 4.46 |
| | DoRA | 12.9 | 4.33 |
| | OPLoRA (ours) | 11.7 | **4.48** |
| | DPLoRA (ours, p=0.6) | **4.7** | 4.10 |
| **LLaMA 3 8B** | LoRA (r=8)† | 21.0 | 5.13 |
| | DoRA | 22.3 | 4.97 |
| | OPLoRA (ours) | 20.5 | **5.20** |
| | DPLoRA (ours, p=0.6) | **8.2** | 4.97 |

**Notes:**
* **Bold** indicates best performance in each column/group.
* `*` indicates results taken from the original papers.
* `†` indicates baselines re-evaluated with our hyperparameter settings (matched).
* '(ours)' denotes a method proposed in this paper.

These results demonstrate our framework's dual strengths: OPLoRA consistently
outperforms existing PEFT baselines on both GLUE and MT-Bench, while the full
DPLoRA framework establishes a new state-of-the-art on GLUE at the
high-performance setting (p=0.4, 86.1 avg) and offers a superior
performance-efficiency trade-off at higher pruning rates.

## Key flags

| Flag | Meaning |
|------|---------|
| `--use_initial_rank_allocation` | Run the Stage-1 ILP rank allocation (OPLoRA); otherwise all target layers use a uniform rank `--lora_r` (Eq. 3/5). |
| `--lora_r_values 0,1,…,16`   | Candidate rank set R for the ILP. |
| `--lora_budget B`            | Stage-1 parameter budget (number of LoRA params). |
| `--apply_pruning`            | Enable Stage 2 progressive pruning (DPLoRA). |
| `--pruning_target_reduction` | Stage-2 target reduction R_target (Eq. 13). |
| `--pruning_steps N`          | Number of Stage-2 prune steps. |
| `--importance_ema_decay α`   | EMA decay β for the importance signal (Eq. 6). |
| `--momentum_penalty_weight`  | Weight of the rank-change penalty (γ term, Eq. 10). |
| `--stable_layer_bonus`       | Bonus for keeping ranks stable (δ term, Eq. 10). |
| `--recovery_steps`           | Recovery training steps after each prune step. |

## Citation

If you find our work useful in your research, please consider citing our paper:

```bibtex
@inproceedings{park2026dplora,
  title={DPLoRA: A Dual-Pruning Framework based on ILP Optimization and Progressive Pruning for Parameter-Efficient LoRA Fine-Tuning},
  author={Park, Changjun and Yoon, Sejong and Kim, Jaekwang},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2026},
  year={2026}
}
```

## License

MIT — see [`LICENSE`](LICENSE).

### Third-party components

This repository vendors third-party code under its original license:

- **loralib** — Microsoft LoRA, MIT License — see [`loralib/LICENSE`](loralib/LICENSE).
- **FastChat** — LMSYS FastChat, Apache License 2.0 — see [`FastChat/LICENSE`](FastChat/LICENSE).
  Only the `llm_judge` (MT-Bench) harness is used. Changes from upstream
  (commit `587d5cf`): added the DPLoRA conversation templates
  (`alpaca_dplora_llama` / `alpaca_dplora_qwen`) to `fastchat/conversation.py`,
  and removed the unused `assets/` demo media.
