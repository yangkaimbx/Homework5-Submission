# Week 5: Fine-Tuning LLMs — SFT, PEFT, and Preference Tuning

## Overview

This week you build a complete fine-tuning pipeline: from raw data collection
and synthetic dataset generation, through LoRA/QLoRA supervised fine-tuning,
all the way to DPO preference alignment and serving your trained adapter via
Ollama. You will also integrate your fine-tuned model back into the RAG
pipeline you built in Week 4, turning a generic base model into a domain-aware
assistant grounded in your own document corpus.

---

## Learning Objectives

By the end of this assignment you will be able to:

- Explain what supervised fine-tuning (SFT) does at the gradient level and why
  it differs from RLHF.
- Apply LoRA and QLoRA to a pre-trained model using HuggingFace PEFT and TRL,
  or MLX-LM on Apple Silicon.
- Generate a structured synthetic instruction dataset using Claude as a data
  synthesis engine.
- Run Direct Preference Optimization (DPO) to align a fine-tuned model with
  human preferences without a reward model.
- Evaluate model quality using LLM-as-judge scoring and compare multiple LoRA
  variant configurations (rank, alpha, dropout, target modules).
- Serve a trained adapter through Ollama using a custom Modelfile and query it
  from a local client.
- Integrate a fine-tuned model into the HW4 RAG retrieval pipeline as a
  skill-aware RESOLVER.
- Understand the nanochat pipeline architecture (Karpathy, Oct 2025) and the
  gbrain skill-routing pattern (Garry Tan).

---

## Setup Options

### Path A: MLX-LM (Recommended on Mac Apple Silicon)

**Prerequisites:** macOS 14+, Apple Silicon (M1/M2/M3/M4), Python 3.10+

```bash
# Install MLX and training stack
pip install mlx-lm>=0.20.0

# Verify installation
python -c "import mlx_lm; print('MLX-LM ready')"
```

**Time estimate:** 5-25 minutes for a 500-step LoRA run on a 0.5-1 B model,
depending on chip generation. See `docs/stack_choice_guide.md` for detailed
benchmarks.

**When to pick this path:** You are on an Apple Silicon Mac and want fast,
free, local training. This is the recommended default for most students.

---

### Path B: HF + PEFT + TRL (Recommended on GPU / Linux)

**Prerequisites:** Linux or WSL2, NVIDIA GPU with >= 8 GB VRAM, CUDA 11.8+,
Python 3.10+

```bash
# Install HuggingFace training stack
pip install transformers>=4.46.0 peft>=0.14.0 trl>=0.13.0 \
            accelerate>=1.0.0 datasets>=3.0.0

# QLoRA support (Linux + CUDA only)
pip install bitsandbytes>=0.44.0

# Verify
python -c "import trl; import peft; import bitsandbytes; print('HF stack ready')"
```

**Time estimate:** 3-14 minutes for a 500-step LoRA run on a 0.5-1 B model on
an RTX 3080 or better. QLoRA cuts memory requirements by ~55%.

**When to pick this path:** You have a CUDA GPU and want access to the full
HuggingFace ecosystem, including DPO, KTO, and GRPO from TRL.

---

### Path C: Cloud GPU (Bonus)

If you want to fine-tune models larger than 1 B parameters, or if you do not
have local GPU hardware, use a cloud provider. The course InferenceAI platform
provides credits — check the course portal for your allocation.

See `docs/stack_choice_guide.md` Section 4 for provider comparison, cost
estimates, and a Modal quickstart.

---

## Prerequisites

- Completed HW1-HW4. NB08 (RAG integration) specifically requires your HW4
  vector store and retrieval pipeline to be working.
- Python 3.10 or higher.
- `ANTHROPIC_API_KEY` set in `.env` (required for synthetic data generation in
  NB03 and LLM-as-judge evaluation in NB06).
- One of the following:
  - Apple Silicon Mac (Path A), OR
  - NVIDIA GPU with >= 8 GB VRAM (Path B), OR
  - Cloud GPU account with credits (Path C), OR
  - CPU-only machine (very slow, debugging only — not recommended for full runs)

---

## Installation

**1. Clone or navigate to the repo**

```bash
git clone <your-repo-url>
cd Homework5-Submission
```

**2. Create and activate a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

On Apple Silicon, `mlx-lm` installs automatically via the platform marker.
On Linux with CUDA, `bitsandbytes` installs automatically.
On Windows or Intel Mac, neither installs — use Path C.

**4. Configure environment variables**

```bash
cp .env.example .env
# Open .env in your editor and fill in ANTHROPIC_API_KEY (required)
# and HF_TOKEN if you plan to use gated models like Llama-3
```

**5. Test your installation**

```bash
python -c "
from dotenv import load_dotenv; load_dotenv()
import anthropic, transformers, peft, trl
print('Core stack OK')
try:
    import mlx_lm; print('MLX-LM OK (Apple Silicon)')
except ImportError:
    pass
try:
    import bitsandbytes; print('bitsandbytes OK (CUDA)')
except ImportError:
    pass
"
```

**6. Launch Jupyter**

```bash
jupyter lab
```

Navigate to the `notebooks/` directory and start with `NB00_setup.ipynb`.

---

## Repository Structure

```
Homework5-Submission/
│
├── README.md                        # This file
├── requirements.txt                 # All Python dependencies
├── .env.example                     # Environment variable template
│
├── docs/
│   └── stack_choice_guide.md        # MLX vs HF vs Cloud decision guide
│
├── notebooks/
│   ├── NB00_setup.ipynb             # Environment check and hardware detection
│   ├── NB01_concepts.ipynb          # SFT, LoRA, DPO concepts with visuals
│   ├── NB02_tokenization.ipynb      # Tokenizers, chat templates, BPE sidebar
│   ├── NB03_data_collection.ipynb   # PDF ingestion and synthetic data gen
│   ├── NB04_lora_variants.ipynb     # LoRA rank/alpha/dropout experiments
│   ├── NB05_sft.ipynb               # Full supervised fine-tuning run
│   ├── NB06_dpo.ipynb               # Direct Preference Optimization
│   ├── NB07_eval_serve.ipynb        # LLM-as-judge eval + Ollama serving
│   └── NB08_rag_integration.ipynb   # Connect fine-tuned model to HW4 RAG
│
├── src/
│   ├── data_utils.py                # Dataset loading and formatting helpers
│   ├── training_utils.py            # LoRA config, QLoRA setup, trainer wrappers
│   ├── eval_utils.py                # LLM-as-judge scoring functions
│   └── rag_resolver.py              # Skill-aware RESOLVER for NB08
│
├── outputs/
│   ├── homework_reflection.md       # Your written reflection (REQUIRED)
│   ├── lora_variants_scoreboard.json  # NB04 experiment results (REQUIRED)
│   ├── synthetic_dataset.json       # NB03 generated dataset (REQUIRED)
│   ├── eval_scoreboard.json         # NB07 judge scores (REQUIRED)
│   ├── preference_tuning_results.json # NB06 DPO results (REQUIRED)
│   ├── my_project_update.md         # Weekly project journal entry (REQUIRED)
│   └── ollama_modelfile.txt         # Your Ollama Modelfile (REQUIRED or explain)
│
└── test_data/
    └── sample_docs/                 # Sample PDFs for NB03 testing
```

---

## Assignment Structure

| NB   | Title                  | Topic                              | Est. Time | Key Deliverable                     |
|------|------------------------|------------------------------------|-----------|-------------------------------------|
| NB00 | Setup & Hardware Check | Environment, GPU/MLX detection     | 20 min    | Confirmation cell passes            |
| NB01 | Concepts               | SFT, LoRA math, DPO intuition      | 45 min    | Concept quiz answers in notebook    |
| NB02 | Tokenization           | Chat templates, BPE, padding       | 45 min    | Tokenizer comparison table          |
| NB03 | Data Collection        | PDF load, synthetic data via Claude| 60 min    | `outputs/synthetic_dataset.json`    |
| NB04 | LoRA Variants          | Rank, alpha, DoRA, RSLoRA, PiSSA   | 90 min    | `outputs/lora_variants_scoreboard.json` |
| NB05 | SFT Run                | Full training loop, loss curves    | 90 min    | Trained adapter saved to disk       |
| NB06 | DPO                    | Preference pairs, DPO training     | 90 min    | `outputs/preference_tuning_results.json` |
| NB07 | Eval + Serve           | LLM-as-judge, Ollama Modelfile     | 60 min    | `outputs/eval_scoreboard.json`      |
| NB08 | RAG Integration        | Skill RESOLVER with HW4 pipeline   | 60 min    | Working end-to-end demo             |

Total estimated time: **9-10 hours** across the week.

---

## Deliverables

### Required (graded, ~70% of score)

Submit all of the following files in the `outputs/` directory:

| File | Description | NB |
|------|-------------|----|
| `outputs/homework_reflection.md` | 300-500 word reflection on what you built, what surprised you, and what you would do differently | All |
| `outputs/lora_variants_scoreboard.json` | Results table comparing at least 4 LoRA configurations (rank, alpha, variant) | NB04 |
| `outputs/synthetic_dataset.json` | At least 50 instruction-response pairs generated via Claude | NB03 |
| `outputs/eval_scoreboard.json` | LLM-as-judge scores for baseline vs. fine-tuned model on >= 20 prompts | NB07 |
| `outputs/preference_tuning_results.json` | DPO training logs and before/after comparison | NB06 |
| `outputs/my_project_update.md` | One-page update on your course project connecting this week's work | All |
| `outputs/ollama_modelfile.txt` | Your working Ollama Modelfile, OR a written explanation of why llama.cpp was not available on your hardware | NB07 |

### Bonus (extra credit, up to 30% additional)

| Task | Description | Points |
|------|-------------|--------|
| Cloud GPU run | Fine-tune a 1 B+ model on a cloud GPU and include logs | +10 |
| Mini BPE tokenizer | Implement a BPE tokenizer from scratch (NB02 sidebar) | +8 |
| HF Hub release | Publish your adapter and synthetic dataset to HuggingFace Hub | +7 |
| DPO + GRPO comparison | Run both algorithms on the same dataset and compare | +8 |
| Multi-skill RESOLVER | Implement a RESOLVER that routes to 2+ skills in NB08 | +7 |

Bonus points are capped at 30% of the base assignment score.

---

## Cost Estimates

This assignment is designed to run with minimal API spend. The only paid
component is the Claude API for synthetic data generation and LLM-as-judge
evaluation.

| Component | Path A (MLX) | Path B (HF+GPU) | Path C (Cloud) |
|-----------|-------------|-----------------|----------------|
| Training compute | $0 (local) | $0 (local GPU) | $1-3 |
| Synthetic data (NB03, 50-100 examples) | $0.50-1.50 | $0.50-1.50 | $0.50-1.50 |
| LLM-as-judge eval (NB07, 20-50 calls) | $0.30-0.80 | $0.30-0.80 | $0.30-0.80 |
| **Total estimate** | **$1-3** | **$1-3** | **$2-5** |

Costs are based on Claude Haiku pricing as of April 2026. Using Claude Sonnet
for judge evaluation will cost ~5x more — Haiku is sufficient and recommended.

To stay within budget:
- Generate synthetic data in batches of 10 with a single API call each.
- Use the `model="claude-haiku-4-5"` for all judge calls.
- Cache API responses with `@lru_cache` or save to JSON after each call.

---

## Troubleshooting

**`bitsandbytes` import error on Mac**

`bitsandbytes` does not support Apple Silicon. Use `mlx-lm` (Path A) instead.
The `requirements.txt` already guards this with `platform_system == "Linux"`.

**MPS out-of-memory (`mps` backend)**

If using HF Transformers with `device_map="mps"`, reduce `max_seq_length` to
256 and `per_device_train_batch_size` to 1. Better option: switch to MLX-LM
which has much more efficient unified memory management.

**Tokenizer padding warning**

If you see `Setting pad_token = eos_token`, this is expected for base models.
Ensure `tokenizer.padding_side = "right"` for SFT with packing.

**`mlx_lm` not found after install**

On Intel Macs or Linux, `mlx-lm` is not installed (platform marker). This is
correct. Use Path B or Path C on non-Apple-Silicon hardware.

**GGUF conversion fails**

GGUF conversion requires `llama.cpp` to be compiled locally. If you cannot
compile it, export your adapter as a HuggingFace model and note this in your
`ollama_modelfile.txt` submission with an explanation.

**Ollama not responding (`connection refused`)**

Ensure the Ollama server is running: `ollama serve`. Check `OLLAMA_HOST` in
your `.env` (default: `http://localhost:11434`). On Mac, Ollama may need to be
started from the menu bar app.

**HuggingFace Hub 403 on gated model (Llama-3)**

Set `HF_TOKEN` in your `.env` and run `huggingface-cli login`. You also need
to accept the model's license on the Hub website before your token will work.

**PEFT version mismatch with TRL**

`peft>=0.14.0` and `trl>=0.13.0` are required together. Older PEFT versions
changed the `LoraConfig` API. If you see `unexpected keyword argument 'use_rslora'`,
upgrade: `pip install --upgrade peft trl`.

**`gradient_checkpointing` error with PEFT**

Add `model.enable_input_require_grads()` after wrapping with `get_peft_model()`.
This is required for gradient checkpointing to work with PEFT adapters.

**DPO training loss goes to 0 immediately**

This usually means the preference pairs are not formatted correctly, or the
reference model and policy model share weights without being properly detached.
Ensure you load the reference model with `model.disable_adapter()` or load a
separate copy.

**Slow training on CPU**

CPU-only training is not supported for full runs. If you do not have GPU
access, use Path C (cloud). For quick debugging, set `max_steps=10` to verify
your pipeline runs end-to-end.

**`sentencepiece` install fails on Windows**

Use WSL2 for full compatibility, or install via conda:
`conda install -c conda-forge sentencepiece`.

**Chat template not applied (garbled outputs)**

Always apply the chat template before tokenizing:
```python
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```
Forgetting this causes the model to see raw text instead of properly formatted
`<|user|>` / `<|assistant|>` tokens, leading to incoherent outputs.

---

## Key Concepts

Brief definitions for terms used throughout the notebooks:

| Term | Definition |
|------|------------|
| **SFT** | Supervised Fine-Tuning — training on (instruction, response) pairs with a standard cross-entropy loss |
| **LoRA** | Low-Rank Adaptation — freezes base model weights and learns low-rank delta matrices for each attention layer |
| **QLoRA** | Quantized LoRA — loads the base model in 4-bit precision to reduce VRAM while LoRA adapters remain in fp16/bf16 |
| **DoRA** | Weight-Decomposed LoRA — decomposes weight matrices into magnitude and direction, learns updates to both |
| **RSLoRA** | Rank-Stabilized LoRA — scales the LoRA output by 1/sqrt(rank) instead of 1/rank for more stable high-rank training |
| **PiSSA** | Principal Singular Values and Singular Vectors Adaptation — initializes LoRA from SVD of the original weight matrix |
| **DPO** | Direct Preference Optimization — aligns model to prefer chosen over rejected responses without a reward model |
| **KTO** | Kahneman-Tversky Optimization — preference tuning from binary feedback (good/bad) rather than paired comparisons |
| **GRPO** | Group Relative Policy Optimization — RL-style alignment using group-normalized rewards (used in DeepSeek-R1) |
| **RLHF** | Reinforcement Learning from Human Feedback — the general family of methods that train a reward model then do PPO |
| **Chat template** | A model-specific string format that separates system, user, and assistant turns (e.g. ChatML, Llama-3 format) |
| **ChatML** | A chat template format using `<|im_start|>` and `<|im_end|>` tokens, originally from OpenAI, used by Qwen |
| **GGUF** | GPT-Generated Unified Format — the file format llama.cpp and Ollama use for quantized model weights |
| **Adapter** | The saved LoRA weight deltas, typically a small folder with `adapter_config.json` and `adapter_model.safetensors` |
| **Skill resolver** | A routing layer that maps an incoming query to one or more specialist sub-models or retrieval tools |

---

## Resources

### Core References

- **Karpathy nanochat** (Oct 2025): https://github.com/karpathy/nanochat
  Full LLM pipeline from scratch — tokenizer, transformer, SFT, RLHF. The
  implementation we study in NB01 and NB02.

- **gbrain (Garry Tan)**: https://github.com/garrytan/gbrain
  Agent skill architecture with a RESOLVER pattern. Basis for the NB08
  integration design.

- **HuggingFace TRL docs**: https://huggingface.co/docs/trl
  Full API reference for SFTTrainer, DPOTrainer, KTOTrainer, GRPOTrainer.

- **PEFT docs**: https://huggingface.co/docs/peft
  LoRA, QLoRA, DoRA, RSLoRA, and PiSSA configuration guides.

- **MLX-LM**: https://github.com/ml-explore/mlx-lm
  Apple's native training and inference library for Apple Silicon.

### Papers Worth Reading

- **LoRA** (Hu et al., 2021): https://arxiv.org/abs/2106.09685
  Original low-rank adaptation paper.

- **QLoRA** (Dettmers et al., 2023): https://arxiv.org/abs/2305.14314
  4-bit quantization + LoRA. Enabled fine-tuning 65 B models on a single GPU.

- **DPO** (Rafailov et al., 2023): https://arxiv.org/abs/2305.18290
  The paper that replaced reward-model + PPO with a single closed-form loss.

- **DoRA** (Liu et al., 2024): https://arxiv.org/abs/2402.09353
  Weight decomposition for more expressive low-rank updates.

- **GRPO** (DeepSeek-AI, 2025): https://arxiv.org/abs/2501.12599
  The RL algorithm behind DeepSeek-R1's reasoning capabilities.

### Tools and Platforms

- **Weights & Biases**: https://wandb.ai — training monitoring (optional, set
  `WANDB_API_KEY` in `.env`)
- **HuggingFace Hub**: https://huggingface.co — model and dataset hosting
- **Ollama**: https://ollama.com — local model serving
- **Modal**: https://modal.com — serverless cloud GPU compute

---

## Timeline (Suggested 7-Day Schedule)

| Day | Notebooks | Focus | Time |
|-----|-----------|-------|------|
| Day 1 (Mon) | NB00, NB01 | Setup + concepts. Verify your backend works end-to-end. Read the LoRA paper abstract. | 1.5 hr |
| Day 2 (Tue) | NB02, NB03 | Tokenization deep dive. Generate your synthetic dataset (this is your main spend day). | 2 hr |
| Day 3 (Wed) | NB04 | LoRA variant experiments. Run at least 4 configs, fill in the scoreboard. | 1.5 hr |
| Day 4 (Thu) | NB05 | Full SFT training run. Let it cook — use the time to read the DPO paper. | 2 hr |
| Day 5 (Fri) | NB06 | DPO training on your preference pairs. Compare outputs to the SFT checkpoint. | 1.5 hr |
| Day 6 (Sat) | NB07 | LLM-as-judge evaluation + Ollama serving. Write your Modelfile. | 1.5 hr |
| Day 7 (Sun) | NB08, reflection | RAG integration + written deliverables. Submit. | 2 hr |

Do not leave NB03 (synthetic data) until the weekend — it is the foundation
for NB05 and NB06 and needs to be done early.

---

## What's Next (Week 6 Preview)

Week 6 moves from single-model fine-tuning to **multi-model inference
optimization**: quantization strategies beyond QLoRA (GGUF, AWQ, GPTQ),
speculative decoding, continuous batching, and deploying models behind a
production-grade inference server. The adapter you trained this week will be
your test subject for Week 6's benchmarking exercises.

---

## Support

**Bug reports and questions:** Open an issue in the course GitHub repository.
Include your operating system, Python version, and the full error traceback.

**Office hours:** See the course portal for the weekly schedule. TA office
hours are the best place for debugging environment issues.

**Slack:** Use the `#hw5-finetuning` channel for peer help. Sharing your LoRA
scoreboard results is encouraged — comparing hyperparameter sensitivity across
different hardware setups is genuinely useful for everyone.
