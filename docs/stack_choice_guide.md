# Choosing Your Fine-Tuning Stack — Week 5 Guide

This guide helps you pick the right training backend for your hardware so you
spend time learning fine-tuning concepts rather than debugging environment
issues. Read the TL;DR tree first, then jump to the section that matches your
setup.

---

## 1. TL;DR Decision Tree

```
Are you on Apple Silicon (M1/M2/M3/M4)?
│
├─ YES ──► Path A: MLX-LM
│            Fast, native, no CUDA needed.
│            Install: pip install mlx-lm
│            See Section 2.
│
└─ NO
    │
    Do you have a CUDA GPU with >= 8 GB VRAM?
    │
    ├─ YES ──► Path B: HF + PEFT + TRL + QLoRA
    │            Best OSS ecosystem, QLoRA cuts memory in half.
    │            Install: pip install transformers peft trl accelerate bitsandbytes
    │            See Section 3.
    │
    └─ NO
        │
        Do you have access to a cloud GPU?
        │
        ├─ YES ──► Path C: Cloud GPU
        │            Modal, Lambda Labs, InferenceAI platform, RunPod.
        │            See Section 4.
        │
        └─ NO ───► HF + CPU (very slow, for debugging only)
                     Set device_map="cpu" in your SFTConfig.
                     Expect 10-50x slower than GPU.
                     Only feasible for < 500 M parameter models.
```

---

## 2. Path A: MLX-LM (Apple Silicon)

### When to Use
- You are on an M1, M2, M3, or M4 Mac.
- You want zero-cost local training with no cloud credits.
- You are fine-tuning models up to ~3 B parameters (M1 16 GB) or ~7 B (M2/M3
  Max with 32-64 GB unified memory).
- You prefer a simple CLI-first workflow with minimal boilerplate.

### Install

```bash
# Requires Python 3.10+ and macOS 14+
pip install mlx-lm

# Verify
python -c "import mlx; import mlx_lm; print('MLX ready')"
```

### Supported Models (as of mlx-lm 0.20)
- Qwen2.5 (0.5 B, 1.5 B, 3 B, 7 B)
- SmolLM2 (135 M, 360 M, 1.7 B)
- Llama 3.2 (1 B, 3 B) — requires HF_TOKEN for gated variants
- Phi-3.5-mini
- Mistral 7 B
- Gemma 2 (2 B, 9 B)

Full list: https://github.com/ml-explore/mlx-lm#supported-models

### Speed Benchmarks (rough estimates, LoRA rank=8)

| Model   | Hardware | Tokens/sec | Time for 500 steps |
|---------|----------|------------|--------------------|
| 0.5 B   | M1 8 GB  | ~1200 t/s  | ~5-10 min          |
| 0.5 B   | M2 16 GB | ~1800 t/s  | ~3-7 min           |
| 1 B     | M1 16 GB | ~700 t/s   | ~15-25 min         |
| 1 B     | M2 16 GB | ~1100 t/s  | ~10-18 min         |
| 3 B     | M2 16 GB | ~400 t/s   | ~25-40 min         |
| 7 B     | M3 Max   | ~250 t/s   | ~40-70 min         |

These figures are for single-GPU (unified memory) training with sequence length
512 and batch size 4. Throughput scales roughly linearly with batch size until
memory pressure hits.

### Basic Fine-Tuning Command

```bash
# SFT with LoRA via mlx-lm CLI
mlx_lm.lora \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --train \
  --data data/ \
  --iters 500 \
  --batch-size 4 \
  --lora-layers 8 \
  --learning-rate 1e-4 \
  --val-batches 25 \
  --save-every 100 \
  --adapter-path adapters/
```

---

## 3. Path B: HF + PEFT + TRL + QLoRA (GPU / Linux)

### When to Use
- You have an NVIDIA GPU with >= 8 GB VRAM (RTX 3080, A100, etc.).
- You want full access to the HuggingFace ecosystem (Hub, datasets, eval).
- You need DPO, KTO, GRPO, or other preference tuning algorithms (TRL).
- You are targeting models > 3 B parameters.
- You want to publish adapters to HuggingFace Hub.

### Install

```bash
pip install transformers>=4.46.0 peft>=0.14.0 trl>=0.13.0 \
            accelerate>=1.0.0 datasets>=3.0.0

# QLoRA (Linux + CUDA only)
pip install bitsandbytes>=0.44.0
```

### QLoRA Memory Savings

| Model Size | Dtype  | VRAM Required |
|------------|--------|---------------|
| 0.5 B      | fp16   | ~1.0 GB       |
| 0.5 B      | 4-bit  | ~0.5 GB       |
| 1 B        | fp16   | ~2.0 GB       |
| 1 B        | 4-bit  | ~0.9 GB       |
| 3 B        | fp16   | ~6.0 GB       |
| 3 B        | 4-bit  | ~2.5 GB       |
| 7 B        | fp16   | ~14.0 GB      |
| 7 B        | 4-bit  | ~5.5 GB       |
| 13 B       | fp16   | OOM on 24 GB  |
| 13 B       | 4-bit  | ~9.0 GB       |

Rule of thumb: 4-bit QLoRA reduces VRAM by roughly 55-60% compared to fp16
full weights, with < 1% performance degradation on most tasks.

### SFTConfig Defaults That Fit 8 GB VRAM

```python
from trl import SFTConfig

config = SFTConfig(
    output_dir="outputs/sft-run",
    num_train_epochs=3,
    per_device_train_batch_size=2,     # keep low for 8 GB
    gradient_accumulation_steps=4,     # effective batch = 8
    gradient_checkpointing=True,       # trade compute for memory
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=True,                         # use bf16 on Ampere+
    fp16=False,
    max_seq_length=512,                # reduce to 256 if OOM
    packing=True,                      # pack short sequences
    logging_steps=10,
    save_steps=100,
    eval_steps=100,
    report_to="none",                  # or "wandb"
)
```

For 4-bit QLoRA, add a `BitsAndBytesConfig` to your model load call and wrap
with `prepare_model_for_kbit_training()` from PEFT.

---

## 4. Path C: Cloud GPU

Use cloud GPUs when:
- Your model is > 1 B parameters and local training is too slow.
- You want to run DPO or GRPO experiments that require more memory.
- You want to compare against a larger baseline (7 B, 13 B).

### Provider Comparison

| Provider             | GPU Options        | Cost (est.)     | Notes                          |
|----------------------|--------------------|-----------------|--------------------------------|
| InferenceAI platform | A10G, A100         | Course credits  | Preferred for this course      |
| Modal                | A10G, H100         | ~$0.76-$4/hr    | Pay-per-second, easy deploy    |
| Lambda Labs          | A100 (80 GB)       | ~$1.99/hr       | Good for large model runs      |
| RunPod               | RTX 4090, A100     | ~$0.44-$2.49/hr | Spot instances available       |
| Google Colab Pro+    | A100 (40 GB)       | ~$50/mo         | Notebook-friendly              |

### Getting Started on Modal

```bash
pip install modal
modal setup  # authenticate
modal run notebooks/NB05_sft.py  # run your training script remotely
```

### Cost Estimate for HW5
A typical SFT run on a 1 B model with 1,000 steps on an A10G takes about
10-20 minutes (~$0.15-$0.30). Full DPO on a 3 B model: ~$0.50-$1.00.
Budget $2-5 in cloud credits for the whole assignment if you use Path C.

---

## 5. Common Gotchas

### bitsandbytes on Mac
`bitsandbytes` does not support Apple Silicon. If you see:
```
RuntimeError: CUDA is not available. bitsandbytes requires CUDA.
```
Use MLX-LM (Path A) instead, or switch to a cloud GPU. Do not try to install
bitsandbytes on macOS — it will silently fall back to CPU and be very slow.

### MPS vs. CUDA Kernels
PyTorch MPS (Metal Performance Shaders) is not the same as MLX. When using HF
Transformers on Mac with `device_map="mps"`, many custom CUDA kernels (Flash
Attention, Triton ops, bitsandbytes) will fail. Stick to MLX-LM for Mac
training.

### Tokenizer Padding
Chat-formatted datasets require a padding token. Many base models (Llama,
Mistral) set `pad_token = None`. Fix:
```python
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # important for SFT packing
```
Forgetting this causes silent loss-of-gradient bugs where the model learns to
predict padding tokens.

### Gradient Checkpointing + PEFT
When using `gradient_checkpointing=True` with PEFT, you must call:
```python
model.enable_input_require_grads()
```
Otherwise you get a `RuntimeError: element 0 of tensors does not require grad`
during the first backward pass.

### GGUF Conversion for Ollama
`llama.cpp` conversion scripts expect a specific directory structure. After
merging your LoRA adapter:
```bash
python -m mlx_lm.convert --hf-path merged-model/ -q  # quantize to 4-bit
```
Then use `ollama create` with a Modelfile pointing to the GGUF file. See
NB07 for the full workflow.

---

## 6. Quick Benchmark Table

| Model  | Backend         | Est. Time (500 steps) | Peak Memory |
|--------|-----------------|-----------------------|-------------|
| 0.5 B  | MLX (M1)        | 5-10 min              | 2-3 GB RAM  |
| 0.5 B  | HF fp16 (GPU)   | 3-8 min               | 1.5 GB VRAM |
| 0.5 B  | HF 4-bit (GPU)  | 4-10 min              | 0.8 GB VRAM |
| 1 B    | MLX (M2)        | 10-18 min             | 4-6 GB RAM  |
| 1 B    | HF fp16 (GPU)   | 6-14 min              | 2.5 GB VRAM |
| 1 B    | HF 4-bit (GPU)  | 8-16 min              | 1.2 GB VRAM |
| 3 B    | MLX (M2 Max)    | 25-40 min             | 10-14 GB RAM|
| 3 B    | HF 4-bit (GPU)  | 15-30 min             | 3.5 GB VRAM |
| 7 B    | MLX (M3 Max)    | 40-70 min             | 20-28 GB RAM|
| 7 B    | HF 4-bit (A100) | 20-40 min             | 7 GB VRAM   |

All estimates assume LoRA rank=8, sequence length=512, batch size=4 with
gradient accumulation. Actual times vary based on dataset, sequence length
distribution, and hardware generation.

---

*Last updated: Week 5, 2026. If you find faster configs, share them in the
course Slack — we update this guide each semester.*
