"""lora_setup.py — LoRA configuration utilities for fine-tuning homework (Week 5)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_TARGET_MODULES: list[str] = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

_VARIANTS = {"vanilla", "rslora", "dora", "pissa"}


# ---------------------------------------------------------------------------
# LoRA config builder
# ---------------------------------------------------------------------------

def build_lora_config(
    rank: int = 16,
    alpha: int = 32,
    target_modules: list[str] | None = None,
    variant: str = "vanilla",
    lora_dropout: float = 0.05,
    task_type: str = "CAUSAL_LM",
) -> Any:
    """Build a PEFT LoraConfig for the specified variant."""
    try:
        from peft import LoraConfig, TaskType  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "[lora_setup] 'peft' is not installed. "
            "Run: pip install peft"
        )

    if variant not in _VARIANTS:
        raise ValueError(f"[lora_setup] Unknown variant '{variant}'. Choose from {sorted(_VARIANTS)}.")

    modules = target_modules if target_modules is not None else list(_DEFAULT_TARGET_MODULES)

    # Map task_type string -> TaskType enum
    task_type_enum = getattr(TaskType, task_type, TaskType.CAUSAL_LM)

    # Variant-specific kwargs
    extra: dict[str, Any] = {}
    if variant == "vanilla":
        extra = {"use_rslora": False, "use_dora": False, "init_lora_weights": True}
    elif variant == "rslora":
        extra = {"use_rslora": True}
    elif variant == "dora":
        extra = {"use_dora": True}
    elif variant == "pissa":
        extra = {"init_lora_weights": "pissa"}

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=modules,
        lora_dropout=lora_dropout,
        task_type=task_type_enum,
        **extra,
    )

    print(f"[lora_setup] LoraConfig built:")
    print(f"  variant         : {variant}")
    print(f"  rank (r)        : {rank}")
    print(f"  alpha           : {alpha}")
    print(f"  lora_dropout    : {lora_dropout}")
    print(f"  target_modules  : {modules}")
    print(f"  task_type       : {task_type}")
    for k, v in extra.items():
        print(f"  {k:<16}: {v}")

    return config


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------

def count_trainable_params(model: Any) -> dict:
    """Count total and trainable parameters; return and print a summary."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable / total if total > 0 else 0.0
    result = {
        "total": total,
        "trainable": trainable,
        "trainable_pct": round(pct, 4),
    }
    print(f"[lora_setup] Parameter summary:")
    print(f"  total params     : {total:,}")
    print(f"  trainable params : {trainable:,}")
    print(f"  trainable %      : {pct:.4f}%")
    return result


# ---------------------------------------------------------------------------
# Variant comparison
# ---------------------------------------------------------------------------

def compare_variants(
    model_name: str,
    variants: list[str] = ("vanilla", "rslora", "dora", "pissa"),
    rank: int = 16,
) -> dict:
    """Load model once per variant, apply LoRA, count params; return comparison dict."""
    try:
        from transformers import AutoModelForCausalLM  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "[lora_setup] 'transformers' is not installed. "
            "Run: pip install transformers"
        )
    try:
        from peft import get_peft_model  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "[lora_setup] 'peft' is not installed. "
            "Run: pip install peft"
        )

    print(f"[lora_setup] compare_variants: model={model_name}, variants={list(variants)}, rank={rank}")

    results: dict = {}
    for variant in variants:
        print(f"\n[lora_setup] --- Variant: {variant} ---")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        config = build_lora_config(rank=rank, variant=variant)
        peft_model = get_peft_model(model, config)
        stats = count_trainable_params(peft_model)
        results[variant] = {
            "trainable_pct": stats["trainable_pct"],
            "total_params": stats["total"],
            "trainable_params": stats["trainable"],
            "config": config,
        }

    # Print comparison table
    print("\n[lora_setup] Variant comparison table:")
    header = f"  {'variant':<10} {'trainable':>14} {'total':>14} {'trainable_%':>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for variant, info in results.items():
        print(
            f"  {variant:<10} {info['trainable_params']:>14,} "
            f"{info['total_params']:>14,} {info['trainable_pct']:>11.4f}%"
        )

    return results


# ---------------------------------------------------------------------------
# Save scoreboard
# ---------------------------------------------------------------------------

def save_lora_scoreboard(
    results: dict,
    path: str = "outputs/lora_variants_scoreboard.json",
) -> None:
    """Save compare_variants output to JSON, omitting non-serialisable fields."""
    serialisable: dict = {}
    for variant, info in results.items():
        serialisable[variant] = {
            k: v
            for k, v in info.items()
            if isinstance(v, (str, int, float, bool, list, dict, type(None)))
        }

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(serialisable, fh, indent=2)
    print(f"[lora_setup] Scoreboard saved to {path} ({len(serialisable)} variants)")
