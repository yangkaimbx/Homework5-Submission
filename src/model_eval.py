"""Week 5: Evaluation utilities — LLM-as-judge, GSM8K micro-eval, latency benchmark."""

from __future__ import annotations

import json
import re
import statistics
import time
from pathlib import Path
from typing import Callable

_DEFAULT_PROMPTS = [
    "What is the capital of France?",
    "Explain what gradient descent is in one sentence.",
    "Write a Python function that returns the factorial of n.",
    "What are the main benefits of fine-tuning an LLM?",
    "Summarize the transformer architecture briefly.",
]

_GSM8K_PROBLEMS = [
    {"question": "If a store has 48 apples and sells 13, how many remain? Show work then #### answer.", "answer": 35},
    {"question": "A car travels 60 mph for 2.5 hours. How many miles? Show work then #### answer.", "answer": 150},
    {"question": "Sam earns $15/hr and works 8 hrs/day for 5 days. Total pay? Show work then #### answer.", "answer": 600},
    {"question": "A box holds 24 oranges. You have 7 boxes. Total oranges? Show work then #### answer.", "answer": 168},
    {"question": "Jenny buys 3 shirts at $12 each and 2 pants at $25 each. Total cost? Show work then #### answer.", "answer": 86},
    {"question": "A class has 30 students. 2/3 passed. How many passed? Show work then #### answer.", "answer": 20},
    {"question": "Tom reads 15 pages/day. How many pages in 3 weeks? Show work then #### answer.", "answer": 315},
    {"question": "A recipe needs 2.5 cups flour per batch. For 4 batches, how many cups? Show work then #### answer.", "answer": 10},
    {"question": "There are 52 cards in a deck. 13 are hearts. How many are not hearts? Show work then #### answer.", "answer": 39},
    {"question": "A train has 8 cars, each with 64 seats. Total seats? Show work then #### answer.", "answer": 512},
]


def judge_llm_eval(
    model_fn: Callable[[str], str],
    test_set: list[dict],
    rubric: str,
    judge_client,
    judge_model: str = "claude-haiku-4-5-20251001",
) -> dict:
    """Evaluate model_fn on each test item using judge_client as LLM-as-judge (1-5 rubric)."""
    print(f"[model_eval] Starting LLM-as-judge eval on {len(test_set)} items with model={judge_model}")

    scores: list[int] = []
    details: list[dict] = []

    for i, item in enumerate(test_set):
        question = item.get("question", "")
        reference = item.get("reference", "")

        print(f"[model_eval] Evaluating item {i + 1}/{len(test_set)}: {question[:60]}...")

        try:
            model_answer = model_fn(question)
        except Exception as e:
            print(f"[model_eval] model_fn failed on item {i + 1}: {e}")
            model_answer = ""

        judge_prompt = (
            f"You are an expert evaluator. Rate the following answer on a scale of 1 to 5.\n\n"
            f"Rubric:\n{rubric}\n\n"
            f"Question: {question}\n\n"
            f"Reference Answer: {reference}\n\n"
            f"Model Answer: {model_answer}\n\n"
            f"Respond with ONLY a single integer from 1 to 5, then a brief one-sentence justification.\n"
            f"Format: <score>\n<justification>"
        )

        try:
            judge_response = judge_client.generate(judge_prompt, model=judge_model)
            score_match = re.search(r"^([1-5])", judge_response.strip())
            score = int(score_match.group(1)) if score_match else 3
        except Exception as e:
            print(f"[model_eval] Judge call failed on item {i + 1}: {e}")
            score = 3
            judge_response = f"Error: {e}"

        print(f"[model_eval] Item {i + 1} score: {score}/5")
        scores.append(score)
        details.append({
            "question": question,
            "reference": reference,
            "model_answer": model_answer,
            "score": score,
            "judge_response": judge_response,
        })

    mean_score = statistics.mean(scores) if scores else 0.0
    pass_rate = sum(1 for s in scores if s >= 3) / len(scores) if scores else 0.0

    print(f"[model_eval] Eval complete: mean={mean_score:.2f}, pass_rate={pass_rate:.1%} (>= 3)")
    return {
        "scores": scores,
        "mean": mean_score,
        "pass_rate": pass_rate,
        "details": details,
    }


def gsm8k_micro_eval(
    model_fn: Callable[[str], str],
    n: int = 10,
) -> dict:
    """Run model_fn on n hardcoded GSM8K problems, check #### answer format."""
    problems = _GSM8K_PROBLEMS[:n]
    print(f"[model_eval] GSM8K micro-eval: running {len(problems)} problems...")

    correct = 0
    pattern = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")

    for i, problem in enumerate(problems):
        question = problem["question"]
        expected = problem["answer"]

        print(f"[model_eval] Problem {i + 1}/{len(problems)}: {question[:60]}...")
        try:
            response = model_fn(question)
        except Exception as e:
            print(f"[model_eval] model_fn failed: {e}")
            response = ""

        match = pattern.search(response)
        if match:
            try:
                predicted = float(match.group(1))
                is_correct = abs(predicted - expected) < 0.01
            except ValueError:
                is_correct = False
        else:
            is_correct = False

        status = "CORRECT" if is_correct else "WRONG"
        print(f"[model_eval] Problem {i + 1}: {status} (expected={expected})")
        if is_correct:
            correct += 1

    accuracy = correct / len(problems) if problems else 0.0
    print(f"[model_eval] GSM8K result: {correct}/{len(problems)} correct ({accuracy:.1%})")
    return {"correct": correct, "total": len(problems), "accuracy": accuracy}


def latency_benchmark(
    model_fn: Callable[[str], str],
    prompts: list[str] | None = None,
    n_calls: int = 5,
) -> dict:
    """Time n_calls to model_fn. Returns min_ms, max_ms, mean_ms, p50_ms, p95_ms, tokens_per_sec."""
    if prompts is None:
        prompts = _DEFAULT_PROMPTS

    effective_prompts = [prompts[i % len(prompts)] for i in range(n_calls)]
    print(f"[model_eval] Latency benchmark: {n_calls} calls...")

    latencies_ms: list[float] = []
    total_tokens = 0

    for i, prompt in enumerate(effective_prompts):
        print(f"[model_eval] Call {i + 1}/{n_calls}...")
        start = time.perf_counter()
        try:
            response = model_fn(prompt)
        except Exception as e:
            print(f"[model_eval] Call {i + 1} failed: {e}")
            response = ""
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies_ms.append(elapsed_ms)
        total_tokens += len(response.split())
        print(f"[model_eval] Call {i + 1}: {elapsed_ms:.1f} ms")

    sorted_lat = sorted(latencies_ms)
    mean_ms = statistics.mean(latencies_ms) if latencies_ms else 0.0
    min_ms = sorted_lat[0] if sorted_lat else 0.0
    max_ms = sorted_lat[-1] if sorted_lat else 0.0
    p50_ms = sorted_lat[len(sorted_lat) // 2] if sorted_lat else 0.0
    p95_idx = int(len(sorted_lat) * 0.95)
    p95_ms = sorted_lat[min(p95_idx, len(sorted_lat) - 1)] if sorted_lat else 0.0
    total_time_s = sum(latencies_ms) / 1000
    tokens_per_sec = total_tokens / total_time_s if total_time_s > 0 else 0.0

    print(f"[model_eval] Benchmark: min={min_ms:.1f}ms, mean={mean_ms:.1f}ms, p50={p50_ms:.1f}ms, p95={p95_ms:.1f}ms, max={max_ms:.1f}ms, tok/s={tokens_per_sec:.1f}")
    return {
        "min_ms": min_ms,
        "max_ms": max_ms,
        "mean_ms": mean_ms,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "tokens_per_sec": tokens_per_sec,
    }


def save_scoreboard(data: dict, path: str = "outputs/eval_scoreboard.json") -> None:
    """Save eval results dict to JSON. Creates parent dirs."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"[model_eval] Scoreboard saved to {out_path}")
