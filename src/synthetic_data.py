"""synthetic_data.py — Synthetic instruction dataset builder for fine-tuning homework (Week 5)."""

from __future__ import annotations

import json
from typing import Any


# ---------------------------------------------------------------------------
# Pricing constants (Claude claude-sonnet-4-6 / claude-opus-4-7 ballpark)
# ---------------------------------------------------------------------------

_PRICE_PER_INPUT_TOKEN = 3e-6    # $3 per 1M input tokens
_PRICE_PER_OUTPUT_TOKEN = 15e-6  # $15 per 1M output tokens
_WARN_THRESHOLD_USD = 1.0
_ABORT_THRESHOLD_USD = 5.0


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


class SyntheticDatasetBuilder:
    """
    Distilabel-inspired synthetic instruction dataset builder using Claude API.

    Pipeline: seed_from_corpus() -> expand() -> critique_and_filter() -> save()
    Cost guard: warns at $1 total spend, aborts at $5.
    """

    def __init__(self, llm_client: Any, max_budget_usd: float = _ABORT_THRESHOLD_USD) -> None:
        self._client = llm_client
        self._max_budget = max_budget_usd
        self._spent_usd: float = 0.0
        print(f"[synthetic_data] SyntheticDatasetBuilder initialised (budget cap: ${max_budget_usd:.2f})")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call(self, prompt: str, system: str = "", max_tokens: int = 1024) -> str:
        """Send a prompt to the LLM client and return the text response."""
        self._check_budget(prompt)
        messages = [{"role": "user", "content": prompt}]
        kwargs: dict = {"messages": messages, "max_tokens": max_tokens}
        if system:
            kwargs["system"] = system
        response = self._client.messages.create(model=self._client.default_model, **kwargs)
        text: str = response.content[0].text
        # Approximate cost accounting
        in_toks = _estimate_tokens(prompt + system)
        out_toks = _estimate_tokens(text)
        cost = in_toks * _PRICE_PER_INPUT_TOKEN + out_toks * _PRICE_PER_OUTPUT_TOKEN
        self._spent_usd += cost
        self._warn_if_needed()
        return text

    def _check_budget(self, prompt: str) -> None:
        """Abort before a call if already over budget."""
        if self._spent_usd >= self._max_budget:
            raise RuntimeError(
                f"[synthetic_data] Budget cap of ${self._max_budget:.2f} reached "
                f"(spent: ${self._spent_usd:.4f}). Aborting."
            )

    def _warn_if_needed(self) -> None:
        if self._spent_usd >= _WARN_THRESHOLD_USD:
            print(
                f"[synthetic_data] WARNING: cumulative spend ${self._spent_usd:.4f} "
                f"has crossed the ${_WARN_THRESHOLD_USD:.2f} warning threshold."
            )

    def _parse_json_list(self, raw: str, fallback_key: str = "items") -> list[dict]:
        """Extract a JSON array from an LLM response that may have markdown fences."""
        text = raw.strip()
        # Strip ```json ... ``` fences if present
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict) and fallback_key in parsed:
                return parsed[fallback_key]
        except json.JSONDecodeError:
            pass
        # Best-effort: find the first '[' to ']'
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                pass
        print(f"[synthetic_data] Could not parse JSON list from response; returning empty list.")
        return []

    # ------------------------------------------------------------------
    # Public pipeline methods
    # ------------------------------------------------------------------

    def seed_from_corpus(self, corpus_text: str, n_seeds: int = 20) -> list[dict]:
        """Generate n_seeds Q&A pairs from corpus_text using the LLM."""
        print(f"[synthetic_data] Generating {n_seeds} seed Q&A pairs from corpus ({len(corpus_text)} chars)...")
        system = (
            "You are a dataset curator. Produce high-quality instruction-tuning examples "
            "grounded strictly in the provided text."
        )
        prompt = (
            f"Read the following text and generate exactly {n_seeds} diverse question-answer pairs "
            "that cover the most important facts, skills, and details in it.\n\n"
            "Return ONLY a JSON array where each element is:\n"
            '  {"question": "<question>", "answer": "<detailed answer>"}\n\n'
            "No markdown outside the JSON array.\n\n"
            f"TEXT:\n{corpus_text[:6000]}"
        )
        raw = self._call(prompt, system=system, max_tokens=4096)
        seeds = self._parse_json_list(raw)
        results: list[dict] = []
        for i, item in enumerate(seeds[:n_seeds]):
            record = {
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "source": "seed",
            }
            results.append(record)
            print(f"  [{i+1}/{n_seeds}] Q: {record['question'][:80]}")
        print(f"[synthetic_data] seed_from_corpus done: {len(results)} seeds (spent so far: ${self._spent_usd:.4f})")
        return results

    def expand(self, seeds: list[dict], factor: int = 3) -> list[dict]:
        """Generate `factor` variations per seed; return seeds + all variations."""
        print(f"[synthetic_data] Expanding {len(seeds)} seeds by factor {factor}...")
        all_records: list[dict] = list(seeds)

        for i, seed in enumerate(seeds):
            self._check_budget("")
            prompt = (
                f"Given this Q&A pair, generate {factor} diverse variations. "
                "Rephrase the question differently each time and elaborate or reframe the answer.\n\n"
                f"Original question: {seed['question']}\n"
                f"Original answer: {seed['answer']}\n\n"
                "Return ONLY a JSON array, each element:\n"
                '  {"question": "<rephrased>", "answer": "<elaborated>"}\n'
                "No markdown outside the array."
            )
            raw = self._call(prompt, max_tokens=2048)
            variations = self._parse_json_list(raw)
            count = 0
            for item in variations[:factor]:
                all_records.append({
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "source": f"expand_seed_{i}",
                })
                count += 1
            print(f"  Seed {i+1}/{len(seeds)}: added {count} variations (spent: ${self._spent_usd:.4f})")

        print(
            f"[synthetic_data] expand done: {len(seeds)} seeds + {len(all_records)-len(seeds)} variations "
            f"= {len(all_records)} total"
        )
        return all_records

    def critique_and_filter(self, records: list[dict], min_score: float = 3.0) -> list[dict]:
        """Score each record 1-5 for quality; keep those with score >= min_score."""
        print(f"[synthetic_data] Scoring {len(records)} records (min_score={min_score})...")
        scored: list[dict] = []
        score_counts: dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

        for i, rec in enumerate(records):
            self._check_budget("")
            prompt = (
                "Rate this Q&A pair on a scale of 1 to 5 for quality:\n"
                "  1 = irrelevant or factually wrong\n"
                "  3 = adequate, somewhat specific\n"
                "  5 = highly relevant, specific, and factually grounded\n\n"
                f"Question: {rec['question']}\n"
                f"Answer: {rec['answer']}\n\n"
                'Respond with ONLY a JSON object: {"score": <1-5>, "reason": "<one sentence>"}'
            )
            raw = self._call(prompt, max_tokens=256)
            try:
                text = raw.strip()
                if text.startswith("```"):
                    lines = text.splitlines()
                    text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
                obj = json.loads(text)
                score = float(obj.get("score", 0))
                reason = obj.get("reason", "")
            except (json.JSONDecodeError, ValueError):
                score = 0.0
                reason = "parse error"
            int_score = max(1, min(5, round(score)))
            score_counts[int_score] = score_counts.get(int_score, 0) + 1
            record_with_score = {**rec, "score": score, "score_reason": reason}
            scored.append(record_with_score)
            if (i + 1) % 10 == 0 or i == len(records) - 1:
                print(f"  Scored {i+1}/{len(records)} (spent: ${self._spent_usd:.4f})")

        kept = [r for r in scored if r["score"] >= min_score]
        print(f"[synthetic_data] Score distribution: {score_counts}")
        print(f"[synthetic_data] critique_and_filter: kept {len(kept)}/{len(records)} records (score >= {min_score})")
        return kept

    def to_chatml(
        self,
        records: list[dict],
        system_prompt: str = "You are a helpful assistant.",
    ) -> list[dict]:
        """Convert {question, answer} records to {messages: [{role, content},...]} format."""
        converted: list[dict] = []
        for rec in records:
            converted.append({
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": rec["question"]},
                    {"role": "assistant", "content": rec["answer"]},
                ]
            })
        print(f"[synthetic_data] to_chatml: converted {len(converted)} records to ChatML message format")
        return converted

    def save(self, records: list[dict], path: str) -> None:
        """Save records to JSONL via data_prep.save_jsonl and print a summary."""
        from data_prep import save_jsonl  # local import — avoids circular dep at module level
        save_jsonl(records, path)
        print(
            f"[synthetic_data] Save complete: {len(records)} records -> {path} "
            f"(total spend: ${self._spent_usd:.4f})"
        )
