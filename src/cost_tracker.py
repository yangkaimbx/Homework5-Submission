"""
Cost Tracker for API Usage

Tracks token usage and costs across different LLM models.
Updated for Week 4 with claude-sonnet-4-6 as default.
"""

from typing import Dict, List, Any
from datetime import datetime


class CostTracker:
    """Track API costs across different models"""

    # Pricing per 1M tokens (as of 2025)
    PRICING = {
        # Claude 4.6 / 4.5 models
        "claude-sonnet-4-6":          {"input": 3.0,  "output": 15.0},
        "claude-opus-4-6":            {"input": 15.0, "output": 75.0},
        "claude-haiku-4-5-20251001":  {"input": 1.0,  "output": 5.0},
        # Legacy aliases
        "claude-sonnet-4-5-20250929": {"input": 3.0,  "output": 15.0},
        "claude-opus-4-5-20251101":   {"input": 15.0, "output": 75.0},
        # Ollama (free)
        "ollama": {"input": 0.0, "output": 0.0},
    }

    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.calls = []

    def add_call(self, response: Dict[str, Any]):
        """
        Add an API call to the tracker.

        Args:
            response: Response dict from LLMClient.generate() or generate_with_thinking()
        """
        if "error" in response:
            return

        model = response['model']
        input_tokens = response['usage']['input_tokens']
        output_tokens = response['usage']['output_tokens']

        # Resolve pricing
        if model in self.PRICING:
            pricing = self.PRICING[model]
        elif any(x in model.lower() for x in ["ollama", "llama", "mistral", "qwen", "phi"]):
            pricing = self.PRICING["ollama"]
        else:
            pricing = self.PRICING["claude-sonnet-4-6"]

        input_cost  = (input_tokens  / 1_000_000) * pricing['input']
        output_cost = (output_tokens / 1_000_000) * pricing['output']
        call_cost   = input_cost + output_cost

        self.total_input_tokens  += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost          += call_cost

        self.calls.append({
            'model': model,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'cost': call_cost,
            'timestamp': datetime.now(),
        })

    def report(self, detailed: bool = False):
        """Print a cost report."""
        print("=" * 60)
        print("API COST REPORT")
        print("=" * 60)
        print(f"Total API calls:     {len(self.calls)}")
        print(f"Total input tokens:  {self.total_input_tokens:,}")
        print(f"Total output tokens: {self.total_output_tokens:,}")
        print(f"Total cost:          ${self.total_cost:.4f}")
        print()

        if self.calls:
            calls_to_show = self.calls if detailed else self.calls[-5:]
            label = "All calls" if detailed else f"Last {len(calls_to_show)} calls"
            print(f"{label}:")
            for i, call in enumerate(calls_to_show, 1):
                ts = call['timestamp'].strftime("%H:%M:%S")
                short_model = call['model'].split('-')[1] if '-' in call['model'] else call['model'][:15]
                print(f"  {i}. [{ts}] {short_model} -- "
                      f"{call['input_tokens']}in/{call['output_tokens']}out -- "
                      f"${call['cost']:.4f}")

        print("=" * 60)

    def reset(self):
        """Reset all tracking data."""
        self.total_input_tokens  = 0
        self.total_output_tokens = 0
        self.total_cost          = 0.0
        self.calls               = []
        print("✓ Cost tracker reset")

    def get_summary(self) -> Dict[str, Any]:
        """Return summary as a dictionary."""
        return {
            "total_calls":           len(self.calls),
            "total_input_tokens":    self.total_input_tokens,
            "total_output_tokens":   self.total_output_tokens,
            "total_cost":            self.total_cost,
            "average_cost_per_call": self.total_cost / len(self.calls) if self.calls else 0,
        }
