"""
Token usage tracker for monitoring Bedrock API costs.

Tracks input/output tokens per session and provides cost estimates
based on Claude Haiku 4.5 pricing.
"""


class TokenTracker:
    """
    Tracks cumulative token usage across a session.

    Pricing (Claude Haiku 4.5 on Bedrock):
      - Input:  $1.00 / 1M tokens
      - Output: $5.00 / 1M tokens
    """

    INPUT_COST_PER_MILLION = 1.00   # USD
    OUTPUT_COST_PER_MILLION = 5.00  # USD

    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        """Record token usage from a single API call."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.call_count += 1

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def estimated_cost_usd(self) -> float:
        """Estimate the session cost in USD."""
        input_cost = (self.total_input_tokens / 1_000_000) * self.INPUT_COST_PER_MILLION
        output_cost = (self.total_output_tokens / 1_000_000) * self.OUTPUT_COST_PER_MILLION
        return input_cost + output_cost

    def stats(self) -> dict:
        """Return usage statistics."""
        return {
            "calls": self.call_count,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost": f"${self.estimated_cost_usd:.4f}",
        }

    def reset(self):
        """Reset the tracker."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0


# Global tracker instance.
token_tracker = TokenTracker()
