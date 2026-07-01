"""
Rate limiter and retry logic for AWS Bedrock API calls.

Handles throttling errors (429) with exponential backoff retry
and limits concurrent requests to stay within Bedrock's rate limits.
"""

import time
import threading
from functools import wraps

from botocore.exceptions import ClientError


class RateLimiter:
    """
    Token bucket rate limiter for controlling request frequency.

    Limits the number of requests per second to avoid Bedrock throttling.
    """

    def __init__(self, max_requests_per_second: float = 5.0):
        self.max_requests = max_requests_per_second
        self.interval = 1.0 / max_requests_per_second
        self.last_request_time = 0.0
        self._lock = threading.Lock()

    def wait(self):
        """Block until a request slot is available."""
        with self._lock:
            now = time.time()
            elapsed = now - self.last_request_time
            if elapsed < self.interval:
                sleep_time = self.interval - elapsed
                time.sleep(sleep_time)
            self.last_request_time = time.time()


# Global rate limiter instance — shared across all Bedrock calls.
bedrock_rate_limiter = RateLimiter(max_requests_per_second=5.0)


def retry_with_backoff(
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    throttle_codes: tuple = ("ThrottlingException", "TooManyRequestsException", "ServiceUnavailableException"),
):
    """
    Decorator that retries a function on Bedrock throttling errors
    with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay in seconds before first retry.
        max_delay: Maximum delay cap in seconds.
        throttle_codes: AWS error codes that trigger a retry.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    # Apply rate limiting before each request.
                    bedrock_rate_limiter.wait()
                    return func(*args, **kwargs)
                except ClientError as e:
                    error_code = e.response["Error"]["Code"]
                    if error_code in throttle_codes and attempt < max_retries:
                        # Exponential backoff: 1s, 2s, 4s, 8s, 16s...
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        print(
                            f"  [Rate Limiter] Throttled ({error_code}). "
                            f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})..."
                        )
                        time.sleep(delay)
                    else:
                        raise
            return func(*args, **kwargs)
        return wrapper
    return decorator
