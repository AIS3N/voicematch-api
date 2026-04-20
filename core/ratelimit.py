from collections import defaultdict
from fastapi import HTTPException

# High fallback limit to protect against direct API abuse
RATE_LIMIT = 200
_counts: dict[str, int] = defaultdict(int)


async def check_rate_limit(ip: str) -> None:
    """
    Simple in-memory fallback rate limit.
    This protects against direct API abuse only.
    """
    _counts[ip] += 1
    if _counts[ip] > RATE_LIMIT:
        raise HTTPException(status_code=429, detail="rate_limit_exceeded")
