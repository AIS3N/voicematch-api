import os
import httpx
from fastapi import HTTPException

RATE_LIMIT = 3


async def check_rate_limit(ip: str) -> int:
    """
    Increment the request counter for the given IP.
    Raises 429 if the limit is exceeded.
    Returns the number of remaining requests for today.
    """
    url = os.environ["UPSTASH_REDIS_REST_URL"]
    token = os.environ["UPSTASH_REDIS_REST_TOKEN"]
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient() as client:
        incr_res = await client.post(
            f"{url}/incr/rl:{ip}",
            headers=headers,
        )
        incr_res.raise_for_status()
        count = incr_res.json()["result"]

        if count == 1:
            await client.post(
                f"{url}/expire/rl:{ip}/86400",
                headers=headers,
            )

    if count > RATE_LIMIT:
        raise HTTPException(status_code=429, detail="rate_limit_exceeded")

    return max(0, RATE_LIMIT - count)
