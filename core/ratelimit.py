import os
import httpx
from fastapi import HTTPException

UPSTASH_REDIS_REST_URL = os.environ["UPSTASH_REDIS_REST_URL"]
UPSTASH_REDIS_REST_TOKEN = os.environ["UPSTASH_REDIS_REST_TOKEN"]

RATE_LIMIT = 3


async def check_rate_limit(ip: str) -> None:
    headers = {"Authorization": f"Bearer {UPSTASH_REDIS_REST_TOKEN}"}

    async with httpx.AsyncClient() as client:
        incr_res = await client.post(
            f"{UPSTASH_REDIS_REST_URL}/incr/rl:{ip}",
            headers=headers,
        )
        incr_res.raise_for_status()
        count = incr_res.json()["result"]

        if count == 1:
            await client.post(
                f"{UPSTASH_REDIS_REST_URL}/expire/rl:{ip}/86400",
                headers=headers,
            )

    if count > RATE_LIMIT:
        raise HTTPException(status_code=429, detail="rate_limit_exceeded")
