import os
import secrets

from fastapi import Header, HTTPException


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """
    Gate an endpoint behind a shared secret.

    The frontend (a server-side proxy) sends the secret as the `X-API-Key`
    header; the secret lives only in the API's and the frontend server's
    environments and never reaches browsers.

    Fails closed: if the server has no key configured, every request is
    rejected rather than silently running unprotected.
    """
    expected = os.environ.get("VOICEMATCH_API_KEY")
    if not expected:
        raise HTTPException(status_code=500, detail="server_auth_misconfigured")
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="invalid_api_key")
