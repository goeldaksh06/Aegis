from __future__ import annotations

from fastapi import Header, HTTPException, status

from app.auth.security import InvalidTokenError, decode_access_token
from app.database.db import User, get_user_by_id


def _extract_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip() or None


async def get_current_user(authorization: str | None = Header(default=None)) -> User:
    """Required auth — raises 401 if there is no valid bearer token.

    Used by endpoints that expose personal data (mission history, mission detail) where
    anonymous access must never be allowed. The user id always comes from the verified JWT,
    never from a request body/query param — a client cannot claim to be a different user.
    """
    token = _extract_bearer_token(authorization)
    if token is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated.")

    try:
        payload = decode_access_token(token)
    except InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token.")

    user = await get_user_by_id(payload["sub"])
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User no longer exists.")

    return user


async def get_current_user_optional(authorization: str | None = Header(default=None)) -> User | None:
    """Best-effort auth — returns None instead of raising when no/invalid token is present.

    Used on /chat and /chat/stream so the existing zero-friction anonymous demo (auto-run on
    page load, scenario chips, no login wall) keeps working exactly as before, while a logged-
    in user's requests still get attributed to their account for personal mission history.
    """
    token = _extract_bearer_token(authorization)
    if token is None:
        return None

    try:
        payload = decode_access_token(token)
    except InvalidTokenError:
        return None

    return await get_user_by_id(payload["sub"])
