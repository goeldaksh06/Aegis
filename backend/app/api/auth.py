from __future__ import annotations

import re

from fastapi import APIRouter, Depends, HTTPException, status

from app.auth.dependencies import get_current_user
from app.auth.security import create_access_token, hash_password, verify_password
from app.database.db import User, create_user, get_user_by_email
from app.models.schemas import LoginRequest, RegisterRequest, TokenResponse, UserOut

router = APIRouter(prefix="/auth", tags=["auth"])

_EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _user_out(user: User) -> UserOut:
    return UserOut(id=user.id, email=user.email, created_at=user.created_at)


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(request: RegisterRequest) -> TokenResponse:
    email = request.email.strip().lower()
    if not _EMAIL_PATTERN.match(email):
        raise HTTPException(status_code=422, detail="Invalid email address.")

    if await get_user_by_email(email) is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="An account with this email already exists.")

    user = await create_user(email=email, hashed_password=hash_password(request.password))
    token = create_access_token(user_id=user.id, email=user.email)
    return TokenResponse(access_token=token, user=_user_out(user))


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest) -> TokenResponse:
    email = request.email.strip().lower()
    user = await get_user_by_email(email)

    # Deliberately identical error for "no such user" and "wrong password" — a distinct
    # message for "no such user" would let an attacker enumerate registered email addresses.
    invalid_credentials = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password."
    )
    if user is None or not verify_password(request.password, user.hashed_password):
        raise invalid_credentials

    token = create_access_token(user_id=user.id, email=user.email)
    return TokenResponse(access_token=token, user=_user_out(user))


@router.get("/me", response_model=UserOut)
async def me(current_user: User = Depends(get_current_user)) -> UserOut:
    return _user_out(current_user)
