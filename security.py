"""
security.py
-----------
Authentication (JWT for users, static bearer API key for sensors), password
hashing, and rate limiting. Designed so every non-public route is protected
by at least one of the two auth dependencies below.
"""

import hmac
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from slowapi import Limiter
from slowapi.util import get_remote_address

from config import get_settings
from schemas import TokenPayload

settings = get_settings()

# ---------------------------------------------------------------------------
# Password hashing (for the demo /auth/token username+password flow).
# In production, back this with a real user store (DB), never inline dicts.
# ---------------------------------------------------------------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

bearer_scheme = HTTPBearer(auto_error=False)

# ---------------------------------------------------------------------------
# Rate limiter — keyed by client IP. Applied per-route via decorator in main.py.
# Protects both the public server and constrained edge/IoT devices from DoS.
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address, default_limits=[settings.RATE_LIMIT_DEFAULT])


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def create_access_token(subject: str, scope: str = "user", expires_minutes: Optional[int] = None) -> str:
    """Issue a signed, short-lived JWT. Never embed secrets/PII in the payload."""
    expire_minutes = expires_minutes or settings.ACCESS_TOKEN_EXPIRE_MINUTES
    expire = datetime.now(timezone.utc) + timedelta(minutes=expire_minutes)
    payload = {"sub": subject, "exp": int(expire.timestamp()), "scope": scope}
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> TokenPayload:
    try:
        raw = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        return TokenPayload(**raw)
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> TokenPayload:
    """Dependency for user-facing (dashboard/browser) routes — requires a valid JWT."""
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return decode_access_token(credentials.credentials)


async def require_admin(user: TokenPayload = Depends(get_current_user)) -> TokenPayload:
    """Dependency for privileged/system-control routes."""
    if user.scope != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin privileges required")
    return user


async def verify_sensor_api_key(request: Request) -> str:
    """
    Dependency for machine-to-machine sensor ingestion routes.
    Uses a constant-time comparison to avoid timing side-channel attacks,
    and reads the key from a header rather than a query param (keeps it out of logs/URLs).
    """
    api_key = request.headers.get("X-API-Key")
    if not api_key or not hmac.compare_digest(api_key, settings.SENSOR_API_KEY):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")
    return api_key


def generate_api_key() -> str:
    """Utility for provisioning new sensor API keys (run offline, store in secrets manager)."""
    return secrets.token_urlsafe(32)
