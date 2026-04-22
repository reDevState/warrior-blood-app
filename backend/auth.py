"""
auth.py — JWT issuance/verification and Fernet PHI encryption.

JWT: python-jose with HS256 (upgrade to RSA in production).
Encryption: cryptography.fernet.Fernet (AES-128-CBC + HMAC-SHA256).

PHI fields (patient name, phone) are encrypted before any DB write and
decrypted only at the application layer — never stored in plaintext.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from cryptography.fernet import Fernet
from jose import JWTError, jwt
from passlib.context import CryptContext

SECRET_KEY = os.getenv("SECRET_KEY", "warrior-blood-mvp-dev-key-change-in-production-32c")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Fernet key for PHI encryption — in production, load from secret manager
_raw_key = os.getenv("FERNET_KEY", "")
if _raw_key:
    _fernet = Fernet(_raw_key.encode())
else:
    # Generate a session key for MVP — data won't survive restarts
    _fernet = Fernet(Fernet.generate_key())

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# MVP: in-memory user store — replace with DB lookup in production
_USERS: dict[str, dict] = {
    "test_patient": {
        "hashed_password": pwd_context.hash("testpassword"),
        "role": "patient",
        "patient_id": "mvp-patient-001",
    },
    "test_chw": {
        "hashed_password": pwd_context.hash("chwpassword"),
        "role": "chw",
        "patient_id": None,
    },
}


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plaintext password against a bcrypt hash."""
    return pwd_context.verify(plain, hashed)


def authenticate_user(username: str, password: str) -> dict | None:
    """
    Authenticate a user by username and password.

    Args:
        username: Registered username.
        password: Plaintext password.

    Returns:
        User dict if valid, None otherwise.
    """
    user = _USERS.get(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return {"username": username, **user}


def create_access_token(data: dict, expires_minutes: int = ACCESS_TOKEN_EXPIRE_MINUTES) -> str:
    """
    Issue a signed JWT access token.

    Args:
        data: Payload dict (will include "exp" claim automatically).
        expires_minutes: Token lifetime in minutes.

    Returns:
        Encoded JWT string.
    """
    payload = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=expires_minutes)
    payload["exp"] = expire
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    """
    Decode and verify a JWT, raising JWTError if invalid or expired.

    Args:
        token: Encoded JWT string.

    Returns:
        Decoded payload dict.

    Raises:
        JWTError: If token is invalid, expired, or tampered.
    """
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])


def encrypt_phi(plaintext: str) -> str:
    """
    Encrypt a PHI string (name, phone) with Fernet AES-128.

    Args:
        plaintext: Raw PHI value.

    Returns:
        URL-safe base64-encoded ciphertext string.
    """
    return _fernet.encrypt(plaintext.encode()).decode()


def decrypt_phi(ciphertext: str) -> str:
    """
    Decrypt a Fernet-encrypted PHI string.

    Args:
        ciphertext: Fernet-encrypted base64 string.

    Returns:
        Decrypted plaintext.
    """
    return _fernet.decrypt(ciphertext.encode()).decode()
