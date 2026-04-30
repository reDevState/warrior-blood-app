"""Seed default CHW and patient accounts for local development."""

from __future__ import annotations

import asyncio

from sqlalchemy import select

from backend.auth import pwd_context
from backend.database import AsyncSessionLocal, create_tables, engine
from backend.models import User

_SEED_USERS = [
    ("test_chw", "chw", "chwpassword"),
    ("test_patient", "patient", "testpassword"),
]


async def seed() -> None:
    await create_tables()
    async with AsyncSessionLocal() as db:
        for username, role, password in _SEED_USERS:
            existing = await db.scalar(select(User).where(User.username == username))
            if existing:
                print(f"  skip {username} (already exists)")
                continue
            db.add(User(
                username=username,
                hashed_password=pwd_context.hash(password),
                role=role,
            ))
            print(f"  created {username} ({role})")
        await db.commit()
    await engine.dispose()
    print("Seed complete.")


if __name__ == "__main__":
    asyncio.run(seed())
