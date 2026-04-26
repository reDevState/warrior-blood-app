import asyncio
from sqlalchemy import select
from backend.database import AsyncSessionLocal, create_tables
from backend.models import User
from backend.auth import pwd_context

async def seed():
    await create_tables()
    async with AsyncSessionLocal() as db:
        for username, role, password in [
            ("test_chw", "CHW", "chwpassword"),
            ("test_patient", "PATIENT", "testpassword"),
        ]:
            existing = await db.execute(select(User).where(User.username == username))
            if existing.scalar_one_or_none():
                print(f"  skip {username} (already exists)")
                continue
            db.add(User(
                username=username,
                hashed_password=pwd_context.hash(password),
                role=role,
            ))
            print(f"  created {username}")
        await db.commit()
    print("Seed complete.")

if __name__ == "__main__":
    asyncio.run(seed())
