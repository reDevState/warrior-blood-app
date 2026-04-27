# CLAUDE.md — Warrior Blood coding standards

Claude Code reads this file at the start of every session.
Follow every rule here without being asked.

---

## Project overview

Warrior Blood is an MSc Software Engineering dissertation project (Renee Tucker,
University of Greater Manchester, 2025–26). It is a Sickle Cell Disease (SCD)
patient monitoring system: FastAPI backend + Streamlit CHW dashboard + rule-based
ML predictor (replace with LightGBM ONNX after training).

**Stack:** Python 3.12 (Docker) / 3.13 (local), FastAPI, SQLAlchemy 2 async,
PostgreSQL (asyncpg), Alembic, Pydantic v2, Streamlit, pytest-asyncio.

---

## Python style

- `from __future__ import annotations` at the top of every module.
- Type-annotate all function signatures. Use `X | None` (not `Optional[X]`).
- No comments explaining *what* code does. Only add a comment when the *why*
  is non-obvious (hidden constraint, clinical reference, workaround).
- No multi-paragraph docstrings. One-line module docstring is enough; function
  docstrings only when the purpose is not obvious from the signature.
- No unused imports. Keep imports sorted: stdlib → third-party → local.
- `from __future__ import annotations` means forward references in type hints
  work without quotes everywhere except explicit string usage.

---

## FastAPI conventions

- Every route has a `tags=[...]` argument and a one-line docstring.
- All DB access goes through `db: AsyncSession = Depends(get_db)`.
- Never import `AsyncSessionLocal` inside a route — only through `get_db`.
- Inline imports inside route functions are forbidden; put them at module top.
- `await db.flush()` after `db.add()` to get the generated PK before returning.
- `await db.get(Model, pk)` for single-row PK lookups.
- Return HTTP 404 with `detail="Patient not found"` (exact string) for missing
  patients — tests assert on this string.

---

## SQLAlchemy / models

- All PKs are UUID strings via `default=_uuid` (see `models.py`).
- PHI fields (name, phone) are Fernet-encrypted before any DB write.
  Never store plaintext PHI. Use `encrypt_phi()` / `decrypt_phi()` from `auth.py`.
- `Mapped[X | None]` with `nullable=True` for optional columns.
- `Mapped[bool]` columns always set `default=False` or `default=True` explicitly.
- Every new model needs a `back_populates` relationship on `Patient` with
  `cascade="all, delete-orphan"`.
- `datetime.utcnow` (not `datetime.now`) for `default=` on DateTime columns.
  (The deprecation warning is known; fix when upgrading to Python 3.13 stdlib
  timezone-aware datetimes in a dedicated task.)

---

## Pydantic schemas

- All request schemas validate at the API boundary. Never trust raw input.
- Use `Field(..., ge=0, le=10)` etc. for clinical range enforcement.
- `@field_validator` with `@classmethod` for multi-value validation (e.g.
  pain location codes, drink type allowlist).
- Response schemas: explicit field listing, no `model_config = {"extra": "allow"}`.
- `from_attributes = True` only when using `model_validate(orm_obj)`.

---

## Alembic migrations

- Never use `alembic revision --autogenerate` against the running container when
  `create_tables()` has already created the schema — it will produce an empty
  migration. Write migrations by hand instead.
- Migration filename format: `{hex_id}_{snake_case_description}.py`
- New table → use `op.create_table(...)` + `op.create_index(...)`.
- New columns on existing table → use `op.add_column(...)` (not recreate).
- After `docker compose up -d`, run `alembic stamp head` only if `create_tables()`
  beat Alembic to the schema. Otherwise run `alembic upgrade head` normally.
- Every migration has a matching `downgrade()` that reverses the change exactly.
- Chain: initial → users → pain_diary → hydration → weather_fields → …
  Each revision's `down_revision` must point to the previous revision ID.

---

## Authentication & security

- JWT: HS256, 30-minute expiry, `{"sub": username, "role": role}` payload.
- Passwords: bcrypt via `passlib.CryptContext`. Never log or return hashes.
- `require_chw` dependency enforces CHW/admin role on sensitive routes.
- Seed users (`test_chw`, `test_patient`) have `patient_id=None` — the FK is
  nullable and nothing in the auth flow reads it.
- Never commit real secrets. `.env` is in `.gitignore`.

---

## Testing

- Test DB: `sqlite+aiosqlite:///:memory:` — never the dev PostgreSQL container.
- All DB access in tests goes through `app.dependency_overrides[get_db]`.
- `TestSessionLocal` (not `AsyncSessionLocal`) for any direct session use in tests.
- `setup_db` fixture: `autouse=True`, creates all tables, seeds `_SEED_USERS`,
  overrides `get_db`, drops all tables on teardown.
- No mocking of `authenticate_user` — tests use real bcrypt via the seeded users.
- `_patient_token(client)` and `_chw_token(client)` helpers for auth in tests.
- HTTP tests use `AsyncClient(transport=ASGITransport(app=app), base_url="http://test")`.
- External HTTP calls (weather, SMS) return offline/stub values when API keys are
  absent — tests never hit real external services.
- Coverage targets: backend ≥ 79%, ml ≥ 91%, hydration 100%, pain_analysis 100%.

---

## Docker / environment

- `docker compose build api` after any change to Python files or `requirements.txt`.
- `docker compose up -d` starts db → redis → api → dashboard in dependency order.
- After rebuilding the api image, always run `alembic upgrade head` (or `stamp head`
  if `create_tables()` already ran).
- `DATABASE_URL` format: `postgresql+asyncpg://user:pass@db:5432/dbname` (asyncpg
  for the app). Alembic `env.py` strips `+asyncpg` → `+psycopg2` for its sync engine.
- `FERNET_KEY` must be a valid URL-safe base64 Fernet key. If absent, a session key
  is generated — data won't survive restarts.

---

## Git workflow

- Branch: `feature/*` → `develop` → `main`.
- Commit format: `type(scope): description` (e.g. `feat(pain): ...`, `fix(docker): ...`).
- Tag format: `vX.Y.Z` with annotated message `vX.Y.Z — short description`.
- Always push the tag separately: `git push origin vX.Y.Z`.
- `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>` in every commit.

---

## Clinical references (cite when relevant)

- Machado et al. (2024) — LightGBM on SCD prediction
- Brandow et al. (2020) — pain trajectories in SCD
- Smith et al. (2008) — breakthrough pain and hospitalisation
- Yallop et al. (2007) — dehydration, AQI and SCD hospitalisation
- Nolan et al. (2008) — temperature and SCD hospitalisation
- Wahl et al. (2018) — interpretability in LMIC health AI
- Lundberg & Lee (2017) — SHAP values
- Armstrong (1994) — urine colour chart (1–8 scale)
- WHO (2005) — oral rehydration guidelines
