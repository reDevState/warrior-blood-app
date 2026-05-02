"""add email_enc to patients

Revision ID: f1a2b3c4d5e6
Revises: e1f2a3b4c5d6
Create Date: 2026-05-02 00:00:00.000000

Adds Fernet-encrypted email column to support patient self-service
login (lookup by email instead of UUID).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'f1a2b3c4d5e6'
down_revision: Union[str, None] = 'e1f2a3b4c5d6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('patients', sa.Column('email_enc', sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column('patients', 'email_enc')
