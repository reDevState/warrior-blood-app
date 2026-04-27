"""add HydrationEntry model

Revision ID: d1e2f3a4b5c6
Revises: c1a2b3d4e5f6
Create Date: 2026-04-27 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd1e2f3a4b5c6'
down_revision: Union[str, None] = 'c1a2b3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "hydration_entries",
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("patient_id", sa.String(36), nullable=False),
        sa.Column("logged_at", sa.DateTime(), nullable=True),
        sa.Column("entry_date", sa.Date(), nullable=False),
        sa.Column("drink_type", sa.String(32), nullable=False),
        sa.Column("drink_volume_ml", sa.Integer(), nullable=False),
        sa.Column("daily_total_ml", sa.Integer(), nullable=True),
        sa.Column("urine_colour", sa.Integer(), nullable=True),
        sa.Column("thirst_level", sa.Integer(), nullable=True),
        sa.Column("dry_mouth", sa.Boolean(), nullable=True),
        sa.Column("dizziness", sa.Boolean(), nullable=True),
        sa.Column("headache", sa.Boolean(), nullable=True),
        sa.Column("dark_urine_flag", sa.Boolean(), nullable=True),
        sa.ForeignKeyConstraint(["patient_id"], ["patients.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_hydration_entries_patient_id", "hydration_entries", ["patient_id"])


def downgrade() -> None:
    op.drop_index("ix_hydration_entries_patient_id", table_name="hydration_entries")
    op.drop_table("hydration_entries")
