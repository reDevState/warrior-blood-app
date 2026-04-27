"""add PainDiaryEntry model

Revision ID: c1a2b3d4e5f6
Revises: b6f2810500ba
Create Date: 2026-04-27 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c1a2b3d4e5f6'
down_revision: Union[str, None] = 'b6f2810500ba'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "pain_diary_entries",
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("patient_id", sa.String(36), nullable=False),
        sa.Column("recorded_at", sa.DateTime(), nullable=True),
        sa.Column("pain_score", sa.Integer(), nullable=False),
        sa.Column("pain_locations", sa.String(128), nullable=True),
        sa.Column("trigger_cold", sa.Boolean(), nullable=True),
        sa.Column("trigger_stress", sa.Boolean(), nullable=True),
        sa.Column("trigger_exercise", sa.Boolean(), nullable=True),
        sa.Column("trigger_infection", sa.Boolean(), nullable=True),
        sa.Column("trigger_dehydration", sa.Boolean(), nullable=True),
        sa.Column("trigger_other", sa.String(128), nullable=True),
        sa.Column("took_paracetamol", sa.Boolean(), nullable=True),
        sa.Column("took_ibuprofen", sa.Boolean(), nullable=True),
        sa.Column("took_opioid", sa.Boolean(), nullable=True),
        sa.Column("pain_relief_rating", sa.Integer(), nullable=True),
        sa.Column("is_breakthrough", sa.Boolean(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["patient_id"], ["patients.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_pain_diary_entries_patient_id", "pain_diary_entries", ["patient_id"])


def downgrade() -> None:
    op.drop_index("ix_pain_diary_entries_patient_id", table_name="pain_diary_entries")
    op.drop_table("pain_diary_entries")
