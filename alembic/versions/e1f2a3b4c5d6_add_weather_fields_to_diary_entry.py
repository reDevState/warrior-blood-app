"""add weather fields to DiaryEntry

Revision ID: e1f2a3b4c5d6
Revises: d1e2f3a4b5c6
Create Date: 2026-04-27 00:00:00.000000

ambient_temp_c and humidity_pct already exist from the initial schema.
This migration adds the six new weather columns.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e1f2a3b4c5d6'
down_revision: Union[str, None] = 'd1e2f3a4b5c6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("diary_entries", sa.Column("feels_like_c", sa.Float(), nullable=True))
    op.add_column("diary_entries", sa.Column("aqi", sa.Integer(), nullable=True))
    op.add_column("diary_entries", sa.Column("pm25_ugm3", sa.Float(), nullable=True))
    op.add_column("diary_entries", sa.Column("cold_stress_alert", sa.Boolean(), nullable=True))
    op.add_column("diary_entries", sa.Column("heat_stress_alert", sa.Boolean(), nullable=True))
    op.add_column("diary_entries", sa.Column("aqi_alert", sa.Boolean(), nullable=True))


def downgrade() -> None:
    op.drop_column("diary_entries", "aqi_alert")
    op.drop_column("diary_entries", "heat_stress_alert")
    op.drop_column("diary_entries", "cold_stress_alert")
    op.drop_column("diary_entries", "pm25_ugm3")
    op.drop_column("diary_entries", "aqi")
    op.drop_column("diary_entries", "feels_like_c")
