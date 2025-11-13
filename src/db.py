"""Database models and helpers."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterable, Sequence

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from .config import get_settings

logger = logging.getLogger(__name__)

metadata = MetaData()
Base = declarative_base(metadata=metadata)


class Measurement(Base):
    __tablename__ = "measurements"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), unique=True, nullable=False)
    pm25 = Column(Float, nullable=False)
    temp = Column(Float)
    humidity = Column(Float)
    wind_speed = Column(Float)


class FeatureRow(Base):
    __tablename__ = "features"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), unique=True, nullable=False)
    data = Column(JSON, nullable=False)


class Forecast(Base):
    __tablename__ = "forecasts"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    model = Column(String(64), nullable=False)
    horizon = Column(Integer, nullable=False)
    yhat = Column(Float, nullable=False)


class Metric(Base):
    __tablename__ = "metrics"
    id = Column(Integer, primary_key=True, autoincrement=True)
    model = Column(String(64), nullable=False)
    metric = Column(String(32), nullable=False)
    value = Column(Float, nullable=False)
    details = Column(JSON)


def get_engine(echo: bool = False):
    settings = get_settings()
    engine = create_engine(settings.db_url, echo=echo, future=True)
    return engine


def create_all() -> None:
    engine = get_engine()
    Base.metadata.create_all(engine)
    logger.info("Ensured database tables exist.")


def get_session() -> Session:
    engine = get_engine()
    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)
    return SessionLocal()


@contextmanager
def session_scope():
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _conflict_columns(model) -> list[str]:
    """Return column names that have unique constraints."""
    table = model.__table__
    # Explicit table-specific rules
    if table.name == "forecasts":
        return ["timestamp", "model", "horizon"]
    # Use SQLAlchemy column metadata (unique=True) if available
    cols = [col.name for col in table.columns if getattr(col, "unique", False)]
    if cols:
        return cols
    # Fallback to primary key
    pk_cols = [col.name for col in table.primary_key]
    return pk_cols or ["id"]


def upsert_batch(model, rows: Sequence[dict]) -> None:
    if not rows:
        return
    engine = get_engine()
    with engine.begin() as conn:
        if engine.dialect.name == "postgresql":
            stmt = pg_insert(model).values(rows)
            primary = stmt.excluded
            update_cols = {col.name: getattr(primary, col.name) for col in model.__table__.columns if col.name not in {"id"}}
            conflict_cols = _conflict_columns(model)
            upsert_stmt = stmt.on_conflict_do_update(index_elements=conflict_cols, set_=update_cols)
            conn.execute(upsert_stmt)
        else:
            for row in rows:
                conn.execute(model.__table__.insert().prefix_with("OR REPLACE"), row)
    logger.info("Upserted %s rows into %s", len(rows), model.__tablename__)
