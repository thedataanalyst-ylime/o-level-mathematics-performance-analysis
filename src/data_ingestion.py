from __future__ import annotations

import sqlite3
from pathlib import Path
import pandas as pd

from . import config


def list_sqlite_tables(database_path: str | Path) -> list[str]:
    database_path = Path(database_path)

    with sqlite3.connect(database_path) as connection:
        query = """
        SELECT name
        FROM sqlite_master
        WHERE type='table'
          AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
        return pd.read_sql_query(query, connection)["name"].tolist()


def resolve_table_name(
    database_path: str | Path,
    table_name: str | None = None,
) -> str:
    tables = list_sqlite_tables(database_path)

    if not tables:
        raise ValueError(f"No user tables found in {database_path}.")

    if table_name is not None:
        if table_name not in tables:
            raise ValueError(
                f"Table '{table_name}' not found. Available tables: {tables}"
            )
        return table_name

    if len(tables) == 1:
        return tables[0]

    raise ValueError(
        "Multiple user tables found. "
        f"Available tables: {tables}. "
        "Set TABLE_NAME in src/config.py or pass --table."
    )


def load_source_data(
    database_path: str | Path = config.DATABASE_PATH,
    table_name: str | None = config.TABLE_NAME,
) -> tuple[pd.DataFrame, str]:
    database_path = Path(database_path)

    if not database_path.exists():
        raise FileNotFoundError(f"SQLite database not found: {database_path}")

    resolved_table = resolve_table_name(database_path, table_name)

    with sqlite3.connect(database_path) as connection:
        df = pd.read_sql_query(
            f'SELECT * FROM "{resolved_table}"',
            connection,
        )

    return df, resolved_table
