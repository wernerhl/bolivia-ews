#!/usr/bin/env python3
import os
from typing import Iterable
import pandas as pd
from pandas.api import types as pdt
from supabase import create_client

def _normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for col in normalized.columns:
        series = normalized[col]
        if pdt.is_datetime64_any_dtype(series) or isinstance(
            series.dtype, pd.PeriodDtype
        ):
            if (
                getattr(series.dt, "tz", None) is None
                and (series.dt.hour == 0).all()
                and (series.dt.minute == 0).all()
                and (series.dt.second == 0).all()
                and (series.dt.microsecond == 0).all()
                and (series.dt.nanosecond == 0).all()
            ):
                normalized[col] = series.dt.strftime("%Y-%m-%d")
            else:
                normalized[col] = series.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    return normalized

def _chunk_iterable(items: list, size: int) -> Iterable[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]

def _supabase_client():
    sb_url = os.getenv("SUPABASE_URL")
    sb_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not sb_url:
        raise RuntimeError("Missing required environment variable: SUPABASE_URL")
    if not sb_key:
        raise RuntimeError(
            "Missing required environment variable: SUPABASE_SERVICE_ROLE_KEY"
        )
    return create_client(sb_url, sb_key)

def upload_dataset(
    name: str, df: pd.DataFrame, keys: list[str], chunk_size: int = 5000
) -> None:
    sb = _supabase_client()
    normalized = _normalize_dates(df)
    normalized = normalized.drop_duplicates(subset=keys, keep="last")
    records = normalized.to_dict(orient="records")
    on_conflict = ",".join(keys)
    for chunk in _chunk_iterable(records, chunk_size):
        sb.table(name).upsert(chunk, on_conflict=on_conflict).execute()
