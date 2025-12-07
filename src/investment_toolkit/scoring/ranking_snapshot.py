"""Utilities to compute and persist daily/weekly score rankings."""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Iterable, List, Optional, Sequence

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from investment_toolkit.utilities.config import DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER

LOGGER = logging.getLogger(__name__)

DAILY_SCOPE = "daily"
WEEKLY_SCOPE = "weekly"
DEFAULT_WEEKLY_METHODS = ["median"]
SUPPORTED_WEEKLY_METHODS = {"mean", "median", "sum", "max", "ewma"}



def build_engine() -> Engine:
    """Construct a SQLAlchemy engine from environment configuration."""
    uri = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(uri, future=True)


def parse_target_date(raw_value: str | date | datetime) -> date:
    """Parse supported target date representations into a date object."""
    if isinstance(raw_value, date) and not isinstance(raw_value, datetime):
        return raw_value
    if isinstance(raw_value, datetime):
        return raw_value.date()
    if isinstance(raw_value, str):
        return datetime.strptime(raw_value, "%Y-%m-%d").date()
    raise TypeError("target_date must be date, datetime, or YYYY-MM-DD string")


def get_recent_trading_dates(
    engine: Engine,
    target_date: date,
    window: int,
) -> List[date]:
    query = text(
        """
        SELECT DISTINCT date
        FROM backtest_results.daily_scores_v2
        WHERE date <= :target_date
        ORDER BY date DESC
        LIMIT :limit
        """
    )
    with engine.connect() as conn:
        frame = pd.read_sql_query(
            query,
            conn,
            params={"target_date": target_date, "limit": window},
        )
    if frame.empty:
        raise ValueError(f"No scores found on or before {target_date}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    dates = frame["date"].dt.date.tolist()
    dates.sort()
    return dates


def infer_market_from_symbol(symbol: str) -> str:
    """
    銘柄コードから市場を自動判定する。

    Parameters:
    -----------
    symbol : str
        銘柄コード

    Returns:
    --------
    str
        'jp' (日本市場) または 'us' (米国市場)

    Notes:
    ------
    - .T で終わる銘柄は東証（日本市場）として扱う
    - それ以外は米国市場として扱う

    Examples:
    ---------
    >>> infer_market_from_symbol('5032.T')
    'jp'
    >>> infer_market_from_symbol('AAPL')
    'us'
    """
    if symbol.endswith('.T'):
        return 'jp'
    return 'us'


def load_scores(
    engine: Engine,
    window_start: date,
    window_end: date,
) -> pd.DataFrame:
    query = text(
        """
        SELECT symbol, date, total_score
        FROM backtest_results.daily_scores_v2
        WHERE date BETWEEN :window_start AND :window_end
        """
    )
    with engine.connect() as conn:
        scores = pd.read_sql_query(
            query,
            conn,
            params={"window_start": window_start, "window_end": window_end},
        )
    if scores.empty:
        raise ValueError("No score rows returned for the requested window")
    scores["date"] = pd.to_datetime(scores["date"], errors="coerce")
    scores["date"] = scores["date"].dt.date
    scores = scores.dropna(subset=["total_score"]).reset_index(drop=True)
    return scores


def compute_percentile(ranks: pd.Series) -> pd.Series:
    size = len(ranks)
    if size <= 1:
        return pd.Series([1.0] * size, index=ranks.index)
    return 1.0 - (ranks - 1) / (size - 1)


def build_daily_snapshot(scores: pd.DataFrame, target_date: date) -> pd.DataFrame:
    daily = scores.loc[scores["date"] == target_date].copy()
    if daily.empty:
        raise ValueError(f"No daily scores available for {target_date}")
    daily["score"] = daily["total_score"]
    daily["rank"] = daily["score"].rank(method="dense", ascending=False).astype(int)
    daily["percentile"] = compute_percentile(daily["rank"])
    daily["observations"] = 1
    daily["universe_size"] = len(daily)
    return daily[["symbol", "score", "rank", "percentile", "observations", "universe_size"]]


def aggregate_weekly_score(values: pd.Series, method: str) -> float:
    if method == "mean":
        return float(values.mean())
    if method == "median":
        return float(values.median())
    if method == "sum":
        return float(values.sum())
    if method == "max":
        return float(values.max())
    if method == "ewma":
        span = max(2, min(len(values), 5))
        return float(values.ewm(span=span, adjust=False).mean().iloc[-1])
    raise ValueError(f"Unsupported weekly method: {method}")


def build_weekly_snapshot(
    scores: pd.DataFrame,
    trading_dates: Sequence[date],
    method: str,
    min_observations: int,
) -> pd.DataFrame:
    window_scores = scores.loc[scores["date"].isin(trading_dates)].copy()
    if window_scores.empty:
        return pd.DataFrame()

    records: List[dict] = []
    for symbol, group in window_scores.groupby("symbol"):
        ordered = group.sort_values("date")
        observations = len(ordered)
        if observations < min_observations:
            continue
        weekly_score = aggregate_weekly_score(ordered["total_score"], method)
        records.append(
            {
                "symbol": symbol,
                "score": weekly_score,
                "observations": observations,
            }
        )

    weekly = pd.DataFrame.from_records(records)
    if weekly.empty:
        return weekly

    weekly["rank"] = weekly["score"].rank(method="dense", ascending=False).astype(int)
    weekly["percentile"] = compute_percentile(weekly["rank"])
    weekly["universe_size"] = len(weekly)
    return weekly


def enrich_snapshot(
    frame: pd.DataFrame,
    target_date: date,
    scope: str,
    method: str,
    window_start: date,
    window_end: date,
    market: str,
    window_size: int,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    timestamp = datetime.utcnow()
    enriched = frame.copy()
    enriched["rank_date"] = target_date
    enriched["ranking_scope"] = scope
    enriched["ranking_method"] = method
    enriched["window_size"] = window_size
    enriched["window_start_date"] = window_start
    enriched["window_end_date"] = window_end

    # 市場ラベルは銘柄コードから自動判定する
    # .Tで終わる銘柄は日本市場、それ以外は米国市場
    # これにより、呼び出し元のmarketパラメータに関わらず正しい市場が設定される
    enriched["market"] = enriched["symbol"].apply(infer_market_from_symbol)

    enriched["created_at"] = timestamp
    enriched["updated_at"] = timestamp
    columns = [
        "rank_date",
        "symbol",
        "ranking_scope",
        "ranking_method",
        "window_size",
        "window_start_date",
        "window_end_date",
        "score",
        "rank",
        "percentile",
        "observations",
        "universe_size",
        "market",
        "created_at",
        "updated_at",
    ]
    return enriched[columns]


def purge_existing_rows(
    conn: Engine,
    rank_date: date,
    scope: str,
    method: str,
    market: str,
) -> None:
    delete_stmt = text(
        """
        DELETE FROM backtest_results.score_rankings_v2
        WHERE rank_date = :rank_date
          AND ranking_scope = :scope
          AND ranking_method = :method
          AND market = :market
        """
    )
    conn.execute(
        delete_stmt,
        {
            "rank_date": rank_date,
            "scope": scope,
            "method": method,
            "market": market,
        },
    )


def persist_snapshot(engine: Engine, payload: pd.DataFrame) -> None:
    if payload.empty:
        LOGGER.warning("No ranking rows to persist; skipping write.")
        return

    combos = payload[["rank_date", "ranking_scope", "ranking_method", "market"]].drop_duplicates()
    with engine.begin() as txn:
        for _, combo in combos.iterrows():
            purge_existing_rows(
                txn,
                rank_date=combo["rank_date"],
                scope=combo["ranking_scope"],
                method=combo["ranking_method"],
                market=combo["market"],
            )
        payload.to_sql(
            "score_rankings_v2",
            txn,
            schema="backtest_results",
            if_exists="append",
            index=False,
            method="multi",
        )


def ensure_methods(requested: Optional[Iterable[str]]) -> List[str]:
    if requested:
        seen = []
        for method in requested:
            if method not in SUPPORTED_WEEKLY_METHODS:
                raise ValueError(f"Unsupported weekly method: {method}")
            if method not in seen:
                seen.append(method)
        return seen
    return DEFAULT_WEEKLY_METHODS.copy()


def generate_score_rankings(
    target_date: date | datetime | str,
    weekly_methods: Optional[Iterable[str]] = None,
    weekly_window: int = 5,
    min_observations: int = 3,
    market: str = "global",
    dry_run: bool = False,
    engine: Optional[Engine] = None,
) -> pd.DataFrame:
    """Compute ranking snapshots and optionally persist them."""

    resolved_date = parse_target_date(target_date)
    methods = ensure_methods(weekly_methods)
    engine = engine or build_engine()

    trading_dates = get_recent_trading_dates(engine, resolved_date, weekly_window)
    scores = load_scores(engine, trading_dates[0], trading_dates[-1])

    frames: List[pd.DataFrame] = []
    daily_snapshot = build_daily_snapshot(scores, resolved_date)
    frames.append(
        enrich_snapshot(
            daily_snapshot,
            resolved_date,
            DAILY_SCOPE,
            "total_score",
            resolved_date,
            resolved_date,
            market,
            1,
        )
    )

    for method in methods:
        weekly_snapshot = build_weekly_snapshot(scores, trading_dates, method, min_observations)
        if weekly_snapshot.empty:
            LOGGER.warning("Weekly snapshot for %s is empty; skipping.", method)
            continue
        frames.append(
            enrich_snapshot(
                weekly_snapshot,
                resolved_date,
                WEEKLY_SCOPE,
                method,
                trading_dates[0],
                trading_dates[-1],
                market,
                len(trading_dates),
            )
        )

    if not frames:
        LOGGER.warning("No ranking data assembled for %s; nothing to do.", resolved_date)
        return pd.DataFrame()

    snapshot = pd.concat(frames, ignore_index=True)

    if dry_run:
        LOGGER.info("Dry-run enabled; returning snapshot without persistence")
        return snapshot

    persist_snapshot(engine, snapshot)
    LOGGER.info("Persisted %s ranking rows for %s.", len(snapshot), resolved_date)
    return snapshot


__all__ = [
    "DAILY_SCOPE",
    "WEEKLY_SCOPE",
    "DEFAULT_WEEKLY_METHODS",
    "SUPPORTED_WEEKLY_METHODS",
    "aggregate_weekly_score",
    "build_daily_snapshot",
    "build_engine",
    "build_weekly_snapshot",
    "compute_percentile",
    "ensure_methods",
    "enrich_snapshot",
    "generate_score_rankings",
    "get_recent_trading_dates",
    "infer_market_from_symbol",
    "load_scores",
    "persist_snapshot",
]
