import pandas as pd
from datetime import date, timedelta
import pytest

from investment_toolkit.scoring.ranking_snapshot import (
    aggregate_weekly_score,
    build_daily_snapshot,
    build_weekly_snapshot,
    ensure_methods,
    enrich_snapshot,
)


def test_build_daily_snapshot_ranking():
    target = date(2025, 1, 2)
    scores = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC"],
            "date": [target] * 3,
            "total_score": [70.0, 85.0, 60.0],
        }
    )

    snapshot = build_daily_snapshot(scores, target)

    assert list(snapshot["symbol"]) == ["BBB", "AAA", "CCC"]
    assert list(snapshot["rank"]) == [1, 2, 3]
    percentiles = snapshot.set_index("symbol")["percentile"].to_dict()
    assert percentiles["BBB"] == pytest.approx(1.0)
    assert percentiles["CCC"] == pytest.approx(0.0)


def test_build_weekly_snapshot_median():
    base_date = date(2025, 1, 1)
    trading_dates = [base_date + timedelta(days=i) for i in range(5)]
    rows = []
    for idx, trading_date in enumerate(trading_dates):
        rows.append({"symbol": "AAA", "date": trading_date, "total_score": 60 + idx})
        rows.append({"symbol": "BBB", "date": trading_date, "total_score": 80 - idx})
    scores = pd.DataFrame(rows)

    snapshot = build_weekly_snapshot(scores, trading_dates, "median", min_observations=5)
    assert set(snapshot["symbol"]) == {"AAA", "BBB"}
    median_scores = snapshot.set_index("symbol")["score"].to_dict()
    assert median_scores["AAA"] == pytest.approx(62.0)
    assert median_scores["BBB"] == pytest.approx(78.0)
    assert list(snapshot.sort_values("rank")["symbol"]) == ["BBB", "AAA"]


def test_aggregate_weekly_score_ewma():
    series = pd.Series([10.0, 12.0, 14.0, 18.0])
    result = aggregate_weekly_score(series, "ewma")
    assert result > series.iloc[-2]
    assert result < series.iloc[-1]


def test_ensure_methods_validation():
    methods = ensure_methods(["mean", "median", "mean"])
    assert methods == ["mean", "median"]
    with pytest.raises(ValueError):
        ensure_methods(["unknown"])


def test_enrich_snapshot_metadata():
    base_frame = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "score": [88.5],
            "rank": [1],
            "percentile": [1.0],
            "observations": [5],
            "universe_size": [100],
        }
    )
    target = date(2025, 1, 10)
    enriched = enrich_snapshot(
        base_frame,
        target_date=target,
        scope="weekly",
        method="median",
        window_start=target - timedelta(days=4),
        window_end=target,
        market="jp",
        window_size=5,
    )
    assert enriched.iloc[0]["ranking_scope"] == "weekly"
    assert enriched.iloc[0]["market"] == "jp"
    assert enriched.iloc[0]["window_size"] == 5
    assert enriched.iloc[0]["rank_date"] == target
    assert "created_at" in enriched.columns
