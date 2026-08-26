"""Unit tests for Phase 2 data modules (schema map, validation, cleaning)."""

import numpy as np
import pandas as pd
import pytest

from src.data import validate
from src.data.clean import CleanLog, drop_exact_duplicates, null_impossible_values


@pytest.fixture
def sample_gen():
    return pd.DataFrame(
        {
            "site_id": [1, 1, 1, 2],
            "timestamp": pd.to_datetime(
                [
                    "2020-01-01 00:15",
                    "2020-01-01 00:30",
                    "2020-01-01 00:45",
                    "2020-01-01 00:15",
                ]
            ),
            "power": [0.0, -5.0, np.nan, 12.0],
        }
    )


def test_check_duplicate_keys(sample_gen):
    result = validate.check_duplicate_keys(sample_gen, "site_id")
    # No duplicate (site_id, timestamp) pairs in fixture.
    assert result["duplicate_key_rows"] == 0

    duplicated = pd.concat([sample_gen.iloc[[0]], sample_gen], ignore_index=True)
    result2 = validate.check_duplicate_keys(duplicated, "site_id")
    assert result2["duplicate_key_rows"] == 2
    assert result2["affected_groups"] == 1


def test_check_gaps_detects_missing_slot():
    df = pd.DataFrame(
        {
            "site_id": [1, 1, 1],
            "timestamp": pd.to_datetime(
                ["2020-01-01 00:00", "2020-01-01 00:15", "2020-01-01 01:00"]
            ),
        }
    )
    result = validate.check_gaps(df, "site_id")
    assert result["total_missing_slots"] == 2  # 00:30 and 00:45 missing
    assert result["per_group"][0]["expected_slots"] == 5


def test_impossible_value_rules():
    df = pd.DataFrame(
        {
            "power": [-1.0, 5.0, np.nan],
            "humidity": [150.0, 50.0, np.nan],
            "wind_speed": [-3.0, 10.0, np.nan],
            "wind_direction": [400.0, 180.0, np.nan],
            "temperature": [70.0, 20.0, np.nan],
        }
    )
    findings = validate.check_impossible_values(df)
    assert set(findings) == {
        "negative_power",
        "humidity_out_of_range",
        "negative_wind_speed",
        "wind_direction_out_of_range",
        "temperature_implausible",
    }


def test_null_impossible_values_keeps_rows_and_logs():
    log = CleanLog()
    df = pd.DataFrame({"power": [-1.0, 5.0]})
    rules = {"power": validate.IMPOSSIBLE_RULES["power"]}
    out = null_impossible_values(df, rules, log, "generation")
    assert len(out) == 2  # no row deletion
    assert out["power"].iloc[0] != out["power"].iloc[0]  # NaN after nulling
    assert log.operations[0]["operation"] == "null_impossible:negative_power"


def test_drop_exact_duplicates_logs(sample_gen):
    log = CleanLog()
    df = pd.concat([sample_gen.iloc[[0]], sample_gen], ignore_index=True)
    out = drop_exact_duplicates(df, log, "generation")
    assert len(out) == len(sample_gen)
    assert any(op["operation"] == "drop_exact_duplicate_rows" for op in log.operations)


def test_schema_map_column_contracts():
    from src.data import schema_map as sm

    assert set(sm.GENERATION_COLUMNS.values()) == {
        "site_id",
        "campus_id",
        "timestamp",
        "power",
    }
    assert set(sm.WEATHER_COLUMNS.values()) == {
        "campus_id",
        "timestamp",
        "temperature",
        "apparent_temperature",
        "dew_point_temperature",
        "humidity",
        "wind_speed",
        "wind_direction",
    }
