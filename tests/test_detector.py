import numpy as np
import pytest

from oddmeter.demo import make_demo
from oddmeter.detector import detect


def test_detection_is_reproducible_and_finds_extreme_points():
    rows = [
        {"x": i / 10, "y": i / 20} for i in range(60)
    ] + [{"x": 1000.0, "y": -900.0}, {"x": -800.0, "y": 1200.0}]
    first = detect(rows, ["x", "y"], contamination=0.05, seed=11)
    second = detect(rows, ["x", "y"], contamination=0.05, seed=11)
    assert np.array_equal(first.flags, second.flags)
    assert np.all(first.flags[-2:])
    assert np.all(first.scores[-2:] < first.scores[:60].min())
    assert len(first.main_signals) == len(rows)


def test_missing_numeric_values_are_imputed():
    rows = [{"x": i, "y": "" if i == 4 else i * 2} for i in range(30)]
    result = detect(rows, ["x", "y"], seed=2)
    assert len(result.flags) == len(rows)


def test_validation_errors_are_clear():
    with pytest.raises(ValueError, match="at least one"):
        detect([{"x": 1}, {"x": 2}, {"x": 3}], [])
    with pytest.raises(ValueError, match="Contamination"):
        detect([{"x": i} for i in range(4)], ["x"], contamination=0)
    with pytest.raises(ValueError, match="non-numeric"):
        detect([{"x": 1}, {"x": "bad"}, {"x": 3}], ["x"])
    with pytest.raises(ValueError, match="At least three"):
        detect([{"x": 1}, {"x": 2}], ["x"])


def test_demo_is_seeded_and_has_known_anomalies():
    a = make_demo(rows=100, seed=9)
    b = make_demo(rows=100, seed=9)
    assert a == b
    assert sum(row["known_event"] == "yes" for row in a) == 4
