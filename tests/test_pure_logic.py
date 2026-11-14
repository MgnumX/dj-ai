import pytest

from analyzer import camelot
from quick_mixer import resumen_transiciones
from set_planner import compatibles, objetivo_nivel


@pytest.mark.parametrize(
    ("pitch_class", "minor", "expected"),
    [(0, False, "8B"), (9, True, "8A")],
)
def test_camelot_mapping(pitch_class, minor, expected):
    assert camelot(pitch_class, minor) == expected


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [("12A", "1B", True), ("8A", "9B", True), ("8A", "10B", False)],
)
def test_harmonic_compatibility(left, right, expected):
    assert compatibles(left, right) is expected


def test_energy_curve_handles_three_tracks():
    assert objetivo_nivel(0, 3, 0.6) == 3.0


def test_single_track_set_has_no_transitions():
    assert resumen_transiciones([], 4) == "1 segmento, sin transiciones."
