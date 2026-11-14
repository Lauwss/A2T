"""Tests for the listening-test re-scorer (boundary rule and session scoring)."""

import importlib.util
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def load_rescore_module():
    spec = importlib.util.spec_from_file_location(
        "rescore_listening", SCRIPTS_DIR / "rescore_listening.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rescore():
    return load_rescore_module()


def test_monotone_responses_give_midpoint(rescore) -> None:
    # levels are model threshold + offset: -6, 0, +6, +12, +18 around -49.0
    levels = {"D": -55.0, "A": -49.0, "B": -43.0, "E": -37.0, "B2": -31.0}
    responses = {"D": "no", "A": "no", "B": "no", "E": "yes", "B2": "yes"}
    boundary = rescore.boundary_for(levels, responses)

    assert boundary["kind"] == "boundary"
    assert boundary["highest_unheard_dbfs"] == -43.0
    assert boundary["lowest_heard_dbfs"] == -37.0
    assert boundary["boundary_dbfs"] == pytest.approx(-40.0)


def test_stray_unsure_at_quiet_level_does_not_drag_boundary(rescore) -> None:
    # Round-2 500 Hz pattern: unsure at the quietest level, monotone above it.
    levels = {"A": -52.89, "D": -46.89, "C": -40.89, "E": -34.89, "B": -28.89}
    responses = {"A": "unsure", "D": "no", "C": "no", "E": "yes", "B": "yes"}
    boundary = rescore.boundary_for(levels, responses)

    assert boundary["kind"] == "boundary"
    assert boundary["boundary_dbfs"] == pytest.approx(-37.89)


def test_all_no_caps_above(rescore) -> None:
    levels = {"A": -50.0, "B": -44.0, "C": -38.0}
    boundary = rescore.boundary_for(levels, {k: "no" for k in levels})

    assert boundary["kind"] == "capped-above"
    assert boundary["boundary_dbfs"] == -38.0


def test_all_yes_caps_below(rescore) -> None:
    levels = {"A": -50.0, "B": -44.0, "C": -38.0}
    boundary = rescore.boundary_for(levels, {k: "yes" for k in levels})

    assert boundary["kind"] == "capped-below"
    assert boundary["boundary_dbfs"] == -50.0


def test_inverted_responses_are_flagged(rescore) -> None:
    # "yes" at the quietest level, "no" above it: false alarm or lapse.
    levels = {"D": -55.22, "C": -49.22, "B": -43.22, "A": -37.22, "E": -31.22}
    responses = {"D": "yes", "C": "no", "B": "no", "A": "yes", "E": "yes"}
    boundary = rescore.boundary_for(levels, responses)

    assert boundary["inverted"] is True
    assert boundary["boundary_dbfs"] == pytest.approx(-49.22)


def test_monotone_responses_have_no_inverted_flag(rescore) -> None:
    levels = {"A": -50.0, "B": -44.0, "C": -38.0}
    responses = {"A": "no", "B": "no", "C": "yes"}

    assert "inverted" not in rescore.boundary_for(levels, responses)


def test_score_session_offsets_against_model_threshold(rescore) -> None:
    results = {
        "trials": [
            {"hz": 1000.0, "label": "C", "response": "no"},
            {"hz": 1000.0, "label": "B", "response": "no"},
            {"hz": 1000.0, "label": "E", "response": "yes"},
            {"hz": 1000.0, "label": "A", "response": None},
        ]
    }
    answers = {
        "levels": {"1000": {"A": -37.22, "B": -43.22, "C": -49.22, "E": -31.22}},
        "model_threshold_dbfs": {"1000": -49.22},
    }
    scored = rescore.score_session(results, answers)

    boundary = scored["1000"]
    assert boundary["kind"] == "boundary"
    assert boundary["boundary_dbfs"] == pytest.approx(-37.22)
    assert boundary["offset_db"] == pytest.approx(12.0)
    assert boundary["responses"]["C"] == "no"


def test_score_session_ignores_unknown_labels(rescore) -> None:
    results = {"trials": [{"hz": 1000.0, "label": "X", "response": "yes"}]}
    answers = {"levels": {"1000": {"A": -49.22}}}

    assert rescore.score_session(results, answers) == {}
