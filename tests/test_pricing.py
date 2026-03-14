"""Tests for the pricing module."""

import json
from unittest.mock import MagicMock, patch

import pytest

from pricing import (
    CostTracker,
    ModelRate,
    PhaseUsage,
    _extract_history_usage,
    _zero_rate,
    fetch_model_rate,
)


# ── ModelRate ─────────────────────────────────────────────────────────────────


def test_model_rate_cost():
    rate = ModelRate(
        model_id="test/model",
        name="Test Model",
        prompt_per_token=0.00001,
        completion_per_token=0.00002,
    )
    cost = rate.cost(prompt_tokens=1000, completion_tokens=500)
    assert cost == pytest.approx(0.01 + 0.01)


def test_model_rate_zero_tokens():
    rate = ModelRate(
        model_id="test/model",
        name="Test",
        prompt_per_token=0.001,
        completion_per_token=0.002,
    )
    assert rate.cost(0, 0) == 0.0


# ── PhaseUsage ────────────────────────────────────────────────────────────────


def test_phase_usage_total_tokens():
    usage = PhaseUsage(name="test", prompt_tokens=100, completion_tokens=50)
    assert usage.total_tokens == 150


def test_phase_usage_defaults():
    usage = PhaseUsage(name="test")
    assert usage.prompt_tokens == 0
    assert usage.completion_tokens == 0
    assert usage.calls == 0
    assert usage.rlm_iterations == 0
    assert usage.total_tokens == 0


# ── _zero_rate ────────────────────────────────────────────────────────────────


def test_zero_rate_returns_zero_pricing(capsys):
    rate = _zero_rate("test/model", "test reason")
    assert rate.prompt_per_token == 0.0
    assert rate.completion_per_token == 0.0
    assert rate.model_id == "test/model"
    assert "test reason" in capsys.readouterr().out


# ── _extract_history_usage ────────────────────────────────────────────────────


def test_extract_history_no_lm():
    pt, ct, calls, cost = _extract_history_usage(None, 0)
    assert (pt, ct, calls, cost) == (0, 0, 0, 0.0)


def test_extract_history_from_entries():
    lm = MagicMock()
    lm.history = [
        {
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            "cost": 0.001,
        },
        {
            "usage": {"prompt_tokens": 200, "completion_tokens": 80, "total_tokens": 280},
            "cost": 0.002,
        },
    ]
    pt, ct, calls, cost = _extract_history_usage(lm, 0)
    assert pt == 300
    assert ct == 130
    assert calls == 2
    assert cost == pytest.approx(0.003)


def test_extract_history_respects_start_index():
    lm = MagicMock()
    lm.history = [
        {"usage": {"prompt_tokens": 100, "completion_tokens": 50}, "cost": 0.001},
        {"usage": {"prompt_tokens": 200, "completion_tokens": 80}, "cost": 0.002},
    ]
    pt, ct, calls, _ = _extract_history_usage(lm, 1)
    assert pt == 200
    assert ct == 80
    assert calls == 1


def test_extract_history_handles_missing_usage():
    lm = MagicMock()
    lm.history = [{"messages": "something", "response": None}]
    pt, ct, calls, _ = _extract_history_usage(lm, 0)
    assert pt == 0
    assert ct == 0
    assert calls == 1


# ── fetch_model_rate ──────────────────────────────────────────────────────────


def test_fetch_model_rate_no_api_key(capsys):
    with patch.dict("os.environ", {}, clear=True):
        rate = fetch_model_rate("test/model", api_key="")
    assert rate.prompt_per_token == 0.0
    assert "no API key" in capsys.readouterr().out


def test_fetch_model_rate_parses_response():
    api_response = {
        "data": [
            {
                "id": "qwen/qwen3.5-35b-a3b",
                "name": "Qwen3.5-35B-A3B",
                "context_length": 262144,
                "pricing": {"prompt": "0.00007", "completion": "0.00003"},
            }
        ]
    }
    with patch("pricing.urlopen") as mock_urlopen:
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(api_response).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        rate = fetch_model_rate("qwen/qwen3.5-35b-a3b", api_key="test-key")

    assert rate.name == "Qwen3.5-35B-A3B"
    assert rate.prompt_per_token == pytest.approx(0.00007)
    assert rate.completion_per_token == pytest.approx(0.00003)
    assert rate.context_length == 262144


def test_fetch_model_rate_model_not_found(capsys):
    api_response = {"data": [{"id": "other/model", "name": "Other"}]}
    with patch("pricing.urlopen") as mock_urlopen:
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(api_response).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        rate = fetch_model_rate("missing/model", api_key="test-key")

    assert rate.prompt_per_token == 0.0
    assert "not found" in capsys.readouterr().out


def test_fetch_model_rate_network_error(capsys):
    with patch("pricing.urlopen", side_effect=ConnectionError("timeout")):
        rate = fetch_model_rate("test/model", api_key="test-key")
    assert rate.prompt_per_token == 0.0


# ── CostTracker ──────────────────────────────────────────────────────────────


def test_cost_tracker_total_cost():
    with patch("pricing.fetch_model_rate") as mock_fetch:
        mock_fetch.return_value = ModelRate(
            model_id="test",
            name="Test",
            prompt_per_token=0.00001,
            completion_per_token=0.00002,
        )
        tracker = CostTracker("test")
        tracker.phases = [
            PhaseUsage(name="a", prompt_tokens=1000, completion_tokens=500),
            PhaseUsage(name="b", prompt_tokens=2000, completion_tokens=1000),
        ]
        # a: 1000*0.00001 + 500*0.00002 = 0.01 + 0.01 = 0.02
        # b: 2000*0.00001 + 1000*0.00002 = 0.02 + 0.02 = 0.04
        assert tracker.total_cost == pytest.approx(0.06)


def test_cost_tracker_total_tokens():
    with patch("pricing.fetch_model_rate") as mock_fetch:
        mock_fetch.return_value = _zero_rate("test", "")
        tracker = CostTracker("test")
        tracker.phases = [
            PhaseUsage(name="a", prompt_tokens=100, completion_tokens=50),
            PhaseUsage(name="b", prompt_tokens=200, completion_tokens=80),
        ]
        assert tracker.total_tokens == 430


def test_cost_tracker_report_runs(capsys):
    with patch("pricing.fetch_model_rate") as mock_fetch:
        mock_fetch.return_value = ModelRate(
            model_id="test/model",
            name="Test Model",
            prompt_per_token=0.00001,
            completion_per_token=0.00002,
            context_length=128000,
        )
        tracker = CostTracker("test/model")
        tracker.phases = [
            PhaseUsage(name="classification", prompt_tokens=1000, completion_tokens=200, calls=4),
        ]
        tracker.report()

    output = capsys.readouterr().out
    assert "COST REPORT" in output
    assert "classification" in output
    assert "TOTAL" in output
