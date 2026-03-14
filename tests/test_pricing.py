"""
Comprehensive tests for the pricing module.

Run:
    uv run -p 3.13 -- pytest tests/test_pricing.py -v
"""

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


# ============================================================================
# ModelRate
# ============================================================================


class TestModelRate:
    def test_cost_calculation(self):
        rate = ModelRate(
            model_id="test/model",
            name="Test Model",
            prompt_per_token=0.00001,
            completion_per_token=0.00002,
        )
        cost = rate.cost(prompt_tokens=1000, completion_tokens=500)
        # 1000 * 0.00001 + 500 * 0.00002 = 0.01 + 0.01 = 0.02
        assert cost == pytest.approx(0.02)

    def test_zero_tokens(self):
        rate = ModelRate(
            model_id="test/model",
            name="Test",
            prompt_per_token=0.001,
            completion_per_token=0.002,
        )
        assert rate.cost(0, 0) == 0.0

    def test_zero_rates(self):
        rate = ModelRate(
            model_id="test/model",
            name="Test",
            prompt_per_token=0.0,
            completion_per_token=0.0,
        )
        assert rate.cost(1000, 500) == 0.0

    def test_default_context_length(self):
        rate = ModelRate(
            model_id="test/model",
            name="Test",
            prompt_per_token=0.0,
            completion_per_token=0.0,
        )
        assert rate.context_length == 0


# ============================================================================
# PhaseUsage
# ============================================================================


class TestPhaseUsage:
    def test_total_tokens(self):
        usage = PhaseUsage(name="test", prompt_tokens=100, completion_tokens=50)
        assert usage.total_tokens == 150

    def test_defaults(self):
        usage = PhaseUsage(name="test")
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.calls == 0
        assert usage.rlm_iterations == 0
        assert usage.rlm_sub_calls == 0
        assert usage.total_tokens == 0

    def test_rlm_fields(self):
        usage = PhaseUsage(name="test", rlm_iterations=5, rlm_sub_calls=10)
        assert usage.rlm_iterations == 5
        assert usage.rlm_sub_calls == 10


# ============================================================================
# _zero_rate
# ============================================================================


class TestZeroRate:
    def test_returns_zero_pricing(self, capsys):
        rate = _zero_rate("test/model", "test reason")
        assert rate.prompt_per_token == 0.0
        assert rate.completion_per_token == 0.0
        assert rate.model_id == "test/model"
        assert "test reason" in capsys.readouterr().out

    def test_empty_reason_no_output(self, capsys):
        rate = _zero_rate("test/model", "")
        assert rate.prompt_per_token == 0.0
        assert capsys.readouterr().out == ""

    def test_name_equals_model_id(self):
        rate = _zero_rate("test/model", "reason")
        assert rate.name == "test/model"


# ============================================================================
# _extract_history_usage
# ============================================================================


class TestExtractHistoryUsage:
    def test_no_lm(self):
        pt, ct, calls, cost = _extract_history_usage(None, 0)
        assert (pt, ct, calls, cost) == (0, 0, 0, 0.0)

    def test_lm_without_history_attr(self):
        lm = MagicMock(spec=[])  # No history attribute
        pt, ct, calls, cost = _extract_history_usage(lm, 0)
        assert (pt, ct, calls, cost) == (0, 0, 0, 0.0)

    def test_from_top_level_usage_dict(self):
        lm = MagicMock()
        lm.history = [
            {
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                },
                "cost": 0.001,
            },
            {
                "usage": {
                    "prompt_tokens": 200,
                    "completion_tokens": 80,
                    "total_tokens": 280,
                },
                "cost": 0.002,
            },
        ]
        pt, ct, calls, cost = _extract_history_usage(lm, 0)
        assert pt == 300
        assert ct == 130
        assert calls == 2
        assert cost == pytest.approx(0.003)

    def test_respects_start_index(self):
        lm = MagicMock()
        lm.history = [
            {"usage": {"prompt_tokens": 100, "completion_tokens": 50}, "cost": 0.001},
            {"usage": {"prompt_tokens": 200, "completion_tokens": 80}, "cost": 0.002},
        ]
        pt, ct, calls, _ = _extract_history_usage(lm, 1)
        assert pt == 200
        assert ct == 80
        assert calls == 1

    def test_handles_missing_usage(self):
        lm = MagicMock()
        lm.history = [{"messages": "something", "response": None}]
        pt, ct, calls, _ = _extract_history_usage(lm, 0)
        assert pt == 0
        assert ct == 0
        assert calls == 1

    def test_fallback_to_response_usage_attributes(self):
        """When top-level usage is missing, fall back to entry['response'].usage."""
        lm = MagicMock()
        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 150
        mock_response.usage.completion_tokens = 75
        lm.history = [
            {
                "usage": {},  # Empty top-level usage
                "response": mock_response,
                "cost": 0.005,
            },
        ]
        pt, ct, calls, cost = _extract_history_usage(lm, 0)
        assert pt == 150
        assert ct == 75
        assert calls == 1
        assert cost == pytest.approx(0.005)

    def test_fallback_when_top_level_usage_is_none(self):
        """When top-level usage key exists but is None, fall back to response.usage."""
        lm = MagicMock()
        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        lm.history = [
            {
                "usage": None,
                "response": mock_response,
            },
        ]
        pt, ct, calls, _ = _extract_history_usage(lm, 0)
        assert pt == 100
        assert ct == 50

    def test_provider_cost_accumulation(self):
        """entry['cost'] values are accumulated as litellm_cost."""
        lm = MagicMock()
        lm.history = [
            {"usage": {"prompt_tokens": 10, "completion_tokens": 5}, "cost": 0.01},
            {"usage": {"prompt_tokens": 20, "completion_tokens": 10}, "cost": 0.02},
            {"usage": {"prompt_tokens": 30, "completion_tokens": 15}, "cost": 0.03},
        ]
        _, _, _, cost = _extract_history_usage(lm, 0)
        assert cost == pytest.approx(0.06)

    def test_non_dict_entries_skipped(self):
        lm = MagicMock()
        lm.history = [
            "not a dict",
            42,
            {"usage": {"prompt_tokens": 100, "completion_tokens": 50}},
        ]
        pt, ct, calls, _ = _extract_history_usage(lm, 0)
        assert pt == 100
        assert ct == 50
        assert calls == 1  # Only the dict entry counted

    def test_zero_usage_triggers_fallback(self):
        """When top-level usage dict has zeros, fall back to response.usage."""
        lm = MagicMock()
        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 200
        mock_response.usage.completion_tokens = 100
        lm.history = [
            {
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                "response": mock_response,
            },
        ]
        pt, ct, calls, _ = _extract_history_usage(lm, 0)
        assert pt == 200
        assert ct == 100


# ============================================================================
# fetch_model_rate
# ============================================================================


class TestFetchModelRate:
    def test_no_api_key(self, capsys):
        with patch.dict("os.environ", {}, clear=True):
            rate = fetch_model_rate("test/model", api_key="")
        assert rate.prompt_per_token == 0.0
        assert "no API key" in capsys.readouterr().out

    def test_parses_api_response(self):
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

    def test_model_not_found(self, capsys):
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

    def test_network_error(self, capsys):
        with patch("pricing.urlopen", side_effect=ConnectionError("timeout")):
            rate = fetch_model_rate("test/model", api_key="test-key")
        assert rate.prompt_per_token == 0.0

    def test_uses_env_api_key(self):
        api_response = {"data": []}
        with patch("pricing.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(api_response).encode()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            with patch.dict("os.environ", {"OPENROUTER_API_KEY": "env-key"}):
                fetch_model_rate("test/model")

        # Should have made the API call (not returned early for no key)
        mock_urlopen.assert_called_once()


# ============================================================================
# CostTracker
# ============================================================================


class TestCostTracker:
    def test_total_cost(self):
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

    def test_total_tokens(self):
        with patch("pricing.fetch_model_rate") as mock_fetch:
            mock_fetch.return_value = _zero_rate("test", "")
            tracker = CostTracker("test")
            tracker.phases = [
                PhaseUsage(name="a", prompt_tokens=100, completion_tokens=50),
                PhaseUsage(name="b", prompt_tokens=200, completion_tokens=80),
            ]
            assert tracker.total_tokens == 430

    def test_report_output(self, capsys):
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
                PhaseUsage(
                    name="classification",
                    prompt_tokens=1000,
                    completion_tokens=200,
                    calls=4,
                ),
            ]
            tracker.report()

        output = capsys.readouterr().out
        assert "COST REPORT" in output
        assert "classification" in output
        assert "TOTAL" in output
        assert "Test Model" in output

    def test_report_with_rlm_info(self, capsys):
        with patch("pricing.fetch_model_rate") as mock_fetch:
            mock_fetch.return_value = ModelRate(
                model_id="test/model",
                name="Test",
                prompt_per_token=0.0,
                completion_per_token=0.0,
            )
            tracker = CostTracker("test/model")
            tracker.phases = [
                PhaseUsage(
                    name="decomposition",
                    prompt_tokens=5000,
                    completion_tokens=2000,
                    calls=10,
                    rlm_iterations=5,
                    rlm_sub_calls=15,
                ),
            ]
            tracker.report()

        output = capsys.readouterr().out
        assert "5 iters" in output
        assert "15 sub" in output

    def test_report_with_litellm_cost(self, capsys):
        with patch("pricing.fetch_model_rate") as mock_fetch:
            mock_fetch.return_value = ModelRate(
                model_id="test/model",
                name="Test",
                prompt_per_token=0.0,
                completion_per_token=0.0,
            )
            tracker = CostTracker("test/model")
            tracker._litellm_total = 0.05
            tracker.phases = [
                PhaseUsage(
                    name="test", prompt_tokens=100, completion_tokens=50, calls=1
                ),
            ]
            tracker.report()

        output = capsys.readouterr().out
        assert "TOTAL (provider)" in output

    def test_phase_context_manager(self):
        """The phase() context manager tracks LM calls during its scope."""
        import dspy

        with patch("pricing.fetch_model_rate") as mock_fetch:
            mock_fetch.return_value = ModelRate(
                model_id="test",
                name="Test",
                prompt_per_token=0.00001,
                completion_per_token=0.00002,
            )
            tracker = CostTracker("test")

        mock_lm = MagicMock()
        mock_lm.history = [
            # Pre-existing entry (before the phase)
            {"usage": {"prompt_tokens": 500, "completion_tokens": 250}, "cost": 0.01},
        ]

        original_lm = dspy.settings.lm
        try:
            dspy.settings.lm = mock_lm
            with tracker.phase("test_phase") as _usage:  # noqa: F841
                # Simulate LM calls happening during the phase
                mock_lm.history.append(
                    {
                        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                        "cost": 0.005,
                    }
                )
                mock_lm.history.append(
                    {
                        "usage": {"prompt_tokens": 200, "completion_tokens": 80},
                        "cost": 0.008,
                    }
                )
        finally:
            dspy.settings.lm = original_lm

        # Phase should have captured only the entries added during its scope
        assert len(tracker.phases) == 1
        phase = tracker.phases[0]
        assert phase.name == "test_phase"
        assert phase.prompt_tokens == 300
        assert phase.completion_tokens == 130
        assert phase.calls == 2

    def test_empty_phases(self):
        with patch("pricing.fetch_model_rate") as mock_fetch:
            mock_fetch.return_value = _zero_rate("test", "")
            tracker = CostTracker("test")
        assert tracker.total_cost == 0.0
        assert tracker.total_tokens == 0
