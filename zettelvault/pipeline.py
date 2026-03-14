"""Pipeline orchestration for ZettelVault.

The Pipeline class replaces all global mutable state. It holds config,
LM instances, and DSPy predictors as instance attributes, and provides
methods for initializing the LM and running the full pipeline.
"""

import os
import shutil
import subprocess
import time

import dspy

from .config import config_get


def _fmt_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def _progress_line(i: int, total: int, t0: float, label: str, detail: str = "") -> str:
    """Build a progress line with percentage, count, elapsed, and ETA."""
    elapsed = time.time() - t0
    pct = i / total * 100 if total else 0
    rate = elapsed / i if i else 0
    eta = rate * (total - i)
    parts = [
        f"  [{label}]",
        f"{i:>4}/{total}",
        f"({pct:5.1f}%)",
        f"elapsed {_fmt_duration(elapsed)}",
        f"eta {_fmt_duration(eta)}",
        f"({rate:.1f}s/note)",
    ]
    if detail:
        parts.append(f"-- {detail}")
    return " ".join(parts)


def _deno_version() -> str:
    try:
        r = subprocess.run(["deno", "--version"], capture_output=True, text=True)
        return r.stdout.split("\n")[0].replace("deno ", "").strip()
    except Exception:
        return "unknown"


def _make_lm(cfg: dict, section: str = "model") -> dspy.LM:
    """Build a dspy.LM from a config section ('model' or 'sub_model')."""
    model_id = config_get(cfg, f"{section}.id", "qwen/qwen3.5-35b-a3b")
    provider = config_get(cfg, f"{section}.provider", "openrouter")
    max_tokens = config_get(cfg, f"{section}.max_tokens", 32000)

    kwargs: dict = {"max_tokens": max_tokens}

    api_base = config_get(cfg, f"{section}.api_base")
    if api_base:
        kwargs["api_base"] = api_base

    api_key_env = config_get(cfg, f"{section}.api_key_env")
    if api_key_env:
        kwargs["api_key"] = os.environ.get(api_key_env, "")

    temperature = config_get(cfg, f"{section}.temperature")
    if temperature is not None:
        kwargs["temperature"] = temperature

    top_p = config_get(cfg, f"{section}.top_p")
    if top_p is not None:
        kwargs["top_p"] = top_p

    route = config_get(cfg, f"{section}.route")
    extra_body: dict = {}
    if route:
        extra_body["provider"] = route

    reasoning = config_get(cfg, f"{section}.reasoning")
    if reasoning:
        extra_body["reasoning"] = reasoning

    if extra_body:
        kwargs["extra_body"] = extra_body

    return dspy.LM(f"{provider}/{model_id}", **kwargs)


class Pipeline:
    """Holds all pipeline state: config, LM instances, and predictors.

    Replaces the module-level globals from the monolith. Instantiate once,
    call init_lm() to set up the language models, then call run() or use
    individual components.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.lm = None
        self.sub_lm = None
        self.classifier = None
        self.decomposer_rlm = None
        self.decomposer_predict = None
        self.use_rlm = False

    def init_lm(self, use_rlm: bool = True):
        """Initialize language models and DSPy predictors.

        Evaluates HAS_DENO lazily on first call rather than at import time.
        """
        from .classify import ClassifyNote
        from .decompose import DecomposeNote

        if self.lm is not None:
            return

        self.lm = _make_lm(self.cfg, "model")
        adapter_name = config_get(self.cfg, "model.adapter")
        adapter = None
        if adapter_name == "xml":
            adapter = dspy.XMLAdapter()
            print("      Adapter: XMLAdapter")
        elif adapter_name == "json":
            adapter = dspy.JSONAdapter()
            print("      Adapter: JSONAdapter")
        configure_kwargs: dict = {"lm": self.lm}
        if adapter:
            configure_kwargs["adapter"] = adapter
        dspy.configure(**configure_kwargs)

        # Sub-LM for RLM's llm_query() calls (can be smaller/cheaper)
        sub_cfg = config_get(self.cfg, "sub_model")
        if sub_cfg and sub_cfg.get("id") != config_get(self.cfg, "model.id"):
            self.sub_lm = _make_lm(self.cfg, "sub_model")
            print(f"      Sub-LM: {config_get(self.cfg, 'sub_model.id')}")

        self.classifier = dspy.Predict(ClassifyNote)
        self.decomposer_predict = dspy.Predict(DecomposeNote)

        # Lazy HAS_DENO check
        has_deno = shutil.which("deno") is not None

        if use_rlm and has_deno:
            try:
                rlm_kwargs: dict = {
                    "max_iterations": config_get(self.cfg, "rlm.max_iterations", 15),
                    "max_llm_calls": config_get(self.cfg, "rlm.max_llm_calls", 30),
                    "max_output_chars": config_get(
                        self.cfg, "rlm.max_output_chars", 15_000
                    ),
                    "verbose": config_get(self.cfg, "rlm.verbose", False),
                }
                if self.sub_lm is not None:
                    rlm_kwargs["sub_lm"] = self.sub_lm

                self.decomposer_rlm = dspy.RLM(DecomposeNote, **rlm_kwargs)
                self.use_rlm = True
                print(f"      RLM enabled (Deno {_deno_version()})")
            except Exception as exc:
                print(f"      RLM init failed ({exc}), using Predict")
                self.use_rlm = False
        elif use_rlm and not has_deno:
            print("      RLM requires Deno (https://deno.land). Using Predict.")
            self.use_rlm = False
        else:
            self.use_rlm = False
