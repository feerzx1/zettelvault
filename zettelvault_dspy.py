#!/usr/bin/env python3
"""
ZettelVault -- Transform an Obsidian vault into PARA + Zettelkasten structure.

Pipeline:
  1. Read all notes from source vault via vlt
  2. Classify each note (PARA bucket + domain) via dspy.Predict
  3. Build concept index (Python), then decompose each note via dspy.RLM
     (falls back to dspy.Predict if RLM is unavailable)
  4. Write atomic notes to destination vault (filesystem)
  5. Resolve links -- fix orphan [[wikilinks]] via fuzzy match, stub creation, or removal

This serves as reference code for using dspy.RLM for document decomposition.
RLM enables the model to programmatically explore note content via a REPL,
write Python code to analyze structure, and use sub-LM calls for semantic tasks.

Usage:
  uv run --env-file .env -- python zettelvault_dspy.py Personal3 ./ZettelVault1
  uv run --env-file .env -- python zettelvault_dspy.py Personal3 ./ZettelVault1 --no-rlm
  uv run --env-file .env -- python zettelvault_dspy.py Personal3 ./ZettelVault1 --limit 4
  uv run --env-file .env -- python zettelvault_dspy.py Personal3 ./ZettelVault1 --skip-classification
"""

import argparse
import datetime
import difflib
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Literal


class _SafeEncoder(json.JSONEncoder):
    """Handle date/datetime objects from YAML frontmatter."""

    def default(self, o):
        if isinstance(o, (datetime.date, datetime.datetime)):
            return o.isoformat()
        return super().default(o)

import time

import yaml

from dotenv import load_dotenv

load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────────

import dspy  # noqa: E402

from pricing import CostTracker

_CONFIG_DEFAULT = Path("config.yaml")
_CONFIG_LOCAL = Path("config.local.yaml")


def load_config(path: Path | None = None) -> dict:
    """Load config from YAML. config.local.yaml overrides config.yaml."""
    cfg = {}
    for p in [_CONFIG_DEFAULT, _CONFIG_LOCAL]:
        if p.exists():
            with open(p) as f:
                override = yaml.safe_load(f)
                if isinstance(override, dict):
                    _deep_merge(cfg, override)
    if path and path.exists():
        with open(path) as f:
            override = yaml.safe_load(f)
            if isinstance(override, dict):
                _deep_merge(cfg, override)
    return cfg


def _deep_merge(base: dict, override: dict):
    """Merge override into base, recursing into nested dicts."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


_cfg: dict = {}


def _get(key: str, default=None):
    """Read a dotted config key like 'model.max_tokens'."""
    node = _cfg
    for part in key.split("."):
        if isinstance(node, dict):
            node = node.get(part)
        else:
            return default
        if node is None:
            return default
    return node


# ── Constants ────────────────────────────────────────────────────────────────

PARA_FOLDERS = ["1. Projects", "2. Areas", "3. Resources", "4. Archive", "MOC"]

BUCKET_FOLDERS = {
    "Projects": "1. Projects",
    "Areas": "2. Areas",
    "Resources": "3. Resources",
    "Archive": "4. Archive",
}

CLASSIFIED_CACHE = Path("classified_notes.json")
ATOMIC_CACHE = Path("atomic_notes.json")
FALLBACK_LOG = Path("fallback_notes.json")

_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

HAS_DENO = shutil.which("deno") is not None


# ── Lazy LM / predictors (avoid import-time API calls) ───────────────────────

_lm = None
_sub_lm = None
_classifier = None
_decomposer_rlm = None
_decomposer_predict = None
_use_rlm = False


def _make_lm(section: str = "model") -> dspy.LM:
    """Build a dspy.LM from a config section ('model' or 'sub_model')."""
    model_id = _get(f"{section}.id", "qwen/qwen3.5-35b-a3b")
    provider = _get(f"{section}.provider", "openrouter")
    max_tokens = _get(f"{section}.max_tokens", 32000)

    kwargs: dict = {"max_tokens": max_tokens}

    api_base = _get(f"{section}.api_base")
    if api_base:
        kwargs["api_base"] = api_base

    api_key_env = _get(f"{section}.api_key_env")
    if api_key_env:
        import os
        kwargs["api_key"] = os.environ.get(api_key_env, "")

    temperature = _get(f"{section}.temperature")
    if temperature is not None:
        kwargs["temperature"] = temperature

    top_p = _get(f"{section}.top_p")
    if top_p is not None:
        kwargs["top_p"] = top_p

    route = _get(f"{section}.route")
    extra_body: dict = {}
    if route:
        extra_body["provider"] = route

    reasoning = _get(f"{section}.reasoning")
    if reasoning:
        extra_body["reasoning"] = reasoning

    if extra_body:
        kwargs["extra_body"] = extra_body

    return dspy.LM(f"{provider}/{model_id}", **kwargs)


def _init_lm(use_rlm: bool = True):
    global _lm, _sub_lm, _classifier, _decomposer_rlm, _decomposer_predict, _use_rlm
    if _lm is not None:
        return

    _lm = _make_lm("model")
    adapter_name = _get("model.adapter")
    adapter = None
    if adapter_name == "xml":
        adapter = dspy.XMLAdapter()
        print(f"      Adapter: XMLAdapter")
    elif adapter_name == "json":
        adapter = dspy.JSONAdapter()
        print(f"      Adapter: JSONAdapter")
    configure_kwargs: dict = {"lm": _lm}
    if adapter:
        configure_kwargs["adapter"] = adapter
    dspy.configure(**configure_kwargs)

    # Sub-LM for RLM's llm_query() calls (can be smaller/cheaper)
    sub_cfg = _get("sub_model")
    if sub_cfg and sub_cfg.get("id") != _get("model.id"):
        _sub_lm = _make_lm("sub_model")
        print(f"      Sub-LM: {_get('sub_model.id')}")

    _classifier = dspy.Predict(ClassifyNote)
    _decomposer_predict = dspy.Predict(DecomposeNote)

    if use_rlm and HAS_DENO:
        try:
            rlm_kwargs: dict = {
                "max_iterations": _get("rlm.max_iterations", 15),
                "max_llm_calls": _get("rlm.max_llm_calls", 30),
                "max_output_chars": _get("rlm.max_output_chars", 15_000),
                "verbose": _get("rlm.verbose", False),
            }
            if _sub_lm is not None:
                rlm_kwargs["sub_lm"] = _sub_lm

            _decomposer_rlm = dspy.RLM(DecomposeNote, **rlm_kwargs)
            _use_rlm = True
            print(f"      RLM enabled (Deno {_deno_version()})")
        except Exception as exc:
            print(f"      RLM init failed ({exc}), using Predict")
            _use_rlm = False
    elif use_rlm and not HAS_DENO:
        print("      RLM requires Deno (https://deno.land). Using Predict.")
        _use_rlm = False
    else:
        _use_rlm = False


def _deno_version() -> str:
    try:
        r = subprocess.run(["deno", "--version"], capture_output=True, text=True)
        return r.stdout.split("\n")[0].replace("deno ", "").strip()
    except Exception:
        return "unknown"


# ── Progress logging ──────────────────────────────────────────────────────────


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


# ── vlt I/O ───────────────────────────────────────────────────────────────────


def vlt_run(vault: str, *args: str) -> str:
    cmd = ["vlt", f"vault={vault}", *args]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"vlt [{' '.join(args[:2])}]: {r.stderr.strip()}")
    return r.stdout.strip()


def resolve_vault_path(vault_name: str) -> Path | None:
    """Resolve a vault name to its filesystem path via vlt."""
    try:
        raw = subprocess.run(
            ["vlt", "vaults", "--json"], capture_output=True, text=True,
        ).stdout
        for v in json.loads(raw):
            if v.get("name") == vault_name:
                return Path(v["path"])
    except Exception:
        pass
    return None


def copy_obsidian_config(source_vault: str, dest_path: Path):
    """Copy .obsidian directory from source vault to destination."""
    src_path = resolve_vault_path(source_vault)
    if not src_path:
        print(f"      Could not resolve path for vault '{source_vault}', skipping .obsidian copy")
        return

    obsidian_dir = src_path / ".obsidian"
    if not obsidian_dir.is_dir():
        print(f"      No .obsidian directory in source vault")
        return

    dest_obsidian = dest_path / ".obsidian"
    if dest_obsidian.exists():
        print(f"      .obsidian already exists in destination, skipping copy")
        return

    shutil.copytree(obsidian_dir, dest_obsidian)
    print(f"      Copied .obsidian ({sum(1 for _ in dest_obsidian.rglob('*'))} files)")


def list_vault_notes(vault: str) -> list[str]:
    raw = vlt_run(vault, "files", "--json")
    try:
        entries = json.loads(raw)
    except json.JSONDecodeError:
        entries = [line.strip() for line in raw.splitlines() if line.strip()]

    titles = []
    for e in entries:
        path = e if isinstance(e, str) else e.get("path", "")
        if path.endswith(".md"):
            titles.append(Path(path).stem)
    return titles


def read_note(vault: str, title: str) -> str:
    try:
        return vlt_run(vault, "read", f"file={title}")
    except RuntimeError:
        return ""


# Wikilink escape tokens -- Unicode guillemets are single characters that can't
# appear in note titles and don't collide with DSPy's [[ ## field ## ]] markers.
_WL_OPEN = "\u00ab"   # <<
_WL_CLOSE = "\u00bb"  # >>


def sanitize_content(content: str) -> str:
    """Strip YAML frontmatter and escape [[wikilinks]] to avoid DSPy template collisions.

    Wikilinks are converted to <<guillemet>> pairs which survive the LLM round-trip.
    Use restore_wikilinks() to convert back to [[brackets]] after DSPy processing.
    """
    content = re.sub(r"^---\n[\s\S]*?\n---\n*", "", content)
    content = re.sub(r"\[\[([^\]]+)\]\]", rf"{_WL_OPEN}\1{_WL_CLOSE}", content)
    return content.strip()


def restore_wikilinks(text: str) -> str:
    """Convert <<escaped links>> back to [[wikilinks]]."""
    return re.sub(rf"{_WL_OPEN}(.+?){_WL_CLOSE}", r"[[\1]]", text)


def extract_frontmatter(content: str) -> dict:
    """Extract YAML frontmatter from note content as a dict.

    Returns empty dict if no frontmatter found or parsing fails.
    """
    match = re.match(r"^---\n([\s\S]*?)\n---", content)
    if not match:
        return {}
    try:
        fm = yaml.safe_load(match.group(1))
        return fm if isinstance(fm, dict) else {}
    except Exception:
        return {}


# ── Phase 1: PARA Classification ─────────────────────────────────────────────


class ClassifyNote(dspy.Signature):
    """
    Classify an Obsidian note into PARA structure.

    Projects:  active, time-bounded work (software project, research experiment, startup)
    Areas:     ongoing responsibilities with no end date (health, investing, skills, home)
    Resources: reference material, tools, papers, recipes, how-tos, clippings, quotes
    Archive:   completed, inactive, or superseded content
    """

    title: str = dspy.InputField()
    content: str = dspy.InputField()

    para_bucket: Literal["Projects", "Areas", "Resources", "Archive"] = dspy.OutputField()
    domain: str = dspy.OutputField(
        desc=(
            "Primary domain. Choose one: AI/ML | Cybersecurity | Investing | "
            "Engineering | Personal | Health | Travel | Business | Recipes | Other"
        )
    )
    subdomain: str = dspy.OutputField(
        desc="Specific area within domain. E.g. 'DSPy', 'Backtesting', 'Portland'"
    )
    tags: list[str] = dspy.OutputField(desc="3-7 lowercase hyphenated tags")


def classify_note(title: str, content: str) -> dict:
    _init_lm()
    max_chars = _get("pipeline.max_input_chars", 8000)
    result = _classifier(title=title, content=sanitize_content(content)[:max_chars])
    return {
        "para_bucket": result.para_bucket,
        "domain": result.domain,
        "subdomain": result.subdomain,
        "tags": result.tags if isinstance(result.tags, list) else [],
    }


# ── Phase 2: Concept Index ────────────────────────────────────────────────────


def build_concept_index(classified: dict[str, dict]) -> dict[str, list[str]]:
    """
    Map meaningful words → [note titles containing them].
    Used to find candidate cross-links before decomposition.
    """
    index: dict[str, list[str]] = {}
    for title, data in classified.items():
        # Draw words from title + first 500 chars of content
        text = title + " " + data.get("content", "")[:500]
        words = re.findall(r"\b[A-Za-z]\w{3,}\b", text)
        for word in set(w.lower() for w in words):
            index.setdefault(word, []).append(title)
    return index


def find_related(title: str, content: str, index: dict[str, list[str]], top: int = 8) -> list[str]:
    """Return the most conceptually similar note titles to this one."""
    text = title + " " + content[:1000]
    words = set(w.lower() for w in re.findall(r"\b[A-Za-z]\w{3,}\b", text))
    candidates: Counter = Counter()
    for word in words:
        for other in index.get(word, []):
            if other != title:
                candidates[other] += 1
    return [t for t, _ in candidates.most_common(top)]


# ── Phase 3: Zettelkasten Decomposition ───────────────────────────────────────


class DecomposeNote(dspy.Signature):
    """Decompose one Obsidian note into several atomic notes.

    Each atomic note covers ONE idea. Split aggressively.
    Preserve all original content. Never invent facts.

    Output format -- separate each atomic note with a line containing only ===

    Title: Specific Descriptive Title
    Tags: tag1, tag2, tag3
    Links: Related Note Title 1, Related Note Title 2
    Body:
    The actual content of this atomic note in markdown.

    ===

    Title: Another Atomic Note Title
    Tags: tag1, tag2
    Links: Some Other Note
    Body:
    More content here.
    """

    note_title: str = dspy.InputField()
    note_content: str = dspy.InputField()
    related_note_titles: str = dspy.InputField(
        desc="Comma-separated titles of related notes -- reference these in Links"
    )

    decomposed: str = dspy.OutputField(
        desc="Atomic notes separated by === lines, each with Title/Tags/Links/Body fields"
    )


def is_valid_output(raw: str) -> bool:
    """Check if decomposition output is usable (not template garbage)."""
    if len(raw) < 100:
        return False
    if "## ]]" in raw or "{decomposed}" in raw:
        return False
    real_title = re.search(r"^Title:\s*[^.\s].{5,}", raw, re.MULTILINE)
    if not real_title:
        return False
    if "Body:" not in raw:
        return False
    return True


def parse_atoms(
    raw: str,
    cls: dict,
    source_title: str,
    original_frontmatter: dict | None = None,
) -> list[dict]:
    """Parse markdown-delimited atomic notes from model output."""
    sections = re.split(r"\n===\s*\n|^===\s*$", raw, flags=re.MULTILINE)
    atoms = []

    for section in sections:
        section = section.strip()
        if not section:
            continue

        title_match = re.search(r"^Title:\s*(.+)", section, re.MULTILINE)
        tags_match = re.search(r"^Tags:\s*(.+)", section, re.MULTILINE)
        links_match = re.search(r"^Links:\s*(.+)", section, re.MULTILINE)
        body_match = re.search(r"^Body:\s*\n?([\s\S]*)", section, re.MULTILINE)

        if not title_match:
            continue

        title = title_match.group(1).strip()
        if title == "..." or len(title) < 3:
            continue

        tags_raw = tags_match.group(1) if tags_match else ""
        tags = []
        for t in tags_raw.split(","):
            t = t.strip().lstrip("#").strip().lower().replace(" ", "-")
            # Split concatenated hashtag tags like "#a-#b-#c"
            for part in t.split("-#"):
                part = part.strip("-").strip()
                if part and part != "...":
                    tags.append(part)

        links_raw = links_match.group(1) if links_match else ""
        links_raw = (
            links_raw.replace("[[", "").replace("]]", "")
            .replace(_WL_OPEN, "").replace(_WL_CLOSE, "")
            .replace(".md", "")
        )
        links = [
            f"[[{l.strip()}]]"
            for l in links_raw.split(",")
            if l.strip() and l.strip() != "..."
        ]

        body = body_match.group(1).strip() if body_match else ""
        body = restore_wikilinks(body)
        if body == "..." or len(body) < 10:
            continue

        atoms.append({
            "title": title,
            "content": body,
            "links": links,
            "tags": tags,
            "para_bucket": cls["para_bucket"],
            "domain": cls["domain"],
            "subdomain": cls["subdomain"],
            "source_note": source_title,
            "original_frontmatter": original_frontmatter or {},
        })

    return atoms


def _fallback_atom(
    title: str, content: str, cls: dict, original_frontmatter: dict | None = None,
) -> dict:
    """Emit the original note as a single atom when decomposition fails."""
    return {
        "title": title,
        "content": content,
        "links": [],
        "tags": cls.get("tags", []),
        "para_bucket": cls["para_bucket"],
        "domain": cls["domain"],
        "subdomain": cls["subdomain"],
        "source_note": title,
        "original_frontmatter": original_frontmatter or {},
    }


def _decompose_with_rlm(
    title: str, content: str, related_str: str
) -> tuple[str, list[dict]]:
    """Attempt decomposition via RLM. Returns (raw_output, trajectory)."""
    max_chars = _get("pipeline.max_input_chars", 8000)
    result = _decomposer_rlm(
        note_title=title,
        note_content=content[:max_chars],
        related_note_titles=related_str,
    )
    raw = result.decomposed
    trajectory = getattr(result, "trajectory", [])
    return raw, trajectory


def _decompose_with_predict(
    title: str, content: str, related_str: str
) -> str:
    """Attempt decomposition via Predict with retry. Returns raw output."""
    max_retries = _get("pipeline.max_retries", 3)
    temp_start = _get("pipeline.retry_temp_start", 0.1)
    temp_step = _get("pipeline.retry_temp_step", 0.3)
    max_chars = _get("pipeline.max_input_chars", 8000)

    raw = ""
    for attempt in range(1, max_retries + 1):
        temp = temp_start + (attempt - 1) * temp_step
        with dspy.context(temperature=temp, cache=False):
            result = _decomposer_predict(
                note_title=title,
                note_content=content[:max_chars],
                related_note_titles=related_str,
            )
        raw = result.decomposed
        if is_valid_output(raw):
            if attempt > 1:
                print(f"              (Predict succeeded on attempt {attempt}, temp={temp})")
            return raw
        print(f"              Predict attempt {attempt} (temp={temp}): invalid ({len(raw)} chars)")
    return raw


def _summarize_trajectory(trajectory: list[dict]) -> tuple[int, int]:
    """Extract iteration count and sub-LM call count from RLM trajectory."""
    iterations = len(trajectory)
    sub_calls = 0
    for step in trajectory:
        code = step.get("code", "")
        sub_calls += code.count("llm_query(")
        sub_calls += code.count("llm_query_batched(")
    return iterations, sub_calls


def decompose_note(
    title: str, data: dict, related: list[str],
) -> tuple[list[dict], int, int, str]:
    """Decompose one classified note.

    Returns (atoms, rlm_iterations, rlm_sub_calls, method).
    method is one of: "rlm", "predict", "passthrough".

    Three-level fallback:
      1. RLM (programmatic decomposition via REPL)
      2. Predict with retry (direct LLM call)
      3. Single-atom passthrough (guaranteed success)
    """
    cls = data["classification"]
    original_fm = extract_frontmatter(data["content"])
    content = sanitize_content(data["content"])
    related_str = ", ".join(related)
    rlm_iters = 0
    rlm_subs = 0

    # Level 1: RLM
    if _use_rlm and _decomposer_rlm is not None:
        try:
            raw, trajectory = _decompose_with_rlm(title, content, related_str)
            rlm_iters, rlm_subs = _summarize_trajectory(trajectory)

            if is_valid_output(raw):
                atoms = parse_atoms(raw, cls, title, original_fm)
                if atoms:
                    return atoms, rlm_iters, rlm_subs, "rlm"

            print(f"        [WARN] RLM output invalid for: {title[:50]}", flush=True)
        except Exception as exc:
            print(f"        [WARN] RLM failed for: {title[:50]} ({exc})", flush=True)

    # Level 2: Predict with retry
    raw = _decompose_with_predict(title, content, related_str)
    if is_valid_output(raw):
        atoms = parse_atoms(raw, cls, title, original_fm)
        if atoms:
            return atoms, rlm_iters, rlm_subs, "predict"

    # Level 3: Single-atom passthrough
    return [_fallback_atom(title, content, cls, original_fm)], rlm_iters, rlm_subs, "passthrough"


def decompose_and_link(
    classified: dict[str, dict],
    phase_usage=None,
    existing_atoms: list[dict] | None = None,
) -> list[dict]:
    """Build concept index, then decompose every note with cross-link awareness.

    If existing_atoms is provided, notes whose source_note matches an entry
    in existing_atoms are skipped (progressive processing).
    """
    print("      Building concept index...")
    index = build_concept_index(classified)
    print(f"      {len(index)} concepts indexed across {len(classified)} notes")
    if _use_rlm:
        print("      Strategy: dspy.RLM (programmatic REPL decomposition)")
    else:
        print("      Strategy: dspy.Predict (direct LLM call)")

    # Progressive: carry forward already-decomposed atoms and skip those titles
    all_atomic: list[dict] = []
    already_done: set[str] = set()
    if existing_atoms:
        all_atomic.extend(existing_atoms)
        already_done = {a.get("source_note", "") for a in existing_atoms}

    titles = list(classified.keys())
    new_titles = [t for t in titles if t not in already_done]

    if already_done:
        print(f"      {len(already_done)} notes already decomposed, "
              f"{len(new_titles)} new")

    total_rlm_iters = 0
    total_rlm_subs = 0
    fallbacks: list[dict] = []

    # Load existing fallback log to accumulate
    if FALLBACK_LOG.exists():
        fallbacks = json.loads(FALLBACK_LOG.read_text())

    dec_total = len(new_titles)
    dec_t0 = time.time()

    for dec_i, title in enumerate(new_titles, 1):
        data = classified[title]
        related = find_related(title, data["content"], index)
        try:
            atoms, iters, subs, method = decompose_note(title, data, related)
            total_rlm_iters += iters
            total_rlm_subs += subs

            tag = method.upper() if method in ("predict", "passthrough") else "RLM"
            if method in ("predict", "passthrough"):
                fallbacks.append({
                    "title": title,
                    "reason": method,
                    "atoms": len(atoms),
                })

            print(_progress_line(
                dec_i, dec_total, dec_t0, f"decompose/{tag}",
                f"{title[:40]} -> {len(atoms)} atoms",
            ), flush=True)

            all_atomic.extend(atoms)
        except Exception as exc:
            print(_progress_line(
                dec_i, dec_total, dec_t0, "FAILED",
                f"{title[:40]}: {exc}",
            ), flush=True)
            fallbacks.append({
                "title": title,
                "reason": str(exc),
                "atoms": 0,
            })

        # Checkpoint after every note -- tokens are expensive
        ATOMIC_CACHE.write_text(json.dumps(all_atomic, indent=2, cls=_SafeEncoder))
        if fallbacks:
            FALLBACK_LOG.write_text(json.dumps(fallbacks, indent=2, cls=_SafeEncoder))

    # Save fallback log
    if fallbacks:
        FALLBACK_LOG.write_text(json.dumps(fallbacks, indent=2, cls=_SafeEncoder))
        print(f"      {len(fallbacks)} notes used fallback (see {FALLBACK_LOG})")

    if total_rlm_iters > 0:
        avg = total_rlm_iters / len(new_titles) if new_titles else 0
        print(
            f"      RLM totals: {total_rlm_iters} iterations "
            f"({avg:.1f} avg), {total_rlm_subs} sub-LM calls"
        )

    # Update phase usage for cost report
    if phase_usage is not None:
        phase_usage.rlm_iterations = total_rlm_iters
        phase_usage.rlm_sub_calls = total_rlm_subs

    return all_atomic


# ── Note builder ──────────────────────────────────────────────────────────────


def _safe_filename(title: str) -> str:
    cleaned = _UNSAFE.sub("-", title).strip(". -")
    return cleaned or "Untitled"


# Fields we generate -- original frontmatter with these keys is overridden.
_GENERATED_FM_KEYS = {"tags", "domain", "subdomain", "source", "type"}


def _build_content(note: dict) -> str:
    tags = note.get("tags", [])
    tags_yaml = "\n".join(f"  - {t}" for t in tags) if tags else "  []"

    frontmatter = (
        f"---\n"
        f"tags:\n{tags_yaml}\n"
        f"domain: {note.get('domain', '')}\n"
        f"subdomain: {note.get('subdomain', '')}\n"
        f"source: {note.get('source_note', '')}\n"
        f"type: zettel\n"
    )

    # Preserve non-conflicting original frontmatter fields (aliases, cssclass, etc.)
    original_fm = note.get("original_frontmatter", {})
    for key, value in original_fm.items():
        if key not in _GENERATED_FM_KEYS:
            frontmatter += yaml.dump(
                {key: value}, default_flow_style=False, allow_unicode=True,
            )

    frontmatter += "---\n"

    body = f"# {note['title']}\n\n{note.get('content', '').strip()}"

    links = note.get("links", [])
    if links:
        body += "\n\n## Related\n\n" + "\n".join(f"- {lnk}" for lnk in links)

    return frontmatter + "\n" + body


# ── Vault writer ──────────────────────────────────────────────────────────────


def write_note(vault_path: Path, note: dict) -> Path:
    bucket_dir = BUCKET_FOLDERS.get(note.get("para_bucket", ""), "3. Resources")
    raw_subdomain = note.get("subdomain", "").strip()
    subdomain = _safe_filename(raw_subdomain) if raw_subdomain else ""
    folder = vault_path / bucket_dir / subdomain if subdomain else vault_path / bucket_dir
    folder.mkdir(parents=True, exist_ok=True)

    stem = _safe_filename(note.get("title", "Untitled"))
    path = folder / f"{stem}.md"

    counter = 1
    while path.exists():
        path = folder / f"{stem}_{counter}.md"
        counter += 1

    path.write_text(_build_content(note), encoding="utf-8")
    return path


def write_moc(vault_path: Path, atomic_notes: list[dict]):
    moc_dir = vault_path / "MOC"
    moc_dir.mkdir(exist_ok=True)

    by_domain: dict[str, list[str]] = {}
    for n in atomic_notes:
        by_domain.setdefault(n.get("domain", "Other"), []).append(n.get("title", ""))

    for domain, titles in sorted(by_domain.items()):
        domain_tag = _UNSAFE.sub("-", domain).lower().replace(" ", "-")
        links = "\n".join(f"- [[{t}]]" for t in sorted(set(titles)) if t)
        content = (
            f"---\ntags:\n  - moc\n  - {domain_tag}\ntype: moc\n---\n\n"
            f"# {domain}\n\n{links}\n"
        )
        (moc_dir / f"{_safe_filename(domain)}.md").write_text(content, encoding="utf-8")


# ── Step 5: Resolve links ─────────────────────────────────────────────────────

_WIKILINK_RE = re.compile(r'\[\[([^\]]+)\]\]')


def resolve_links(dest: Path):
    """Scan the destination vault and resolve orphan [[wikilinks]].

    1. Collect all note titles (from filenames) and all wikilinks per file.
    2. Identify orphan links (links with no matching note).
    3. Try to resolve each orphan via case-insensitive or fuzzy match.
    4. For remaining orphans: create stub notes (3+ refs) or remove dead links (1-2 refs).
    """
    # ── Collect titles and links ──────────────────────────────────────────────
    title_set: dict[str, Path] = {}          # title -> file path
    title_lower: dict[str, str] = {}         # lowercase title -> original title
    links_by_file: dict[Path, list[str]] = {}  # file -> list of wikilink targets

    for md_file in dest.rglob("*.md"):
        if ".obsidian" in md_file.parts:
            continue
        title = md_file.stem
        title_set[title] = md_file
        title_lower[title.lower()] = title

        text = md_file.read_text(encoding="utf-8")
        found = _WIKILINK_RE.findall(text)
        if found:
            links_by_file[md_file] = found

    all_titles = set(title_set.keys())

    # ── Find orphan links ─────────────────────────────────────────────────────
    # orphan_target -> list of source file paths
    orphan_sources: dict[str, list[Path]] = {}
    for src_file, targets in links_by_file.items():
        for target in targets:
            if target not in all_titles:
                orphan_sources.setdefault(target, []).append(src_file)

    if not orphan_sources:
        print("      No orphan links found.")
        return

    print(f"      {len(orphan_sources)} unique orphan link targets across vault")

    # ── Resolve: case-insensitive + fuzzy ─────────────────────────────────────
    resolved: dict[str, str] = {}       # orphan_target -> actual title
    unresolved: dict[str, list[Path]] = {}

    all_titles_list = list(all_titles)  # for fuzzy matching

    for orphan, sources in orphan_sources.items():
        # (a) Case-insensitive exact match
        ci_match = title_lower.get(orphan.lower())
        if ci_match and ci_match != orphan:
            resolved[orphan] = ci_match
            continue

        # (b) Fuzzy match via SequenceMatcher
        best_ratio = 0.0
        best_match = None
        orphan_lower = orphan.lower()
        for candidate in all_titles_list:
            ratio = difflib.SequenceMatcher(
                None, orphan_lower, candidate.lower(),
            ).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = candidate
        if best_ratio >= 0.85 and best_match and best_match != orphan:
            resolved[orphan] = best_match
            continue

        # No match found
        unresolved[orphan] = sources

    # ── Rewrite resolved links in source files ────────────────────────────────
    # Group rewrites per file to avoid reading/writing the same file many times.
    rewrites_per_file: dict[Path, dict[str, str]] = {}
    for orphan, actual in resolved.items():
        for src_file in orphan_sources[orphan]:
            rewrites_per_file.setdefault(src_file, {})[orphan] = actual

    for filepath, replacements in rewrites_per_file.items():
        text = filepath.read_text(encoding="utf-8")
        for old_target, new_target in replacements.items():
            text = text.replace(f"[[{old_target}]]", f"[[{new_target}]]")
        filepath.write_text(text, encoding="utf-8")

    # ── Handle unresolved links ───────────────────────────────────────────────
    stubs_created = 0
    dead_removed = 0

    for orphan, sources in unresolved.items():
        if len(sources) >= 3:
            # Create a stub note in the most common PARA folder among referencing notes
            para_counts: Counter = Counter()
            for src_file in sources:
                for para_folder in BUCKET_FOLDERS.values():
                    if para_folder in str(src_file):
                        para_counts[para_folder] += 1
                        break
            if para_counts:
                best_folder = para_counts.most_common(1)[0][0]
            else:
                best_folder = "3. Resources"

            ref_links = "\n".join(
                f"- [[{sf.stem}]]" for sf in sorted(sources, key=lambda p: p.stem)
            )
            stub_content = (
                f"---\ntags: []\ntype: stub\n---\n\n"
                f"# {orphan}\n\n"
                f"This concept is referenced by multiple notes "
                f"but has no dedicated content yet.\n\n"
                f"## Referenced by\n\n{ref_links}\n"
            )
            stub_dir = dest / best_folder
            stub_dir.mkdir(parents=True, exist_ok=True)
            stub_path = stub_dir / f"{_safe_filename(orphan)}.md"
            if not stub_path.exists():
                stub_path.write_text(stub_content, encoding="utf-8")
                stubs_created += 1
        else:
            # Remove dead links from files with 1-2 references
            for src_file in sources:
                text = src_file.read_text(encoding="utf-8")
                text = text.replace(f"[[{orphan}]]", orphan)
                src_file.write_text(text, encoding="utf-8")
            dead_removed += len(sources)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(
        f"      {len(resolved)} links resolved by fuzzy match, "
        f"{stubs_created} stub notes created, "
        f"{dead_removed} dead links removed"
    )


# ── Orchestration ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Transform an Obsidian vault into PARA + Zettelkasten structure"
    )
    parser.add_argument("source_vault", nargs="+", help="Source vault name(s) (as known to vlt)")
    parser.add_argument("dest_vault", help="Destination vault path (absolute or ~/...)")
    parser.add_argument("--dry-run", action="store_true", help="No file writes; preview only")
    parser.add_argument(
        "--no-rlm",
        action="store_true",
        help="Disable RLM; use dspy.Predict for decomposition",
    )
    parser.add_argument(
        "--skip-classification",
        action="store_true",
        help=f"Load pre-classified notes from {CLASSIFIED_CACHE}",
    )
    parser.add_argument(
        "--skip-decomposition",
        action="store_true",
        help=f"Load atomic notes from {ATOMIC_CACHE} (implies --skip-classification)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        metavar="N",
        help="Process only the first N notes (0 = all)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="FILE",
        help="Path to config YAML (default: config.yaml + config.local.yaml)",
    )
    args = parser.parse_args()

    dest = Path(args.dest_vault).expanduser().resolve()

    # ── Initialize ───────────────────────────────────────────────────────────
    global _cfg
    _cfg = load_config(args.config)

    model_id = _get("model.id", "qwen/qwen3.5-35b-a3b")
    tracker = CostTracker(model_id)
    _init_lm(use_rlm=not args.no_rlm)

    # ── Step 1: Read ──────────────────────────────────────────────────────────
    # Map each title to its source vault so read_note() uses the right one.
    title_vault: dict[str, str] = {}
    for vault_name in args.source_vault:
        print(f"[1/5] Reading '{vault_name}'...")
        for t in list_vault_notes(vault_name):
            if t not in title_vault:          # first vault wins on collision
                title_vault[t] = vault_name
    titles = list(title_vault.keys())
    if args.limit:
        titles = titles[: args.limit]
        title_vault = {t: title_vault[t] for t in titles}
        print(f"      {len(titles)} notes (limited to {args.limit})")
    else:
        for v in args.source_vault:
            count = sum(1 for tv in title_vault.values() if tv == v)
            print(f"      {v}: {count} notes")
        print(f"      {len(titles)} total notes")

    # ── Step 2: Classify ──────────────────────────────────────────────────────
    skip_cls = args.skip_classification or args.skip_decomposition

    if skip_cls and CLASSIFIED_CACHE.exists():
        print(f"[2/5] Loading pre-classified notes from {CLASSIFIED_CACHE}")
        classified = json.loads(CLASSIFIED_CACHE.read_text())
        if args.limit:
            classified = dict(list(classified.items())[: args.limit])
    else:
        # Progressive: load existing cache and only classify new notes
        classified = {}
        if CLASSIFIED_CACHE.exists():
            classified = json.loads(CLASSIFIED_CACHE.read_text())

        new_titles = [t for t in titles if t not in classified]
        cached_titles = [t for t in titles if t in classified]

        if cached_titles:
            print(f"[2/5] Classifying notes (PARA + domain)... "
                  f"({len(cached_titles)} cached, {len(new_titles)} new)")
        else:
            print("[2/5] Classifying notes (PARA + domain)...")

        if new_titles:
            cls_total = len(new_titles)
            cls_t0 = time.time()
            # Print every 10 notes or every 10%, whichever comes first
            cls_milestone = max(1, cls_total // 10)
            cls_interval = min(10, cls_milestone)

            with tracker.phase("classification"):
                for cls_i, title in enumerate(new_titles, 1):
                    content = read_note(title_vault.get(title, args.source_vault[0]), title)
                    if not content:
                        continue
                    cls = classify_note(title, content)
                    classified[title] = {"content": content, "classification": cls}
                    bucket = cls['para_bucket'] or '?'
                    domain = cls['domain'] or '?'

                    if cls_i % cls_interval == 0 or cls_i == cls_total:
                        print(_progress_line(
                            cls_i, cls_total, cls_t0, "classify",
                            f"[{bucket}] [{domain}] {title[:35]}",
                        ), flush=True)

                    # Save cache every 50 notes for crash resilience
                    if cls_i % 50 == 0:
                        CLASSIFIED_CACHE.write_text(json.dumps(classified, indent=2, cls=_SafeEncoder))

            CLASSIFIED_CACHE.write_text(json.dumps(classified, indent=2, cls=_SafeEncoder))
            print(f"      Saved to {CLASSIFIED_CACHE}", flush=True)
        else:
            print("      All notes already classified")

        # Filter classified to only the titles we're processing this run
        classified = {t: classified[t] for t in titles if t in classified}

    # ── Step 3: Decompose + link ───────────────────────────────────────────────
    if args.skip_decomposition and ATOMIC_CACHE.exists():
        print(f"[3/5] Loading atomic notes from {ATOMIC_CACHE}")
        atomic = json.loads(ATOMIC_CACHE.read_text())
    else:
        # Progressive: load existing atoms so already-decomposed notes are skipped
        existing_atoms = None
        if ATOMIC_CACHE.exists():
            existing_atoms = json.loads(ATOMIC_CACHE.read_text())

        print("[3/5] Decomposing and cross-linking...")
        with tracker.phase("decomposition") as phase:
            atomic = decompose_and_link(
                classified, phase_usage=phase, existing_atoms=existing_atoms,
            )

        if not atomic:
            print("ERROR: Decomposition returned no notes.", file=sys.stderr)
            sys.exit(1)

        new_count = len(atomic) - (len(existing_atoms) if existing_atoms else 0)
        print(f"      {len(atomic)} total atomic notes ({new_count} new)")
        ATOMIC_CACHE.write_text(json.dumps(atomic, indent=2, cls=_SafeEncoder))
        print(f"      Saved to {ATOMIC_CACHE}")

    # ── Step 4: Write ─────────────────────────────────────────────────────────
    if args.dry_run:
        print("[DRY RUN] Sample output (first 10 notes):")
        for n in atomic[:10]:
            print(
                f"  [{n.get('para_bucket', '?'):8}] "
                f"[{n.get('domain', '?'):15}] {n.get('title', '?')}"
            )
        domains = {n.get("domain") for n in atomic}
        print(f"  ... {len(atomic)} total notes across {len(domains)} domains")
        tracker.report()
        return

    print(f"[4/5] Writing to {dest}...")
    dest.mkdir(parents=True, exist_ok=True)
    for folder in PARA_FOLDERS:
        (dest / folder).mkdir(exist_ok=True)

    copy_obsidian_config(args.source_vault[0], dest)

    for note in atomic:
        write_note(dest, note)

    write_moc(dest, atomic)

    # ── Step 5: Resolve links ────────────────────────────────────────────────
    print(f"[5/5] Resolving orphan links...")
    resolve_links(dest)

    domains = {n.get("domain") for n in atomic}
    print(f"Done. {len(atomic)} notes + {len(domains)} MOC pages written to {dest}")
    tracker.report()


if __name__ == "__main__":
    main()
