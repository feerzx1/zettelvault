"""Zettelkasten decomposition of notes into atomic units.

Decomposes classified notes into atomic Zettelkasten notes using a
three-level fallback strategy: RLM (programmatic REPL), Predict
(direct LLM), and single-atom passthrough.
"""

import json
import re
import time

import dspy

from .config import (
    ATOMIC_CACHE,
    FALLBACK_LOG,
    config_get,
)
from .classify import build_concept_index, find_related
from .sanitize import (
    WL_CLOSE,
    WL_OPEN,
    SafeEncoder,
    extract_frontmatter,
    restore_wikilinks,
    sanitize_content,
)


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
    classification: dict,
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
            links_raw.replace("[[", "")
            .replace("]]", "")
            .replace(WL_OPEN, "")
            .replace(WL_CLOSE, "")
            .replace(".md", "")
        )
        links = [
            f"[[{link_text.strip()}]]"
            for link_text in links_raw.split(",")
            if link_text.strip() and link_text.strip() != "..."
        ]

        body = body_match.group(1).strip() if body_match else ""
        body = restore_wikilinks(body)
        if body == "..." or len(body) < 10:
            continue

        atoms.append(
            {
                "title": title,
                "content": body,
                "links": links,
                "tags": tags,
                "para_bucket": classification["para_bucket"],
                "domain": classification["domain"],
                "subdomain": classification["subdomain"],
                "source_note": source_title,
                "original_frontmatter": original_frontmatter or {},
            }
        )

    return atoms


def _fallback_atom(
    title: str,
    content: str,
    classification: dict,
    original_frontmatter: dict | None = None,
) -> dict:
    """Emit the original note as a single atom when decomposition fails."""
    return {
        "title": title,
        "content": content,
        "links": [],
        "tags": classification.get("tags", []),
        "para_bucket": classification["para_bucket"],
        "domain": classification["domain"],
        "subdomain": classification["subdomain"],
        "source_note": title,
        "original_frontmatter": original_frontmatter or {},
    }


def _decompose_with_rlm(
    title: str,
    content: str,
    related_str: str,
    *,
    decomposer_rlm,
    cfg: dict,
) -> tuple[str, list[dict]]:
    """Attempt decomposition via RLM. Returns (raw_output, trajectory)."""
    max_chars = config_get(cfg, "pipeline.max_input_chars", 8000)
    result = decomposer_rlm(
        note_title=title,
        note_content=content[:max_chars],
        related_note_titles=related_str,
    )
    raw = result.decomposed
    trajectory = getattr(result, "trajectory", [])
    return raw, trajectory


def _decompose_with_predict(
    title: str,
    content: str,
    related_str: str,
    *,
    decomposer_predict,
    cfg: dict,
) -> str:
    """Attempt decomposition via Predict with retry. Returns raw output."""
    max_retries = config_get(cfg, "pipeline.max_retries", 3)
    temp_start = config_get(cfg, "pipeline.retry_temp_start", 0.1)
    temp_step = config_get(cfg, "pipeline.retry_temp_step", 0.3)
    max_chars = config_get(cfg, "pipeline.max_input_chars", 8000)

    raw = ""
    for attempt in range(1, max_retries + 1):
        temp = temp_start + (attempt - 1) * temp_step
        with dspy.context(temperature=temp, cache=False):
            result = decomposer_predict(
                note_title=title,
                note_content=content[:max_chars],
                related_note_titles=related_str,
            )
        raw = result.decomposed
        if is_valid_output(raw):
            if attempt > 1:
                print(
                    f"              (Predict succeeded on attempt {attempt}, temp={temp})"
                )
            return raw
        print(
            f"              Predict attempt {attempt} (temp={temp}): invalid ({len(raw)} chars)"
        )
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
    title: str,
    classified_note: dict,
    related: list[str],
    *,
    use_rlm: bool,
    decomposer_rlm,
    decomposer_predict,
    cfg: dict,
) -> tuple[list[dict], int, int, str]:
    """Decompose one classified note into atomic notes.

    Returns (atoms, rlm_iterations, rlm_sub_calls, method).
    method is one of: "rlm", "predict", "passthrough".

    Three-level fallback:
      1. RLM (programmatic decomposition via REPL)
      2. Predict with retry (direct LLM call)
      3. Single-atom passthrough (guaranteed success)
    """
    classification = classified_note["classification"]
    original_fm = extract_frontmatter(classified_note["content"])
    content = sanitize_content(classified_note["content"])
    related_str = ", ".join(related)
    rlm_iters = 0
    rlm_subs = 0

    # Level 1: RLM
    if use_rlm and decomposer_rlm is not None:
        try:
            raw, trajectory = _decompose_with_rlm(
                title,
                content,
                related_str,
                decomposer_rlm=decomposer_rlm,
                cfg=cfg,
            )
            rlm_iters, rlm_subs = _summarize_trajectory(trajectory)

            if is_valid_output(raw):
                atoms = parse_atoms(raw, classification, title, original_fm)
                if atoms:
                    return atoms, rlm_iters, rlm_subs, "rlm"

            print(f"        [WARN] RLM output invalid for: {title[:50]}", flush=True)
        except Exception as exc:
            print(f"        [WARN] RLM failed for: {title[:50]} ({exc})", flush=True)

    # Level 2: Predict with retry
    raw = _decompose_with_predict(
        title,
        content,
        related_str,
        decomposer_predict=decomposer_predict,
        cfg=cfg,
    )
    if is_valid_output(raw):
        atoms = parse_atoms(raw, classification, title, original_fm)
        if atoms:
            return atoms, rlm_iters, rlm_subs, "predict"

    # Level 3: Single-atom passthrough
    return (
        [_fallback_atom(title, content, classification, original_fm)],
        rlm_iters,
        rlm_subs,
        "passthrough",
    )


def decompose_and_link(
    classified: dict[str, dict],
    *,
    use_rlm: bool,
    decomposer_rlm,
    decomposer_predict,
    cfg: dict,
    phase_usage=None,
    existing_atoms: list[dict] | None = None,
    progress_line_fn=None,
) -> list[dict]:
    """Build concept index, then decompose every note with cross-link awareness.

    If existing_atoms is provided, notes whose source_note matches an entry
    in existing_atoms are skipped (progressive processing).
    """
    concept_min_word_len = config_get(cfg, "pipeline.concept_min_word_len", 4)
    related_top_n = config_get(cfg, "pipeline.related_top_n", 20)

    print("      Building concept index...")
    index = build_concept_index(classified, min_word_len=concept_min_word_len)
    print(f"      {len(index)} concepts indexed across {len(classified)} notes")
    if use_rlm:
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
        print(
            f"      {len(already_done)} notes already decomposed, "
            f"{len(new_titles)} new"
        )

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
        related = find_related(title, data["content"], index, top=related_top_n)
        try:
            atoms, iters, subs, method = decompose_note(
                title,
                data,
                related,
                use_rlm=use_rlm,
                decomposer_rlm=decomposer_rlm,
                decomposer_predict=decomposer_predict,
                cfg=cfg,
            )
            total_rlm_iters += iters
            total_rlm_subs += subs

            tag = method.upper() if method in ("predict", "passthrough") else "RLM"
            if method in ("predict", "passthrough"):
                fallbacks.append(
                    {
                        "title": title,
                        "reason": method,
                        "atoms": len(atoms),
                    }
                )

            if progress_line_fn:
                print(
                    progress_line_fn(
                        dec_i,
                        dec_total,
                        dec_t0,
                        f"decompose/{tag}",
                        f"{title[:40]} -> {len(atoms)} atoms",
                    ),
                    flush=True,
                )

            all_atomic.extend(atoms)
        except Exception as exc:
            if progress_line_fn:
                print(
                    progress_line_fn(
                        dec_i,
                        dec_total,
                        dec_t0,
                        "FAILED",
                        f"{title[:40]}: {exc}",
                    ),
                    flush=True,
                )
            fallbacks.append(
                {
                    "title": title,
                    "reason": str(exc),
                    "atoms": 0,
                }
            )

        # Checkpoint after every note -- tokens are expensive
        ATOMIC_CACHE.write_text(json.dumps(all_atomic, indent=2, cls=SafeEncoder))
        if fallbacks:
            FALLBACK_LOG.write_text(json.dumps(fallbacks, indent=2, cls=SafeEncoder))

    # Save fallback log
    if fallbacks:
        FALLBACK_LOG.write_text(json.dumps(fallbacks, indent=2, cls=SafeEncoder))
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
