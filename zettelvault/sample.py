"""Sample vault creation for pipeline preview.

Selects a representative subset of notes from a source vault using pure
Python heuristics (no LLM calls, no API key needed). The user can then
run the full pipeline on this small sample to preview results before
committing to a multi-day full-vault run.

All thresholds are loaded from config.yaml under the ``sample`` key.
"""

import json
import re
import statistics
import sys
from pathlib import Path

from .config import config_get
from .vault_io import list_vault_notes, read_note


# -- Feature extraction -------------------------------------------------------


def _count_headings(lines: list[str]) -> int:
    """Count lines that start with a markdown heading (# through ######)."""
    count = 0
    for line in lines:
        stripped = line.lstrip()
        if re.match(r"^#{1,6}\s", stripped):
            count += 1
    return count


def _count_bullets(lines: list[str]) -> int:
    """Count lines that start with a bullet marker (-, *, +) with optional indent."""
    count = 0
    for line in lines:
        stripped = line.lstrip()
        if re.match(r"^[-*+]\s", stripped):
            count += 1
    return count


def _count_wikilinks(content: str) -> int:
    """Count [[wikilink]] patterns in content."""
    return len(re.findall(r"\[\[[^\]]+\]\]", content))


def _count_codeblocks(content: str) -> int:
    """Count matched pairs of triple-backtick fences."""
    fences = re.findall(r"^```", content, re.MULTILINE)
    return len(fences) // 2


def _count_tags(content: str) -> int:
    """Count inline #tags that are not inside code blocks.

    Splits content on triple-backtick fences and only counts tags in
    non-code sections. Tags must match the pattern #word (letters,
    digits, hyphens, underscores).
    """
    parts = re.split(r"```[\s\S]*?```", content)
    total = 0
    for part in parts:
        # Match #word but not at the start of a heading line (## Heading)
        # and not preceded by & (HTML entities like &#123;)
        matches = re.findall(r"(?<!\w)#([A-Za-z][A-Za-z0-9_-]*)\b", part)
        # Exclude matches that look like heading markers
        for m in matches:
            total += 1
    return total


def extract_features(content: str) -> dict:
    """Compute structural features from a note's raw markdown content.

    Returns a dict with keys: char_count, has_frontmatter, heading_count,
    bullet_count, wikilink_count, codeblock_count, tag_count.
    """
    lines = content.splitlines()
    return {
        "char_count": len(content),
        "has_frontmatter": content.startswith("---\n"),
        "heading_count": _count_headings(lines),
        "bullet_count": _count_bullets(lines),
        "wikilink_count": _count_wikilinks(content),
        "codeblock_count": _count_codeblocks(content),
        "tag_count": _count_tags(content),
    }


# -- Structure classification -------------------------------------------------


def classify_structure(features: dict, total_lines: int, cfg: dict) -> str:
    """Classify a note's structure type based on line ratios and config thresholds.

    Returns one of: "bullet-heavy", "heading-heavy", "prose-heavy", "mixed".
    """
    if total_lines == 0:
        return "mixed"

    bullet_ratio = features["bullet_count"] / total_lines
    heading_ratio = features["heading_count"] / total_lines
    prose_lines = total_lines - features["bullet_count"] - features["heading_count"]
    prose_ratio = prose_lines / total_lines

    bullet_threshold = config_get(cfg, "sample.bullet_heavy_threshold", 0.40)
    heading_threshold = config_get(cfg, "sample.heading_heavy_threshold", 0.15)
    prose_threshold = config_get(cfg, "sample.prose_heavy_threshold", 0.70)

    if bullet_ratio > bullet_threshold:
        return "bullet-heavy"
    if heading_ratio > heading_threshold and bullet_ratio < 0.20:
        return "heading-heavy"
    if prose_ratio > prose_threshold:
        return "prose-heavy"
    return "mixed"


# -- Size bucketing ------------------------------------------------------------


def compute_size_buckets(char_counts: list[int]) -> dict[str, tuple[int, int]]:
    """Compute quartile boundaries for size bucketing.

    Returns a dict mapping bucket names (Q1-Q4) to (min, max) char_count
    ranges. With fewer than 4 notes, some buckets may overlap.
    """
    if not char_counts:
        return {}

    sorted_counts = sorted(char_counts)
    n = len(sorted_counts)

    if n < 4:
        # With fewer than 4 notes, assign each to its own bucket
        buckets = {}
        for i, c in enumerate(sorted_counts):
            buckets[f"Q{i + 1}"] = (c, c)
        return buckets

    q1 = sorted_counts[n // 4]
    q2 = sorted_counts[n // 2]
    q3 = sorted_counts[3 * n // 4]

    return {
        "Q1": (sorted_counts[0], q1),
        "Q2": (q1 + 1, q2),
        "Q3": (q2 + 1, q3),
        "Q4": (q3 + 1, sorted_counts[-1]),
    }


def assign_size_bucket(char_count: int, buckets: dict[str, tuple[int, int]]) -> str:
    """Assign a note to a size bucket based on its char_count."""
    for name, (lo, hi) in buckets.items():
        if lo <= char_count <= hi:
            return name
    # Edge case: if char_count falls outside all ranges, assign to nearest
    if buckets:
        names = list(buckets.keys())
        return names[-1]
    return "Q1"


# -- Greedy set-cover selection ------------------------------------------------


def _coverage_slots(features: dict, size_bucket: str, structure: str) -> set[str]:
    """Compute the set of coverage slots a note can fill."""
    slots = set()
    slots.add(f"size:{size_bucket}")
    slots.add(f"structure:{structure}")
    if features["has_frontmatter"]:
        slots.add("feature:has_frontmatter")
    if features["wikilink_count"] > 0:
        slots.add("feature:has_wikilinks")
    if features["codeblock_count"] > 0:
        slots.add("feature:has_codeblocks")
    if features["tag_count"] > 0:
        slots.add("feature:has_tags")
    return slots


def greedy_select(
    notes: list[dict],
    sample_size: int,
    median_char_count: float,
) -> list[dict]:
    """Select notes using greedy set-cover to maximize diversity.

    Each note has pre-computed 'slots' (a set of coverage categories).
    Selection proceeds by picking the note that covers the most uncovered
    slots. Ties are broken by preferring notes closest to median char_count,
    then alphabetically by title.

    After coverage saturates (no new slots available), remaining slots are
    filled by maximizing size diversity (pick notes furthest from already-
    selected char_counts).

    Args:
        notes: List of dicts with keys: title, content, features,
               size_bucket, structure, slots.
        sample_size: Target number of notes to select.
        median_char_count: Median char_count across all notes.

    Returns:
        List of selected note dicts, each with an added 'selection_reason' key.
    """
    if len(notes) <= sample_size:
        for note in notes:
            note["selection_reason"] = (
                "all notes selected (vault smaller than sample size)"
            )
        return list(notes)

    selected: list[dict] = []
    covered: set[str] = set()
    remaining = list(notes)

    # Phase 1: greedy set-cover
    while len(selected) < sample_size and remaining:
        # Score each candidate by number of new slots it fills
        best = None
        best_new_count = -1
        best_distance = float("inf")
        best_title = ""

        for note in remaining:
            new_slots = note["slots"] - covered
            new_count = len(new_slots)
            distance = abs(note["features"]["char_count"] - median_char_count)
            title = note["title"]

            if new_count > best_new_count:
                best = note
                best_new_count = new_count
                best_distance = distance
                best_title = title
            elif new_count == best_new_count:
                # Tie-break: prefer closer to median, then alphabetical
                if distance < best_distance or (
                    distance == best_distance and title < best_title
                ):
                    best = note
                    best_distance = distance
                    best_title = title

        if best is None:
            break

        if best_new_count > 0:
            reason_slots = sorted(best["slots"] - covered)
            best["selection_reason"] = f"covers: {', '.join(reason_slots)}"
        else:
            # Phase 2: maximize size diversity
            selected_sizes = [s["features"]["char_count"] for s in selected]
            best = None
            best_min_dist = -1
            best_title = ""

            for note in remaining:
                min_dist = min(
                    abs(note["features"]["char_count"] - sc) for sc in selected_sizes
                )
                if min_dist > best_min_dist or (
                    min_dist == best_min_dist and note["title"] < best_title
                ):
                    best = note
                    best_min_dist = min_dist
                    best_title = note["title"]

            if best is None:
                break
            best["selection_reason"] = "size diversity"

        covered.update(best["slots"])
        selected.append(best)
        remaining.remove(best)

    return selected


# -- Main entry point ----------------------------------------------------------


def sample_vault(
    source_vaults: list[str],
    cfg: dict,
    sample_size: int | None = None,
    output_dir: str | None = None,
) -> Path:
    """Select representative notes from source vaults and write them to a sample directory.

    Reads all notes, computes features, classifies structure, buckets by
    size, and runs greedy set-cover selection. Writes selected notes as
    plain .md files and a _sample_manifest.json with metadata.

    Args:
        source_vaults: List of vault names (as known to vlt).
        cfg: Loaded config dict.
        sample_size: Number of notes to select. Falls back to
                     config ``sample.size``, then 10.
        output_dir: Base output directory. Falls back to "./_sample".

    Returns:
        Path to the output directory containing sampled notes.
    """
    if sample_size is None:
        sample_size = config_get(cfg, "sample.size", 10)
    if output_dir is None:
        output_dir = "./_sample"

    output_base = Path(output_dir)

    # -- Read all notes from all source vaults ---------------------------------
    all_notes: list[dict] = []
    for vault_name in source_vaults:
        print(f"[sample] Reading '{vault_name}'...")
        titles = list_vault_notes(vault_name)
        print(f"         {len(titles)} notes found")

        for title in sorted(titles):  # sorted for determinism
            content = read_note(vault_name, title)
            if not content:
                continue
            all_notes.append(
                {
                    "title": title,
                    "vault": vault_name,
                    "content": content,
                }
            )

    if not all_notes:
        print("[sample] No notes found in source vaults.", file=sys.stderr)
        return output_base

    # Sort alphabetically for determinism
    all_notes.sort(key=lambda n: n["title"])

    # -- Extract features and classify -----------------------------------------
    for note in all_notes:
        note["features"] = extract_features(note["content"])
        total_lines = len(note["content"].splitlines())
        note["structure"] = classify_structure(note["features"], total_lines, cfg)

    # -- Size bucketing --------------------------------------------------------
    char_counts = [n["features"]["char_count"] for n in all_notes]
    buckets = compute_size_buckets(char_counts)
    median_cc = statistics.median(char_counts) if char_counts else 0

    for note in all_notes:
        note["size_bucket"] = assign_size_bucket(
            note["features"]["char_count"],
            buckets,
        )
        note["slots"] = _coverage_slots(
            note["features"],
            note["size_bucket"],
            note["structure"],
        )

    # -- Warn if fewer notes than sample_size ----------------------------------
    if len(all_notes) < sample_size:
        print(
            f"[sample] Warning: vault has {len(all_notes)} notes, "
            f"fewer than requested sample size {sample_size}. Selecting all.",
            file=sys.stderr,
        )

    # -- Select ----------------------------------------------------------------
    selected = greedy_select(all_notes, sample_size, median_cc)

    # -- Write output ----------------------------------------------------------
    # Use the first vault name as the directory name (spaces replaced)
    vault_dir_name = source_vaults[0].replace(" ", "_")
    dest = output_base / vault_dir_name
    dest.mkdir(parents=True, exist_ok=True)

    for note in selected:
        out_file = dest / f"{note['title']}.md"
        out_file.write_text(note["content"], encoding="utf-8")

    # -- Write manifest --------------------------------------------------------
    manifest = {
        "source_vaults": source_vaults,
        "total_notes": len(all_notes),
        "sample_size": len(selected),
        "size_buckets": {k: list(v) for k, v in buckets.items()},
        "median_char_count": median_cc,
        "notes": [],
    }
    for note in selected:
        manifest["notes"].append(
            {
                "title": note["title"],
                "vault": note["vault"],
                "features": note["features"],
                "structure": note["structure"],
                "size_bucket": note["size_bucket"],
                "selection_reason": note.get("selection_reason", ""),
            }
        )

    manifest_path = dest / "_sample_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # -- Print summary ---------------------------------------------------------
    print(f"\n[sample] Selected {len(selected)} / {len(all_notes)} notes:")
    structures = {}
    size_dist = {}
    for note in selected:
        structures[note["structure"]] = structures.get(note["structure"], 0) + 1
        size_dist[note["size_bucket"]] = size_dist.get(note["size_bucket"], 0) + 1

    print(f"         Structure types: {structures}")
    print(f"         Size buckets:    {size_dist}")
    print(f"         Output:          {dest}")
    print(f"         Manifest:        {manifest_path}")

    return dest
