"""Orphan wikilink resolution for the destination vault.

Scans all notes in the destination vault, identifies orphan [[wikilinks]]
that don't match any existing note, and resolves them via case-insensitive
matching, fuzzy matching, stub creation, or dead link removal.
"""

import difflib
import re
from collections import Counter
from pathlib import Path

from .config import BUCKET_FOLDERS, config_get
from .writer import _safe_filename


_WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")


def resolve_links(dest: Path, *, cfg: dict | None = None):
    """Scan the destination vault and resolve orphan [[wikilinks]].

    1. Collect all note titles (from filenames) and all wikilinks per file.
    2. Identify orphan links (links with no matching note).
    3. Try to resolve each orphan via case-insensitive or fuzzy match.
    4. For remaining orphans: create stub notes (3+ refs) or remove dead links (1-2 refs).
    """
    fuzzy_threshold = config_get(cfg or {}, "resolve.fuzzy_threshold", 0.85)
    stub_min_refs = config_get(cfg or {}, "resolve.stub_min_refs", 3)

    # -- Collect titles and links -------------------------------------------------
    title_set: dict[str, Path] = {}  # title -> file path
    title_lower: dict[str, str] = {}  # lowercase title -> original title
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

    # -- Find orphan links --------------------------------------------------------
    orphan_sources: dict[str, list[Path]] = {}
    for src_file, targets in links_by_file.items():
        for target in targets:
            if target not in all_titles:
                orphan_sources.setdefault(target, []).append(src_file)

    if not orphan_sources:
        print("      No orphan links found.")
        return

    print(f"      {len(orphan_sources)} unique orphan link targets across vault")

    # -- Resolve: case-insensitive + fuzzy ----------------------------------------
    resolved: dict[str, str] = {}  # orphan_target -> actual title
    unresolved: dict[str, list[Path]] = {}

    all_titles_list = list(all_titles)

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
                None,
                orphan_lower,
                candidate.lower(),
            ).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = candidate
        if best_ratio >= fuzzy_threshold and best_match and best_match != orphan:
            resolved[orphan] = best_match
            continue

        # No match found
        unresolved[orphan] = sources

    # -- Rewrite resolved links in source files -----------------------------------
    rewrites_per_file: dict[Path, dict[str, str]] = {}
    for orphan, actual in resolved.items():
        for src_file in orphan_sources[orphan]:
            rewrites_per_file.setdefault(src_file, {})[orphan] = actual

    for filepath, replacements in rewrites_per_file.items():
        text = filepath.read_text(encoding="utf-8")
        for old_target, new_target in replacements.items():
            text = text.replace(f"[[{old_target}]]", f"[[{new_target}]]")
        filepath.write_text(text, encoding="utf-8")

    # -- Handle unresolved links --------------------------------------------------
    stubs_created = 0
    dead_removed = 0

    for orphan, sources in unresolved.items():
        if len(sources) >= stub_min_refs:
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
            # Remove dead links from files with fewer than stub_min_refs references
            for src_file in sources:
                text = src_file.read_text(encoding="utf-8")
                text = text.replace(f"[[{orphan}]]", orphan)
                src_file.write_text(text, encoding="utf-8")
            dead_removed += len(sources)

    # -- Summary ------------------------------------------------------------------
    print(
        f"      {len(resolved)} links resolved by fuzzy match, "
        f"{stubs_created} stub notes created, "
        f"{dead_removed} dead links removed"
    )
