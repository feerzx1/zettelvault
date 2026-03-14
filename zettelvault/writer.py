"""Vault file writer for atomic notes and MOC (Map of Content) pages.

Writes atomic notes to the PARA folder structure and generates
domain-level MOC index pages with wikilinks.
"""

import re
from pathlib import Path

import yaml

from .config import BUCKET_FOLDERS


_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

# Fields we generate -- original frontmatter with these keys is overridden.
_GENERATED_FM_KEYS = {"tags", "domain", "subdomain", "source", "type"}


def _safe_filename(title: str) -> str:
    """Sanitize a note title for use as a filesystem filename."""
    cleaned = _UNSAFE.sub("-", title).strip(". -")
    return cleaned or "Untitled"


def _build_content(note: dict) -> str:
    """Build the full markdown content for a note file."""
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
                {key: value},
                default_flow_style=False,
                allow_unicode=True,
            )

    frontmatter += "---\n"

    body = f"# {note['title']}\n\n{note.get('content', '').strip()}"

    links = note.get("links", [])
    if links:
        body += "\n\n## Related\n\n" + "\n".join(f"- {lnk}" for lnk in links)

    return frontmatter + "\n" + body


def write_note(vault_path: Path, note: dict) -> Path:
    """Write an atomic note to the PARA folder structure, handling collisions."""
    bucket_dir = BUCKET_FOLDERS.get(note.get("para_bucket", ""), "3. Resources")
    raw_subdomain = note.get("subdomain", "").strip()
    subdomain = _safe_filename(raw_subdomain) if raw_subdomain else ""
    folder = (
        vault_path / bucket_dir / subdomain if subdomain else vault_path / bucket_dir
    )
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
    """Generate Map of Content pages, one per domain."""
    moc_dir = vault_path / "MOC"
    moc_dir.mkdir(exist_ok=True)

    by_domain: dict[str, list[str]] = {}
    for note in atomic_notes:
        by_domain.setdefault(note.get("domain", "Other"), []).append(
            note.get("title", "")
        )

    for domain, titles in sorted(by_domain.items()):
        domain_tag = _UNSAFE.sub("-", domain).lower().replace(" ", "-")
        links = "\n".join(f"- [[{t}]]" for t in sorted(set(titles)) if t)
        content = (
            f"---\ntags:\n  - moc\n  - {domain_tag}\ntype: moc\n---\n\n"
            f"# {domain}\n\n{links}\n"
        )
        (moc_dir / f"{_safe_filename(domain)}.md").write_text(content, encoding="utf-8")
