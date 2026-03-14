"""Content sanitization for safe LLM round-tripping.

Handles YAML frontmatter extraction/stripping and wikilink escaping
so that [[brackets]] don't collide with DSPy's template markers.
"""

import datetime
import json
import re

import yaml


# Wikilink escape tokens -- Unicode guillemets are single characters that can't
# appear in note titles and don't collide with DSPy's [[ ## field ## ]] markers.
WL_OPEN = "\u00ab"  # <<
WL_CLOSE = "\u00bb"  # >>


class SafeEncoder(json.JSONEncoder):
    """JSON encoder that handles date/datetime objects from YAML frontmatter."""

    def default(self, o):
        if isinstance(o, (datetime.date, datetime.datetime)):
            return o.isoformat()
        return super().default(o)


def sanitize_content(content: str) -> str:
    """Strip YAML frontmatter and escape [[wikilinks]] for DSPy processing.

    Wikilinks are converted to guillemet pairs which survive the LLM round-trip.
    Use restore_wikilinks() to convert back after DSPy processing.
    """
    content = re.sub(r"^---\n[\s\S]*?\n---\n*", "", content)
    content = re.sub(r"\[\[([^\]]+)\]\]", rf"{WL_OPEN}\1{WL_CLOSE}", content)
    return content.strip()


def restore_wikilinks(text: str) -> str:
    """Convert guillemet-escaped links back to [[wikilinks]]."""
    return re.sub(rf"{WL_OPEN}(.+?){WL_CLOSE}", r"[[\1]]", text)


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
