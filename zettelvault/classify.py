"""PARA classification and concept indexing.

Classifies Obsidian notes into PARA buckets (Projects, Areas, Resources,
Archive) with domain/subdomain tags, and builds a word-level inverted
index for finding related notes.
"""

import re
from collections import Counter
from typing import Literal

import dspy

from .config import config_get
from .sanitize import sanitize_content


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

    para_bucket: Literal["Projects", "Areas", "Resources", "Archive"] = (
        dspy.OutputField()
    )
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


def classify_note(
    title: str,
    content: str,
    *,
    classifier,
    cfg: dict,
) -> dict:
    """Classify a single note into PARA structure with domain and tags."""
    max_chars = config_get(cfg, "pipeline.max_input_chars", 8000)
    result = classifier(title=title, content=sanitize_content(content)[:max_chars])
    return {
        "para_bucket": result.para_bucket,
        "domain": result.domain,
        "subdomain": result.subdomain,
        "tags": result.tags if isinstance(result.tags, list) else [],
    }


def build_concept_index(
    classified: dict[str, dict], *, min_word_len: int = 4
) -> dict[str, list[str]]:
    """Build an inverted index mapping words to note titles containing them.

    Used to find candidate cross-links before decomposition.
    """
    index: dict[str, list[str]] = {}
    for title, data in classified.items():
        text = title + " " + data.get("content", "")[:500]
        words = re.findall(r"\b[A-Za-z]\w{3,}\b", text)
        for word in set(w.lower() for w in words):
            index.setdefault(word, []).append(title)
    return index


def find_related(
    title: str,
    content: str,
    index: dict[str, list[str]],
    top: int = 8,
) -> list[str]:
    """Return the most conceptually similar note titles based on shared words."""
    text = title + " " + content[:1000]
    words = set(w.lower() for w in re.findall(r"\b[A-Za-z]\w{3,}\b", text))
    candidates: Counter = Counter()
    for word in words:
        for other in index.get(word, []):
            if other != title:
                candidates[other] += 1
    return [t for t, _ in candidates.most_common(top)]
