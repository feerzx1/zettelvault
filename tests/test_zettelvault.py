"""
Unit and integration tests for zettelvault_dspy.py

Run unit tests (no API key):
    uv run --env-file .env -- pytest tests/ -v -m "not integration"

Run all tests (requires OPENROUTER_API_KEY):
    uv run --env-file .env -- pytest tests/ -v
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from zettelvault_dspy import (
    _build_content,
    _deep_merge,
    _get,
    _safe_filename,
    _WL_CLOSE,
    _WL_OPEN,
    extract_frontmatter,
    is_valid_output,
    list_vault_notes,
    load_config,
    parse_atoms,
    read_note,
    restore_wikilinks,
    sanitize_content,
    write_moc,
    write_note,
)

# ── config ────────────────────────────────────────────────────────────────────


def test_deep_merge_flat():
    base = {"a": 1, "b": 2}
    _deep_merge(base, {"b": 3, "c": 4})
    assert base == {"a": 1, "b": 3, "c": 4}


def test_deep_merge_nested():
    base = {"model": {"id": "a", "max_tokens": 1000}}
    _deep_merge(base, {"model": {"id": "b"}})
    assert base["model"]["id"] == "b"
    assert base["model"]["max_tokens"] == 1000  # preserved


def test_load_config_from_file(tmp_path):
    cfg_file = tmp_path / "test.yaml"
    cfg_file.write_text("model:\n  id: test/model\n  max_tokens: 4000\n")
    cfg = load_config(cfg_file)
    assert cfg["model"]["id"] == "test/model"
    assert cfg["model"]["max_tokens"] == 4000


def test_get_dotted_key():
    import zettelvault_dspy
    original = zettelvault_dspy._cfg
    try:
        zettelvault_dspy._cfg = {"model": {"id": "test", "route": {"only": ["X"]}}}
        assert _get("model.id") == "test"
        assert _get("model.route.only") == ["X"]
        assert _get("model.missing", 42) == 42
        assert _get("nonexistent.deep.key", "default") == "default"
    finally:
        zettelvault_dspy._cfg = original


# ── _safe_filename ────────────────────────────────────────────────────────────


def test_safe_filename_clean():
    assert _safe_filename("Normal Title") == "Normal Title"


def test_safe_filename_strips_colon():
    assert _safe_filename("Title: Subtitle") == "Title- Subtitle"


def test_safe_filename_strips_slash():
    assert _safe_filename("AI/ML Overview") == "AI-ML Overview"


def test_safe_filename_strips_backslash():
    assert _safe_filename("Path\\Note") == "Path-Note"


def test_safe_filename_strips_leading_dots():
    assert _safe_filename("...hidden") == "hidden"


def test_safe_filename_empty_input():
    assert _safe_filename("") == "Untitled"


def test_safe_filename_all_unsafe():
    assert _safe_filename(":::") == "Untitled"


# ── sanitize_content ─────────────────────────────────────────────────────────


def test_sanitize_strips_frontmatter():
    content = "---\ntags:\n  - test\ntype: zettel\n---\n\n# Title\nBody text."
    result = sanitize_content(content)
    assert result.startswith("# Title")
    assert "---" not in result


def test_sanitize_escapes_wikilinks():
    content = "See [[Some Note]] and [[Another Note]] for details."
    result = sanitize_content(content)
    assert "[[" not in result
    assert "]]" not in result
    assert f"{_WL_OPEN}Some Note{_WL_CLOSE}" in result
    assert f"{_WL_OPEN}Another Note{_WL_CLOSE}" in result


def test_sanitize_escapes_both():
    content = "---\ntags: []\n---\n\nCheck [[My Link]] here."
    result = sanitize_content(content)
    assert result == f"Check {_WL_OPEN}My Link{_WL_CLOSE} here."


def test_sanitize_preserves_plain_text():
    content = "Just plain text with no frontmatter or links."
    assert sanitize_content(content) == content


def test_sanitize_preserves_alias_syntax():
    content = "See [[Note Title|display text]] for more."
    result = sanitize_content(content)
    assert f"{_WL_OPEN}Note Title|display text{_WL_CLOSE}" in result


# ── restore_wikilinks ───────────────────────────────────────────────────────


def test_restore_wikilinks_basic():
    escaped = f"See {_WL_OPEN}Modern Portfolio Theory{_WL_CLOSE} for details."
    assert restore_wikilinks(escaped) == "See [[Modern Portfolio Theory]] for details."


def test_restore_wikilinks_multiple():
    escaped = f"Link {_WL_OPEN}A{_WL_CLOSE} and {_WL_OPEN}B{_WL_CLOSE}."
    assert restore_wikilinks(escaped) == "Link [[A]] and [[B]]."


def test_restore_wikilinks_with_parens_in_title():
    escaped = f"See {_WL_OPEN}Note (2024){_WL_CLOSE} here."
    assert restore_wikilinks(escaped) == "See [[Note (2024)]] here."


def test_restore_wikilinks_preserves_plain_text():
    text = "No escaped links in this text."
    assert restore_wikilinks(text) == text


def test_sanitize_then_restore_roundtrip():
    original = "See [[Some Note]] and [[Another (v2)]] for details."
    sanitized = sanitize_content(original)
    restored = restore_wikilinks(sanitized)
    assert restored == original


# ── extract_frontmatter ─────────────────────────────────────────────────────


def test_extract_frontmatter_basic():
    content = "---\ntags:\n  - ai\n  - investing\ntype: note\n---\n\n# Title"
    fm = extract_frontmatter(content)
    assert fm["tags"] == ["ai", "investing"]
    assert fm["type"] == "note"


def test_extract_frontmatter_plugin_fields():
    content = "---\naliases:\n  - Portfolio Opt\ncssclass: wide\npublish: true\n---\n\nBody."
    fm = extract_frontmatter(content)
    assert fm["aliases"] == ["Portfolio Opt"]
    assert fm["cssclass"] == "wide"
    assert fm["publish"] is True


def test_extract_frontmatter_missing():
    content = "# Just a heading\n\nNo frontmatter here."
    assert extract_frontmatter(content) == {}


def test_extract_frontmatter_malformed():
    content = "---\n: invalid yaml [\n---\n\nBody."
    assert extract_frontmatter(content) == {}


def test_extract_frontmatter_empty_block():
    content = "---\n---\n\nBody."
    assert extract_frontmatter(content) == {}


# ── is_valid_output ──────────────────────────────────────────────────────────


def test_valid_output_good():
    raw = """Title: A Real Atomic Note Title
Tags: tag1, tag2
Links: Some Other Note
Body:
This is real content with enough characters to pass validation.
It has multiple sentences and meaningful information about the topic."""
    assert is_valid_output(raw) is True


def test_valid_output_too_short():
    assert is_valid_output("Title: X\nBody:\nshort") is False


def test_valid_output_template_garbage():
    assert is_valid_output("## ]] some garbage text " * 10) is False


def test_valid_output_placeholder_variable():
    assert is_valid_output("{decomposed} placeholder " * 10) is False


def test_valid_output_placeholder_title():
    raw = "Title: ...\nTags: a, b\nBody:\nSome content here that is long enough to pass."
    assert is_valid_output(raw) is False


def test_valid_output_missing_body():
    raw = "Title: A Good Title That Is Long Enough\nTags: a, b\nLinks: Note A\nContent without Body label " * 3
    assert is_valid_output(raw) is False


# ── parse_atoms ──────────────────────────────────────────────────────────────

SAMPLE_CLS = {
    "para_bucket": "Resources",
    "domain": "AI/ML",
    "subdomain": "DSPy",
    "tags": ["dspy", "llm"],
}


def test_parse_atoms_single():
    raw = """Title: DSPy Signature System
Tags: dspy, signatures, type-system
Links: DSPy Predict Module, LiteLLM Integration
Body:
Signatures define the input/output contract for an LLM call in DSPy."""
    atoms = parse_atoms(raw, SAMPLE_CLS, "DSPy")
    assert len(atoms) == 1
    assert atoms[0]["title"] == "DSPy Signature System"
    assert "dspy" in atoms[0]["tags"]
    assert "[[DSPy Predict Module]]" in atoms[0]["links"]
    assert atoms[0]["para_bucket"] == "Resources"
    assert atoms[0]["source_note"] == "DSPy"


def test_parse_atoms_multiple():
    raw = """Title: First Atomic Note
Tags: tag1, tag2
Links: Note A
Body:
Content of the first note with enough characters.

===

Title: Second Atomic Note
Tags: tag3, tag4
Links: Note B, Note C
Body:
Content of the second note with enough characters."""
    atoms = parse_atoms(raw, SAMPLE_CLS, "Source")
    assert len(atoms) == 2
    assert atoms[0]["title"] == "First Atomic Note"
    assert atoms[1]["title"] == "Second Atomic Note"
    assert len(atoms[1]["links"]) == 2


def test_parse_atoms_skips_placeholder_title():
    raw = """Title: ...
Tags: a, b
Body:
Some content here."""
    atoms = parse_atoms(raw, SAMPLE_CLS, "Source")
    assert len(atoms) == 0


def test_parse_atoms_skips_short_title():
    raw = """Title: AB
Tags: a
Body:
Some content here that is long enough."""
    atoms = parse_atoms(raw, SAMPLE_CLS, "Source")
    assert len(atoms) == 0


def test_parse_atoms_skips_short_body():
    raw = """Title: A Good Title
Tags: a
Body:
Short."""
    atoms = parse_atoms(raw, SAMPLE_CLS, "Source")
    assert len(atoms) == 0


def test_parse_atoms_cleans_wikilink_brackets_in_links():
    raw = """Title: Note With Bracket Links
Tags: test
Links: [[Already Bracketed]], Plain Link
Body:
Content that is long enough to pass the minimum length check."""
    atoms = parse_atoms(raw, SAMPLE_CLS, "Source")
    assert len(atoms) == 1
    assert "[[Already Bracketed]]" in atoms[0]["links"]
    assert "[[Plain Link]]" in atoms[0]["links"]


def test_parse_atoms_strips_md_extension_from_links():
    raw = """Title: Note With Extension Links
Tags: test
Links: SomeNote.md, AnotherNote
Body:
Content that is long enough to pass the minimum length check."""
    atoms = parse_atoms(raw, SAMPLE_CLS, "Source")
    assert "[[SomeNote]]" in atoms[0]["links"]
    assert "[[AnotherNote]]" in atoms[0]["links"]


def test_parse_atoms_empty_input():
    assert parse_atoms("", SAMPLE_CLS, "Source") == []


def test_parse_atoms_no_title_match():
    raw = "Just some text without any Title: field markers."
    assert parse_atoms(raw, SAMPLE_CLS, "Source") == []


def test_parse_atoms_tags_lowercased_and_hyphenated():
    raw = """Title: Proper Title Here
Tags: Machine Learning, Deep Neural Nets
Links: Note A
Body:
Content that is long enough to pass the minimum length check."""
    atoms = parse_atoms(raw, SAMPLE_CLS, "Source")
    assert "machine-learning" in atoms[0]["tags"]
    assert "deep-neural-nets" in atoms[0]["tags"]


def test_parse_atoms_restores_wikilinks_in_body():
    raw = f"""Title: Note With Escaped Links
Tags: test
Links: Note A
Body:
This references {_WL_OPEN}Modern Portfolio Theory{_WL_CLOSE} and {_WL_OPEN}Risk Assessment{_WL_CLOSE}."""
    atoms = parse_atoms(raw, SAMPLE_CLS, "Source")
    assert len(atoms) == 1
    assert "[[Modern Portfolio Theory]]" in atoms[0]["content"]
    assert "[[Risk Assessment]]" in atoms[0]["content"]


def test_parse_atoms_handles_guillemet_links():
    raw = f"""Title: Note With Guillemet Links
Tags: test
Links: {_WL_OPEN}Already Escaped{_WL_CLOSE}, Plain Link
Body:
Content that is long enough to pass the minimum length check."""
    atoms = parse_atoms(raw, SAMPLE_CLS, "Source")
    assert len(atoms) == 1
    assert "[[Already Escaped]]" in atoms[0]["links"]
    assert "[[Plain Link]]" in atoms[0]["links"]


def test_parse_atoms_tags_strips_hashtags():
    raw = """Title: Note With Hashtag Tags
Tags: #holistic-solution-#investment-decision-making-#integrated-tools
Links: Note A
Body:
Content that is long enough to pass the minimum length check."""
    atoms = parse_atoms(raw, SAMPLE_CLS, "Source")
    assert "holistic-solution" in atoms[0]["tags"]
    assert "investment-decision-making" in atoms[0]["tags"]
    assert "integrated-tools" in atoms[0]["tags"]


# ── _build_content ────────────────────────────────────────────────────────────


def test_build_content_starts_with_frontmatter():
    note = {
        "title": "Test Note",
        "content": "Some atomic idea.",
        "tags": ["test", "zettel"],
        "domain": "Engineering",
        "subdomain": "Python",
        "source_note": "Original Note",
        "links": ["[[Other Note]]"],
    }
    content = _build_content(note)
    assert content.startswith("---\n")
    assert "type: zettel" in content


def test_build_content_has_domain_and_subdomain():
    note = {
        "title": "T",
        "content": "x",
        "tags": [],
        "domain": "Investing",
        "subdomain": "Backtesting",
        "source_note": "",
        "links": [],
    }
    content = _build_content(note)
    assert "domain: Investing" in content
    assert "subdomain: Backtesting" in content


def test_build_content_tags_in_yaml():
    note = {
        "title": "T",
        "content": "x",
        "tags": ["alpha", "beta"],
        "domain": "Other",
        "subdomain": "",
        "source_note": "",
        "links": [],
    }
    content = _build_content(note)
    assert "  - alpha\n" in content
    assert "  - beta\n" in content


def test_build_content_title_as_heading():
    note = {
        "title": "My Atomic Note",
        "content": "The idea.",
        "tags": [],
        "domain": "Other",
        "subdomain": "",
        "source_note": "",
        "links": [],
    }
    assert "# My Atomic Note" in _build_content(note)


def test_build_content_links_in_related_section():
    note = {
        "title": "T",
        "content": "x",
        "tags": [],
        "domain": "Other",
        "subdomain": "",
        "source_note": "",
        "links": ["[[Note A]]", "[[Note B]]"],
    }
    content = _build_content(note)
    assert "## Related" in content
    assert "[[Note A]]" in content
    assert "[[Note B]]" in content


def test_build_content_no_related_section_when_no_links():
    note = {
        "title": "T",
        "content": "x",
        "tags": [],
        "domain": "Other",
        "subdomain": "",
        "source_note": "",
        "links": [],
    }
    assert "## Related" not in _build_content(note)


def test_build_content_preserves_original_frontmatter():
    note = {
        "title": "T",
        "content": "x",
        "tags": ["generated-tag"],
        "domain": "AI/ML",
        "subdomain": "DSPy",
        "source_note": "Source",
        "links": [],
        "original_frontmatter": {
            "aliases": ["Portfolio Opt"],
            "cssclass": "wide",
            "publish": True,
            "tags": ["should-be-overridden"],  # conflicts with our field
            "type": "should-be-overridden",    # conflicts with our field
        },
    }
    content = _build_content(note)
    # Our generated fields take precedence
    assert "  - generated-tag" in content
    assert "domain: AI/ML" in content
    assert "type: zettel" in content
    # Original non-conflicting fields are preserved
    assert "aliases:" in content
    assert "Portfolio Opt" in content
    assert "cssclass: wide" in content
    assert "publish: true" in content
    # Conflicting original fields are NOT duplicated
    assert content.count("type:") == 1
    assert "should-be-overridden" not in content


# ── write_note ────────────────────────────────────────────────────────────────


def _make_note(**overrides) -> dict:
    base = {
        "title": "Test Note",
        "content": "An atomic idea.",
        "tags": ["test"],
        "domain": "AI/ML",
        "subdomain": "DSPy",
        "source_note": "DSPy",
        "links": [],
        "para_bucket": "Resources",
    }
    base.update(overrides)
    return base


def test_write_note_creates_file(tmp_path):
    note = _make_note()
    written = write_note(tmp_path, note)
    assert written.exists()
    assert written.name == "Test Note.md"
    assert "# Test Note" in written.read_text()


def test_write_note_correct_bucket_path(tmp_path):
    note = _make_note(para_bucket="Projects", subdomain="Hextropian")
    write_note(tmp_path, note)
    expected = tmp_path / "1. Projects" / "Hextropian" / "Test Note.md"
    assert expected.exists()


def test_write_note_archive_bucket(tmp_path):
    note = _make_note(para_bucket="Archive", subdomain="")
    write_note(tmp_path, note)
    expected = tmp_path / "4. Archive" / "Test Note.md"
    assert expected.exists()


def test_write_note_no_subdomain(tmp_path):
    note = _make_note(subdomain="")
    write_note(tmp_path, note)
    expected = tmp_path / "3. Resources" / "Test Note.md"
    assert expected.exists()


def test_write_note_collision_handling(tmp_path):
    note = _make_note(subdomain="")
    write_note(tmp_path, note)
    write_note(tmp_path, note)
    files = list((tmp_path / "3. Resources").rglob("*.md"))
    assert len(files) == 2
    names = {f.name for f in files}
    assert "Test Note.md" in names
    assert "Test Note_1.md" in names


def test_write_note_unsafe_title(tmp_path):
    note = _make_note(title="Note: With/Bad Chars", subdomain="")
    written = write_note(tmp_path, note)
    assert written.exists()
    assert "/" not in written.name
    assert ":" not in written.name


# ── write_moc ────────────────────────────────────────────────────────────────


def test_write_moc_creates_one_file_per_domain(tmp_path):
    notes = [
        {"title": "Note A", "domain": "AI/ML"},
        {"title": "Note B", "domain": "AI/ML"},
        {"title": "Note C", "domain": "Investing"},
    ]
    write_moc(tmp_path, notes)
    assert (tmp_path / "MOC" / "AI-ML.md").exists()
    assert (tmp_path / "MOC" / "Investing.md").exists()


def test_write_moc_contains_wikilinks(tmp_path):
    notes = [{"title": "Backtesting", "domain": "Investing"}]
    write_moc(tmp_path, notes)
    content = (tmp_path / "MOC" / "Investing.md").read_text()
    assert "[[Backtesting]]" in content


def test_write_moc_has_moc_frontmatter(tmp_path):
    notes = [{"title": "Note", "domain": "Engineering"}]
    write_moc(tmp_path, notes)
    content = (tmp_path / "MOC" / "Engineering.md").read_text()
    assert "type: moc" in content
    assert "  - moc" in content


def test_write_moc_deduplicates_titles(tmp_path):
    notes = [
        {"title": "Same Note", "domain": "AI/ML"},
        {"title": "Same Note", "domain": "AI/ML"},
    ]
    write_moc(tmp_path, notes)
    content = (tmp_path / "MOC" / "AI-ML.md").read_text()
    assert content.count("[[Same Note]]") == 1


# ── vlt helpers (with subprocess mock) ───────────────────────────────────────


def test_list_vault_notes_json_format():
    files_json = json.dumps(["1. Projects/DSPy/DSPy.md", "2. Areas/Health/Supplements.md"])
    with patch("zettelvault_dspy.vlt_run", return_value=files_json):
        titles = list_vault_notes("FakeVault")
    assert "DSPy" in titles
    assert "Supplements" in titles


def test_list_vault_notes_plain_text_fallback():
    plain = "1. Projects/Note One.md\n2. Areas/Note Two.md\n"
    with patch("zettelvault_dspy.vlt_run", return_value=plain):
        titles = list_vault_notes("FakeVault")
    assert "Note One" in titles
    assert "Note Two" in titles


def test_list_vault_notes_skips_non_md():
    mixed = json.dumps(["folder/Note.md", "folder/image.png", "folder/.obsidian/config.json"])
    with patch("zettelvault_dspy.vlt_run", return_value=mixed):
        titles = list_vault_notes("FakeVault")
    assert titles == ["Note"]


def test_read_note_returns_empty_on_error():
    with patch("zettelvault_dspy.vlt_run", side_effect=RuntimeError("not found")):
        result = read_note("FakeVault", "Missing Note")
    assert result == ""


# ── Integration: requires OPENROUTER_API_KEY ─────────────────────────────────


@pytest.mark.integration
def test_classify_real_note():
    """Classifies a short note using the real LLM. Requires OPENROUTER_API_KEY."""
    from zettelvault_dspy import classify_note

    result = classify_note(
        title="DSPy",
        content=(
            "DSPy is a framework for programming with language models. "
            "It uses signatures to define typed input/output contracts and "
            "provides optimizers like BootstrapFewShot and MIPROv2."
        ),
    )
    assert result["para_bucket"] in {"Projects", "Areas", "Resources", "Archive"}
    assert result["domain"]
    assert isinstance(result["tags"], list)
    assert len(result["tags"]) > 0


@pytest.mark.integration
def test_classify_travel_note():
    """Travel content should map to Resources or Areas, with Travel domain."""
    from zettelvault_dspy import classify_note

    result = classify_note(
        title="Bidwell Hotel",
        content=(
            "The Bidwell Hotel in Portland, Oregon. Great location near Powell's Books. "
            "Comfortable rooms, friendly staff. Good base for exploring the city."
        ),
    )
    assert result["para_bucket"] in {"Resources", "Areas"}
    assert "Travel" in result["domain"] or "travel" in [t.lower() for t in result["tags"]]


@pytest.mark.integration
def test_decompose_real_note():
    """Decompose a note using the real LLM. Requires OPENROUTER_API_KEY."""
    from zettelvault_dspy import _init_lm, decompose_note

    _init_lm(use_rlm=False)  # Use Predict for faster test

    data = {
        "content": (
            "---\ntags: [ai, investing]\n---\n\n"
            "# AI Portfolio Optimizer\n\n"
            "This project combines portfolio optimization with machine learning risk management. "
            "It provides real-time risk scoring and automated rebalancing. "
            "The system uses [[Modern Portfolio Theory]] as a foundation "
            "and adds ML-based predictions for [[Risk Assessment]].\n\n"
            "## Features\n\n"
            "- Real-time portfolio risk scoring\n"
            "- Automated rebalancing triggers\n"
            "- ML-based market regime detection\n"
            "- Integration with broker APIs"
        ),
        "classification": {
            "para_bucket": "Projects",
            "domain": "Investing",
            "subdomain": "Portfolio Optimization",
            "tags": ["ai", "investing", "portfolio"],
        },
    }
    related = ["Modern Portfolio Theory", "Risk Assessment", "Market Analysis"]
    atoms, rlm_iters, rlm_subs = decompose_note("AI Portfolio Optimizer", data, related)
    assert len(atoms) >= 1
    for atom in atoms:
        assert atom["title"]
        assert atom["content"]
        assert atom["para_bucket"] == "Projects"
        assert atom["source_note"] == "AI Portfolio Optimizer"
    # Predict mode: no RLM iterations
    assert rlm_iters == 0
