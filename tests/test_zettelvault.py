"""
Comprehensive unit and integration tests for the zettelvault package.

Tests are organized as specifications -- they define correctness.
Some tests may initially fail because the code needs fixing; that is
intentional. The tests describe the DESIRED end state.

Run unit tests (no API key):
    uv run -p 3.13 -- pytest tests/test_zettelvault.py -v -m "not integration"

Run integration tests (requires OPENROUTER_API_KEY):
    uv run --env-file .env -p 3.13 -- pytest tests/test_zettelvault.py -v -m integration
"""

import datetime
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from zettelvault.config import (
    config_get,
    deep_merge,
    load_config,
)
from zettelvault.sanitize import (
    SafeEncoder,
    WL_CLOSE,
    WL_OPEN,
    extract_frontmatter,
    restore_wikilinks,
    sanitize_content,
)
from zettelvault.vault_io import (
    copy_obsidian_config,
    list_vault_notes,
    read_note,
    resolve_vault_path,
    vlt_run,
)
from zettelvault.classify import (
    build_concept_index,
    classify_note,
    find_related,
)
from zettelvault.decompose import (
    _fallback_atom,
    _decompose_with_predict,
    _decompose_with_rlm,
    _summarize_trajectory,
    decompose_and_link,
    decompose_note,
    is_valid_output,
    parse_atoms,
)
from zettelvault.writer import (
    _build_content,
    _safe_filename,
    write_moc,
    write_note,
)
from zettelvault.resolve import resolve_links
from zettelvault.pipeline import _fmt_duration, _progress_line
from zettelvault.sample import (
    assign_size_bucket,
    classify_structure,
    compute_size_buckets,
    extract_features,
    greedy_select,
    sample_vault,
    _coverage_slots,
)


# ============================================================================
# Config: deep_merge
# ============================================================================


class TestDeepMerge:
    def test_flat_merge(self):
        base = {"a": 1, "b": 2}
        deep_merge(base, {"b": 3, "c": 4})
        assert base == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge_preserves_unset_keys(self):
        base = {"model": {"id": "a", "max_tokens": 1000}}
        deep_merge(base, {"model": {"id": "b"}})
        assert base["model"]["id"] == "b"
        assert base["model"]["max_tokens"] == 1000

    def test_override_replaces_entire_value(self):
        base = {"a": 1}
        deep_merge(base, {"a": {"nested": True}})
        assert base["a"] == {"nested": True}

    def test_non_dict_override_replaces_dict(self):
        base = {"model": {"id": "a"}}
        deep_merge(base, {"model": "simple-string"})
        assert base["model"] == "simple-string"

    def test_empty_override_is_noop(self):
        base = {"a": 1}
        deep_merge(base, {})
        assert base == {"a": 1}

    def test_empty_base_takes_override(self):
        base = {}
        deep_merge(base, {"a": 1, "b": {"c": 2}})
        assert base == {"a": 1, "b": {"c": 2}}


# ============================================================================
# Config: load_config
# ============================================================================


class TestLoadConfig:
    def test_default_only(self, tmp_path):
        """load_config with no explicit path loads config.yaml and config.local.yaml."""
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text("model:\n  id: test/model\n  max_tokens: 4000\n")
        cfg = load_config(cfg_file)
        assert cfg["model"]["id"] == "test/model"
        assert cfg["model"]["max_tokens"] == 4000

    def test_explicit_file_overlays_defaults(self, tmp_path):
        cfg_file = tmp_path / "override.yaml"
        cfg_file.write_text("model:\n  id: override/model\n")
        cfg = load_config(cfg_file)
        # The explicit file should set model.id
        assert cfg["model"]["id"] == "override/model"

    def test_missing_explicit_file_does_not_crash(self, tmp_path):
        missing = tmp_path / "nonexistent.yaml"
        # Should not raise; just loads defaults
        cfg = load_config(missing)
        assert isinstance(cfg, dict)

    def test_non_dict_content_ignored(self, tmp_path):
        cfg_file = tmp_path / "bad.yaml"
        cfg_file.write_text("just a string\n")
        cfg = load_config(cfg_file)
        # Non-dict YAML should not crash or pollute config
        assert isinstance(cfg, dict)


# ============================================================================
# Config: config_get
# ============================================================================


class TestGet:
    def test_dotted_key_access(self):
        cfg = {"model": {"id": "test", "route": {"only": ["X"]}}}
        assert config_get(cfg, "model.id") == "test"
        assert config_get(cfg, "model.route.only") == ["X"]

    def test_missing_key_returns_default(self):
        cfg = {"model": {"id": "test"}}
        assert config_get(cfg, "model.missing", 42) == 42
        assert config_get(cfg, "nonexistent.deep.key", "default") == "default"

    def test_intermediate_non_dict_returns_default(self):
        cfg = {"model": "just-a-string"}
        assert config_get(cfg, "model.id", "fallback") == "fallback"

    def test_top_level_key(self):
        cfg = {"simple_key": "value"}
        assert config_get(cfg, "simple_key") == "value"


# ============================================================================
# Vault I/O: vlt_run
# ============================================================================


class TestVltRun:
    def test_successful_call(self):
        with patch("zettelvault.vault_io.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="  output text  ", stderr=""
            )
            result = vlt_run("TestVault", "files", "--json")
        assert result == "output text"
        mock_run.assert_called_once_with(
            ["vlt", "vault=TestVault", "files", "--json"],
            capture_output=True,
            text=True,
        )

    def test_failed_call_raises_runtime_error(self):
        with patch("zettelvault.vault_io.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stdout="", stderr="vault not found"
            )
            with pytest.raises(RuntimeError, match="vault not found"):
                vlt_run("Missing", "read", "file=X")

    def test_command_construction(self):
        with patch("zettelvault.vault_io.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
            vlt_run("MyVault", "read", "file=Note Title")
        expected_cmd = ["vlt", "vault=MyVault", "read", "file=Note Title"]
        mock_run.assert_called_once_with(expected_cmd, capture_output=True, text=True)


# ============================================================================
# Vault I/O: resolve_vault_path
# ============================================================================


class TestResolveVaultPath:
    def test_success(self):
        vaults_json = json.dumps(
            [
                {"name": "Personal", "path": "/Users/test/vaults/Personal"},
                {"name": "Work", "path": "/Users/test/vaults/Work"},
            ]
        )
        with patch("zettelvault.vault_io.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=vaults_json, stderr=""
            )
            result = resolve_vault_path("Personal")
        assert result == Path("/Users/test/vaults/Personal")

    def test_vault_not_found_returns_none(self):
        vaults_json = json.dumps([{"name": "Other", "path": "/tmp/Other"}])
        with patch("zettelvault.vault_io.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=vaults_json, stderr=""
            )
            result = resolve_vault_path("Missing")
        assert result is None

    def test_subprocess_error_returns_none(self):
        with patch(
            "zettelvault.vault_io.subprocess.run", side_effect=FileNotFoundError
        ):
            result = resolve_vault_path("Any")
        assert result is None


# ============================================================================
# Vault I/O: list_vault_notes
# ============================================================================


class TestListVaultNotes:
    def test_json_format(self):
        files_json = json.dumps(
            ["1. Projects/DSPy/DSPy.md", "2. Areas/Health/Supplements.md"]
        )
        with patch("zettelvault.vault_io.vlt_run", return_value=files_json):
            titles = list_vault_notes("FakeVault")
        assert "DSPy" in titles
        assert "Supplements" in titles

    def test_plain_text_fallback(self):
        plain = "1. Projects/Note One.md\n2. Areas/Note Two.md\n"
        with patch("zettelvault.vault_io.vlt_run", return_value=plain):
            titles = list_vault_notes("FakeVault")
        assert "Note One" in titles
        assert "Note Two" in titles

    def test_skips_non_md_files(self):
        mixed = json.dumps(
            ["folder/Note.md", "folder/image.png", "folder/.obsidian/config.json"]
        )
        with patch("zettelvault.vault_io.vlt_run", return_value=mixed):
            titles = list_vault_notes("FakeVault")
        assert titles == ["Note"]

    def test_empty_vault(self):
        with patch("zettelvault.vault_io.vlt_run", return_value="[]"):
            titles = list_vault_notes("FakeVault")
        assert titles == []


# ============================================================================
# Vault I/O: read_note
# ============================================================================


class TestReadNote:
    def test_success(self):
        with patch(
            "zettelvault.vault_io.vlt_run", return_value="# My Note\n\nContent here."
        ):
            result = read_note("FakeVault", "My Note")
        assert "# My Note" in result

    def test_error_returns_empty_string(self):
        with patch(
            "zettelvault.vault_io.vlt_run", side_effect=RuntimeError("not found")
        ):
            result = read_note("FakeVault", "Missing Note")
        assert result == ""


# ============================================================================
# Vault I/O: copy_obsidian_config
# ============================================================================


class TestCopyObsidianConfig:
    def test_vault_not_found_skips(self, capsys):
        with patch("zettelvault.vault_io.resolve_vault_path", return_value=None):
            copy_obsidian_config("Missing", Path("/tmp/dest"))
        assert "Could not resolve" in capsys.readouterr().out

    def test_no_obsidian_dir_skips(self, tmp_path, capsys):
        src = tmp_path / "source"
        src.mkdir()
        # No .obsidian dir in source
        with patch("zettelvault.vault_io.resolve_vault_path", return_value=src):
            copy_obsidian_config("Source", tmp_path / "dest")
        assert "No .obsidian" in capsys.readouterr().out

    def test_already_exists_skips(self, tmp_path, capsys):
        src = tmp_path / "source"
        (src / ".obsidian").mkdir(parents=True)
        dest = tmp_path / "dest"
        (dest / ".obsidian").mkdir(parents=True)
        with patch("zettelvault.vault_io.resolve_vault_path", return_value=src):
            copy_obsidian_config("Source", dest)
        assert "already exists" in capsys.readouterr().out

    def test_successful_copy(self, tmp_path, capsys):
        src = tmp_path / "source"
        obsidian = src / ".obsidian"
        obsidian.mkdir(parents=True)
        (obsidian / "app.json").write_text("{}")
        dest = tmp_path / "dest"
        dest.mkdir()
        with patch("zettelvault.vault_io.resolve_vault_path", return_value=src):
            copy_obsidian_config("Source", dest)
        assert (dest / ".obsidian" / "app.json").exists()
        assert "Copied .obsidian" in capsys.readouterr().out


# ============================================================================
# Content Processing: sanitize_content
# ============================================================================


class TestSanitizeContent:
    def test_strips_frontmatter(self):
        content = "---\ntags:\n  - test\ntype: zettel\n---\n\n# Title\nBody text."
        result = sanitize_content(content)
        assert result.startswith("# Title")
        assert "---" not in result

    def test_escapes_wikilinks(self):
        content = "See [[Some Note]] and [[Another Note]] for details."
        result = sanitize_content(content)
        assert "[[" not in result
        assert "]]" not in result
        assert f"{WL_OPEN}Some Note{WL_CLOSE}" in result
        assert f"{WL_OPEN}Another Note{WL_CLOSE}" in result

    def test_escapes_wikilinks_with_parentheses(self):
        content = "See [[Note (2024)]] here."
        result = sanitize_content(content)
        assert f"{WL_OPEN}Note (2024){WL_CLOSE}" in result
        assert "[[" not in result

    def test_escapes_wikilinks_with_aliases(self):
        content = "See [[Note Title|display text]] for more."
        result = sanitize_content(content)
        assert f"{WL_OPEN}Note Title|display text{WL_CLOSE}" in result

    def test_escapes_nested_brackets(self):
        content = "See [[Note with [brackets]]] for details."
        result = sanitize_content(content)
        assert "[[" not in result

    def test_both_frontmatter_and_wikilinks(self):
        content = "---\ntags: []\n---\n\nCheck [[My Link]] here."
        result = sanitize_content(content)
        assert result == f"Check {WL_OPEN}My Link{WL_CLOSE} here."

    def test_preserves_plain_text(self):
        content = "Just plain text with no frontmatter or links."
        assert sanitize_content(content) == content

    def test_strips_and_trims(self):
        content = "---\na: 1\n---\n\n\n  text  \n\n"
        result = sanitize_content(content)
        assert result == "text"


# ============================================================================
# Content Processing: restore_wikilinks
# ============================================================================


class TestRestoreWikilinks:
    def test_basic_restoration(self):
        escaped = f"See {WL_OPEN}Modern Portfolio Theory{WL_CLOSE} for details."
        assert (
            restore_wikilinks(escaped) == "See [[Modern Portfolio Theory]] for details."
        )

    def test_multiple_links(self):
        escaped = f"Link {WL_OPEN}A{WL_CLOSE} and {WL_OPEN}B{WL_CLOSE}."
        assert restore_wikilinks(escaped) == "Link [[A]] and [[B]]."

    def test_parenthetical_content(self):
        escaped = f"See {WL_OPEN}Note (2024){WL_CLOSE} here."
        assert restore_wikilinks(escaped) == "See [[Note (2024)]] here."

    def test_alias_syntax(self):
        escaped = f"See {WL_OPEN}Note|alias{WL_CLOSE} here."
        assert restore_wikilinks(escaped) == "See [[Note|alias]] here."

    def test_plain_text_unchanged(self):
        text = "No escaped links in this text."
        assert restore_wikilinks(text) == text

    def test_roundtrip_with_sanitize(self):
        original = "See [[Some Note]] and [[Another (v2)]] for details."
        sanitized = sanitize_content(original)
        restored = restore_wikilinks(sanitized)
        assert restored == original


# ============================================================================
# Content Processing: extract_frontmatter
# ============================================================================


class TestExtractFrontmatter:
    def test_basic_yaml(self):
        content = "---\ntags:\n  - ai\n  - investing\ntype: note\n---\n\n# Title"
        fm = extract_frontmatter(content)
        assert fm["tags"] == ["ai", "investing"]
        assert fm["type"] == "note"

    def test_plugin_fields(self):
        content = "---\naliases:\n  - Portfolio Opt\ncssclass: wide\npublish: true\n---\n\nBody."
        fm = extract_frontmatter(content)
        assert fm["aliases"] == ["Portfolio Opt"]
        assert fm["cssclass"] == "wide"
        assert fm["publish"] is True

    def test_missing_frontmatter(self):
        content = "# Just a heading\n\nNo frontmatter here."
        assert extract_frontmatter(content) == {}

    def test_malformed_yaml(self):
        content = "---\n: invalid yaml [\n---\n\nBody."
        assert extract_frontmatter(content) == {}

    def test_empty_block(self):
        content = "---\n---\n\nBody."
        assert extract_frontmatter(content) == {}

    def test_non_dict_yaml(self):
        content = "---\n- just\n- a\n- list\n---\n\nBody."
        assert extract_frontmatter(content) == {}


# ============================================================================
# Content Processing: is_valid_output
# ============================================================================


class TestIsValidOutput:
    def test_valid_output(self):
        raw = (
            "Title: A Real Atomic Note Title\n"
            "Tags: tag1, tag2\n"
            "Links: Some Other Note\n"
            "Body:\n"
            "This is real content with enough characters to pass validation.\n"
            "It has multiple sentences and meaningful information about the topic."
        )
        assert is_valid_output(raw) is True

    def test_too_short(self):
        assert is_valid_output("Title: X\nBody:\nshort") is False

    def test_template_garbage_brackets(self):
        assert is_valid_output("## ]] some garbage text " * 10) is False

    def test_template_garbage_braces(self):
        assert is_valid_output("{decomposed} placeholder " * 10) is False

    def test_placeholder_title(self):
        raw = "Title: ...\nTags: a, b\nBody:\nSome content here that is long enough to pass."
        assert is_valid_output(raw) is False

    def test_missing_body_marker(self):
        raw = (
            "Title: A Good Title That Is Long Enough\nTags: a, b\nLinks: Note A\nContent without Body label "
            * 3
        )
        assert is_valid_output(raw) is False

    def test_short_title_fails(self):
        raw = "Title: AB\nTags: a\nBody:\nThis is content that is definitely long enough to pass the check."
        assert is_valid_output(raw) is False


# ============================================================================
# Concept Index: build_concept_index
# ============================================================================


class TestBuildConceptIndex:
    def test_word_extraction(self):
        classified = {
            "Machine Learning Basics": {
                "content": "This note covers gradient descent and backpropagation."
            },
        }
        index = build_concept_index(classified)
        # "Machine" is 7 chars, should be extracted
        assert "machine" in index
        assert "Machine Learning Basics" in index["machine"]

    def test_minimum_word_length(self):
        classified = {
            "Note": {"content": "The fox ran far and wide."},
        }
        index = build_concept_index(classified)
        # "The", "fox", "ran", "far", "and" are all <= 3 chars, should be excluded
        assert "the" not in index
        assert "fox" not in index
        assert "ran" not in index
        assert "wide" in index  # 4 chars, should be included

    def test_case_normalization(self):
        classified = {
            "Test Note": {
                "content": "DSPy Framework is great. DSPY is case-insensitive."
            },
        }
        index = build_concept_index(classified)
        assert "dspy" in index
        assert "framework" in index

    def test_inverted_index_structure(self):
        classified = {
            "Note A": {"content": "Python programming guide."},
            "Note B": {"content": "Python data science."},
        }
        index = build_concept_index(classified)
        assert "python" in index
        assert "Note A" in index["python"]
        assert "Note B" in index["python"]


# ============================================================================
# Concept Index: find_related
# ============================================================================


class TestFindRelated:
    def test_top_n_related(self):
        index = {
            "python": ["Note A", "Note B", "Note C"],
            "data": ["Note B", "Note C"],
            "machine": ["Note C"],
        }
        related = find_related("Note A", "Python data machine learning", index, top=2)
        # Note C shares 3 words, Note B shares 2
        assert related[0] == "Note C"
        assert related[1] == "Note B"

    def test_self_exclusion(self):
        index = {"python": ["Note A", "Note B"]}
        related = find_related("Note A", "Python programming", index)
        assert "Note A" not in related

    def test_empty_index(self):
        related = find_related("Note A", "Python programming", {})
        assert related == []

    def test_no_overlap(self):
        index = {"completely": ["Note B"], "unrelated": ["Note B"]}
        related = find_related("Note A", "Different words here", index)
        assert related == []


# ============================================================================
# Classification: classify_note
# ============================================================================


class TestClassifyNote:
    def test_unit_with_mocked_classifier(self):
        mock_result = MagicMock()
        mock_result.para_bucket = "Resources"
        mock_result.domain = "AI/ML"
        mock_result.subdomain = "DSPy"
        mock_result.tags = ["dspy", "framework"]

        mock_classifier = MagicMock(return_value=mock_result)
        cfg = {"pipeline": {"max_input_chars": 8000}}

        result = classify_note(
            "DSPy",
            "DSPy is a framework for LLM programming.",
            classifier=mock_classifier,
            cfg=cfg,
        )

        assert result["para_bucket"] == "Resources"
        assert result["domain"] == "AI/ML"
        assert result["subdomain"] == "DSPy"
        assert result["tags"] == ["dspy", "framework"]

    def test_unit_truncates_content_via_sanitize(self):
        """Verify the content is sanitized and truncated before classification."""
        mock_result = MagicMock()
        mock_result.para_bucket = "Resources"
        mock_result.domain = "Other"
        mock_result.subdomain = ""
        mock_result.tags = []

        mock_classifier = MagicMock(return_value=mock_result)
        cfg = {"pipeline": {"max_input_chars": 50}}

        content = "---\ntags: []\n---\n\n" + "A" * 200
        classify_note("Test", content, classifier=mock_classifier, cfg=cfg)
        # The content arg should be sanitized (no frontmatter) and truncated to 50 chars
        call_kwargs = mock_classifier.call_args
        actual_content = (
            call_kwargs.kwargs.get("content")
            or call_kwargs[1].get("content")
            or call_kwargs[0][1]
        )
        assert len(actual_content) <= 50
        assert "---" not in actual_content


# ============================================================================
# Parse Atoms
# ============================================================================


SAMPLE_CLS = {
    "para_bucket": "Resources",
    "domain": "AI/ML",
    "subdomain": "DSPy",
    "tags": ["dspy", "llm"],
}


class TestParseAtoms:
    def test_single_atom(self):
        raw = (
            "Title: DSPy Signature System\n"
            "Tags: dspy, signatures, type-system\n"
            "Links: DSPy Predict Module, LiteLLM Integration\n"
            "Body:\n"
            "Signatures define the input/output contract for an LLM call in DSPy."
        )
        atoms = parse_atoms(raw, SAMPLE_CLS, "DSPy")
        assert len(atoms) == 1
        assert atoms[0]["title"] == "DSPy Signature System"
        assert "dspy" in atoms[0]["tags"]
        assert "[[DSPy Predict Module]]" in atoms[0]["links"]
        assert atoms[0]["para_bucket"] == "Resources"
        assert atoms[0]["source_note"] == "DSPy"

    def test_multiple_atoms(self):
        raw = (
            "Title: First Atomic Note\n"
            "Tags: tag1, tag2\n"
            "Links: Note A\n"
            "Body:\n"
            "Content of the first note with enough characters.\n"
            "\n===\n\n"
            "Title: Second Atomic Note\n"
            "Tags: tag3, tag4\n"
            "Links: Note B, Note C\n"
            "Body:\n"
            "Content of the second note with enough characters."
        )
        atoms = parse_atoms(raw, SAMPLE_CLS, "Source")
        assert len(atoms) == 2
        assert atoms[0]["title"] == "First Atomic Note"
        assert atoms[1]["title"] == "Second Atomic Note"
        assert len(atoms[1]["links"]) == 2

    def test_skips_placeholder_title(self):
        raw = "Title: ...\nTags: a, b\nBody:\nSome content here."
        atoms = parse_atoms(raw, SAMPLE_CLS, "Source")
        assert len(atoms) == 0

    def test_skips_short_title(self):
        raw = "Title: AB\nTags: a\nBody:\nSome content here that is long enough."
        atoms = parse_atoms(raw, SAMPLE_CLS, "Source")
        assert len(atoms) == 0

    def test_skips_short_body(self):
        raw = "Title: A Good Title\nTags: a\nBody:\nShort."
        atoms = parse_atoms(raw, SAMPLE_CLS, "Source")
        assert len(atoms) == 0

    def test_cleans_wikilink_brackets_in_links(self):
        raw = (
            "Title: Note With Bracket Links\n"
            "Tags: test\n"
            "Links: [[Already Bracketed]], Plain Link\n"
            "Body:\n"
            "Content that is long enough to pass the minimum length check."
        )
        atoms = parse_atoms(raw, SAMPLE_CLS, "Source")
        assert len(atoms) == 1
        assert "[[Already Bracketed]]" in atoms[0]["links"]
        assert "[[Plain Link]]" in atoms[0]["links"]

    def test_strips_md_extension_from_links(self):
        raw = (
            "Title: Note With Extension Links\n"
            "Tags: test\n"
            "Links: SomeNote.md, AnotherNote\n"
            "Body:\n"
            "Content that is long enough to pass the minimum length check."
        )
        atoms = parse_atoms(raw, SAMPLE_CLS, "Source")
        assert "[[SomeNote]]" in atoms[0]["links"]
        assert "[[AnotherNote]]" in atoms[0]["links"]

    def test_empty_input(self):
        assert parse_atoms("", SAMPLE_CLS, "Source") == []

    def test_no_title_match(self):
        raw = "Just some text without any Title: field markers."
        assert parse_atoms(raw, SAMPLE_CLS, "Source") == []

    def test_tags_lowercased_and_hyphenated(self):
        raw = (
            "Title: Proper Title Here\n"
            "Tags: Machine Learning, Deep Neural Nets\n"
            "Links: Note A\n"
            "Body:\n"
            "Content that is long enough to pass the minimum length check."
        )
        atoms = parse_atoms(raw, SAMPLE_CLS, "Source")
        assert "machine-learning" in atoms[0]["tags"]
        assert "deep-neural-nets" in atoms[0]["tags"]

    def test_original_frontmatter_propagation(self):
        raw = (
            "Title: Test Note\n"
            "Tags: test\n"
            "Links: Note A\n"
            "Body:\n"
            "Content that is long enough to pass the minimum length check."
        )
        fm = {"aliases": ["Test"], "publish": True}
        atoms = parse_atoms(raw, SAMPLE_CLS, "Source", original_frontmatter=fm)
        assert atoms[0]["original_frontmatter"] == fm

    def test_original_frontmatter_default_empty(self):
        raw = (
            "Title: Test Note\n"
            "Tags: test\n"
            "Links: Note A\n"
            "Body:\n"
            "Content that is long enough to pass the minimum length check."
        )
        atoms = parse_atoms(raw, SAMPLE_CLS, "Source")
        assert atoms[0]["original_frontmatter"] == {}

    def test_filters_ellipsis_tags(self):
        raw = (
            "Title: Note With Ellipsis Tag\n"
            "Tags: real-tag, ..., another-tag\n"
            "Links: Note A\n"
            "Body:\n"
            "Content that is long enough to pass the minimum length check."
        )
        atoms = parse_atoms(raw, SAMPLE_CLS, "Source")
        assert "..." not in atoms[0]["tags"]
        assert "real-tag" in atoms[0]["tags"]
        assert "another-tag" in atoms[0]["tags"]

    def test_filters_ellipsis_links(self):
        raw = (
            "Title: Note With Ellipsis Link\n"
            "Tags: test\n"
            "Links: Real Note, ..., Another Note\n"
            "Body:\n"
            "Content that is long enough to pass the minimum length check."
        )
        atoms = parse_atoms(raw, SAMPLE_CLS, "Source")
        assert "[[Real Note]]" in atoms[0]["links"]
        assert "[[Another Note]]" in atoms[0]["links"]
        # "..." should have been filtered out
        assert all("..." not in lnk for lnk in atoms[0]["links"])

    def test_restores_wikilinks_in_body(self):
        raw = (
            f"Title: Note With Escaped Links\n"
            f"Tags: test\n"
            f"Links: Note A\n"
            f"Body:\n"
            f"This references {WL_OPEN}Modern Portfolio Theory{WL_CLOSE} and {WL_OPEN}Risk Assessment{WL_CLOSE}."
        )
        atoms = parse_atoms(raw, SAMPLE_CLS, "Source")
        assert len(atoms) == 1
        assert "[[Modern Portfolio Theory]]" in atoms[0]["content"]
        assert "[[Risk Assessment]]" in atoms[0]["content"]

    def test_guillemet_link_restoration(self):
        raw = (
            f"Title: Note With Guillemet Links\n"
            f"Tags: test\n"
            f"Links: {WL_OPEN}Already Escaped{WL_CLOSE}, Plain Link\n"
            f"Body:\n"
            f"Content that is long enough to pass the minimum length check."
        )
        atoms = parse_atoms(raw, SAMPLE_CLS, "Source")
        assert len(atoms) == 1
        assert "[[Already Escaped]]" in atoms[0]["links"]
        assert "[[Plain Link]]" in atoms[0]["links"]

    def test_hashtag_splitting(self):
        raw = (
            "Title: Note With Hashtag Tags\n"
            "Tags: #holistic-solution-#investment-decision-making-#integrated-tools\n"
            "Links: Note A\n"
            "Body:\n"
            "Content that is long enough to pass the minimum length check."
        )
        atoms = parse_atoms(raw, SAMPLE_CLS, "Source")
        assert "holistic-solution" in atoms[0]["tags"]
        assert "investment-decision-making" in atoms[0]["tags"]
        assert "integrated-tools" in atoms[0]["tags"]


# ============================================================================
# Decomposition: _fallback_atom
# ============================================================================


class TestFallbackAtom:
    def test_structure(self):
        classification = {
            "para_bucket": "Projects",
            "domain": "AI/ML",
            "subdomain": "DSPy",
            "tags": ["dspy", "framework"],
        }
        atom = _fallback_atom("My Note", "Original content here.", classification)
        assert atom["title"] == "My Note"
        assert atom["content"] == "Original content here."
        assert atom["links"] == []
        assert atom["tags"] == ["dspy", "framework"]
        assert atom["para_bucket"] == "Projects"
        assert atom["domain"] == "AI/ML"
        assert atom["subdomain"] == "DSPy"
        assert atom["source_note"] == "My Note"
        assert atom["original_frontmatter"] == {}

    def test_tags_from_classification_propagated(self):
        classification = {
            "para_bucket": "Resources",
            "domain": "Other",
            "subdomain": "",
            "tags": ["tag1", "tag2"],
        }
        atom = _fallback_atom("X", "Content", classification)
        assert atom["tags"] == ["tag1", "tag2"]

    def test_source_note_equals_title(self):
        classification = {
            "para_bucket": "Resources",
            "domain": "Other",
            "subdomain": "",
            "tags": [],
        }
        atom = _fallback_atom("Original Title", "Content", classification)
        assert atom["source_note"] == "Original Title"

    def test_original_frontmatter_passed_through(self):
        classification = {
            "para_bucket": "Resources",
            "domain": "Other",
            "subdomain": "",
            "tags": [],
        }
        fm = {"aliases": ["alias1"], "publish": True}
        atom = _fallback_atom("X", "Content", classification, original_frontmatter=fm)
        assert atom["original_frontmatter"] == fm


# ============================================================================
# Decomposition: _decompose_with_predict
# ============================================================================


class TestDecomposeWithPredict:
    def test_successful_first_attempt(self):
        valid_output = (
            "Title: A Proper Atomic Note Title\n"
            "Tags: tag1, tag2\n"
            "Links: Related Note\n"
            "Body:\n"
            "This is real content with enough characters to pass validation. "
            "It has meaningful information and multiple sentences."
        )
        mock_predict = MagicMock()
        mock_predict.return_value = MagicMock(decomposed=valid_output)
        cfg = {
            "pipeline": {
                "max_retries": 3,
                "retry_temp_start": 0.1,
                "retry_temp_step": 0.3,
                "max_input_chars": 8000,
            }
        }

        result = _decompose_with_predict(
            "Title",
            "Content",
            "Related A, Related B",
            decomposer_predict=mock_predict,
            cfg=cfg,
        )
        assert result == valid_output
        assert mock_predict.call_count == 1

    def test_retry_with_escalating_temperature(self):
        valid_output = (
            "Title: A Proper Atomic Note Title\n"
            "Tags: tag1, tag2\n"
            "Links: Related Note\n"
            "Body:\n"
            "This is real content with enough characters to pass validation. "
            "Multiple sentences for validation."
        )
        mock_predict = MagicMock()
        # First attempt returns garbage, second returns valid
        mock_predict.side_effect = [
            MagicMock(decomposed="short"),
            MagicMock(decomposed=valid_output),
        ]
        cfg = {
            "pipeline": {
                "max_retries": 3,
                "retry_temp_start": 0.1,
                "retry_temp_step": 0.3,
                "max_input_chars": 8000,
            }
        }

        result = _decompose_with_predict(
            "Title",
            "Content",
            "Related",
            decomposer_predict=mock_predict,
            cfg=cfg,
        )
        assert result == valid_output
        assert mock_predict.call_count == 2

    def test_all_retries_exhausted(self):
        mock_predict = MagicMock()
        mock_predict.return_value = MagicMock(decomposed="garbage")
        cfg = {
            "pipeline": {
                "max_retries": 3,
                "retry_temp_start": 0.1,
                "retry_temp_step": 0.3,
                "max_input_chars": 8000,
            }
        }

        result = _decompose_with_predict(
            "Title",
            "Content",
            "Related",
            decomposer_predict=mock_predict,
            cfg=cfg,
        )
        assert result == "garbage"  # last attempt's output returned
        assert mock_predict.call_count == 3


# ============================================================================
# Decomposition: _decompose_with_rlm
# ============================================================================


class TestDecomposeWithRlm:
    def test_mock_rlm_call(self):
        mock_rlm = MagicMock()
        mock_result = MagicMock()
        mock_result.decomposed = "Title: Test\nBody:\nContent"
        mock_result.trajectory = [
            {"code": 'llm_query("generate title")', "output": "Test"},
            {"code": 'print("done")', "output": "done"},
        ]
        mock_rlm.return_value = mock_result
        cfg = {"pipeline": {"max_input_chars": 8000}}

        raw, trajectory = _decompose_with_rlm(
            "Test",
            "Content",
            "Related",
            decomposer_rlm=mock_rlm,
            cfg=cfg,
        )
        assert raw == "Title: Test\nBody:\nContent"
        assert len(trajectory) == 2


# ============================================================================
# Decomposition: _summarize_trajectory
# ============================================================================


class TestSummarizeTrajectory:
    def test_iteration_count(self):
        trajectory = [
            {"code": "x = 1"},
            {"code": "y = 2"},
            {"code": "z = 3"},
        ]
        iters, subs = _summarize_trajectory(trajectory)
        assert iters == 3
        assert subs == 0

    def test_llm_query_counting(self):
        trajectory = [
            {"code": 'result = llm_query("generate title")'},
            {
                "code": 'titles = llm_query_batched(["a", "b"])\nmore = llm_query("another")'
            },
        ]
        iters, subs = _summarize_trajectory(trajectory)
        assert iters == 2
        assert subs == 3  # 1 + 1 + 1


# ============================================================================
# Decomposition: decompose_note
# ============================================================================


class TestDecomposeNote:
    def _make_data(
        self, content="Some note content that is long enough to be meaningful."
    ):
        return {
            "content": content,
            "classification": {
                "para_bucket": "Resources",
                "domain": "AI/ML",
                "subdomain": "DSPy",
                "tags": ["dspy"],
            },
        }

    def test_rlm_success_returns_4_tuple_with_rlm_method(self):
        """decompose_note returns a 4-tuple (atoms, rlm_iters, rlm_subs, method)."""
        valid_output = (
            "Title: A Proper Atomic Note Title Here\n"
            "Tags: tag1, tag2\n"
            "Links: Related Note\n"
            "Body:\n"
            "This is real content with enough characters to pass validation. "
            "Multiple sentences and meaningful information."
        )
        mock_rlm = MagicMock()
        mock_result = MagicMock()
        mock_result.decomposed = valid_output
        mock_result.trajectory = [{"code": 'llm_query("x")', "output": "y"}]
        mock_rlm.return_value = mock_result
        cfg = {"pipeline": {"max_input_chars": 8000}}

        result = decompose_note(
            "Test Note",
            self._make_data(),
            ["Related Note"],
            use_rlm=True,
            decomposer_rlm=mock_rlm,
            decomposer_predict=MagicMock(),
            cfg=cfg,
        )
        assert len(result) == 4
        atoms, rlm_iters, rlm_subs, method = result
        assert method == "rlm"
        assert len(atoms) >= 1
        assert rlm_iters == 1
        assert rlm_subs == 1

    def test_rlm_fails_predict_succeeds(self):
        valid_output = (
            "Title: A Proper Atomic Note Title Here\n"
            "Tags: tag1, tag2\n"
            "Links: Related Note\n"
            "Body:\n"
            "This is real content with enough characters to pass validation. "
            "Multiple sentences present."
        )
        mock_predict = MagicMock()
        mock_predict.return_value = MagicMock(decomposed=valid_output)
        cfg = {
            "pipeline": {
                "max_retries": 1,
                "retry_temp_start": 0.1,
                "retry_temp_step": 0.3,
                "max_input_chars": 8000,
            }
        }

        atoms, rlm_iters, rlm_subs, method = decompose_note(
            "Test",
            self._make_data(),
            ["Related"],
            use_rlm=True,
            decomposer_rlm=MagicMock(side_effect=RuntimeError("RLM crash")),
            decomposer_predict=mock_predict,
            cfg=cfg,
        )
        assert method == "predict"
        assert len(atoms) >= 1

    def test_both_fail_passthrough(self):
        mock_predict = MagicMock()
        mock_predict.return_value = MagicMock(decomposed="garbage")
        cfg = {
            "pipeline": {
                "max_retries": 1,
                "retry_temp_start": 0.1,
                "retry_temp_step": 0.3,
                "max_input_chars": 8000,
            }
        }

        atoms, rlm_iters, rlm_subs, method = decompose_note(
            "Test Note",
            self._make_data(),
            [],
            use_rlm=True,
            decomposer_rlm=MagicMock(side_effect=RuntimeError("fail")),
            decomposer_predict=mock_predict,
            cfg=cfg,
        )
        assert method == "passthrough"
        assert len(atoms) == 1
        # Passthrough preserves original content (sanitized)
        assert atoms[0]["source_note"] == "Test Note"

    def test_rlm_exception_graceful_fallback(self):
        """RLM raising an unexpected exception should not crash -- falls through to Predict."""
        valid_output = (
            "Title: A Proper Atomic Note Title Here\n"
            "Tags: tag1\n"
            "Body:\n"
            "This is real content with enough characters for validation purposes. "
            "It has multiple sentences."
        )
        mock_predict = MagicMock()
        mock_predict.return_value = MagicMock(decomposed=valid_output)
        cfg = {
            "pipeline": {
                "max_retries": 1,
                "retry_temp_start": 0.1,
                "retry_temp_step": 0.3,
                "max_input_chars": 8000,
            }
        }

        atoms, _, _, method = decompose_note(
            "Test",
            self._make_data(),
            [],
            use_rlm=True,
            decomposer_rlm=MagicMock(side_effect=TypeError("unexpected")),
            decomposer_predict=mock_predict,
            cfg=cfg,
        )
        # Should NOT raise -- should fall through to predict
        assert method == "predict"

    def test_predict_only_mode(self):
        """When use_rlm is False, goes straight to Predict."""
        valid_output = (
            "Title: A Proper Atomic Note Title Here\n"
            "Tags: tag1\n"
            "Body:\n"
            "This is real content with enough characters for validation purposes."
            " Multiple sentences."
        )
        mock_rlm = MagicMock()  # Should NOT be called
        mock_predict = MagicMock()
        mock_predict.return_value = MagicMock(decomposed=valid_output)
        cfg = {
            "pipeline": {
                "max_retries": 1,
                "retry_temp_start": 0.1,
                "retry_temp_step": 0.3,
                "max_input_chars": 8000,
            }
        }

        atoms, rlm_iters, rlm_subs, method = decompose_note(
            "Test",
            self._make_data(),
            [],
            use_rlm=False,
            decomposer_rlm=mock_rlm,
            decomposer_predict=mock_predict,
            cfg=cfg,
        )
        assert method == "predict"
        mock_rlm.assert_not_called()


# ============================================================================
# Decomposition: decompose_and_link
# ============================================================================


class TestDecomposeAndLink:
    def test_progressive_processing_skips_existing(self, tmp_path, monkeypatch):
        """Notes whose source_note matches existing atoms are skipped."""
        import zettelvault.decompose as decompose_mod

        # Mock out file I/O for caches
        monkeypatch.setattr(decompose_mod, "ATOMIC_CACHE", tmp_path / "atomic.json")
        monkeypatch.setattr(decompose_mod, "FALLBACK_LOG", tmp_path / "fallback.json")

        valid_output = (
            "Title: New Decomposed Note From Title\n"
            "Tags: tag1\n"
            "Body:\n"
            "This is real content with enough characters for validation purposes. "
            "Multiple sentences."
        )

        mock_predict = MagicMock()
        mock_predict.return_value = MagicMock(decomposed=valid_output)
        cfg = {
            "pipeline": {
                "max_retries": 1,
                "retry_temp_start": 0.1,
                "retry_temp_step": 0.3,
                "max_input_chars": 8000,
                "concept_min_word_len": 4,
                "related_top_n": 20,
            }
        }

        classified = {
            "Already Done": {
                "content": "Old content",
                "classification": {
                    "para_bucket": "Resources",
                    "domain": "Other",
                    "subdomain": "",
                    "tags": [],
                },
            },
            "New Note": {
                "content": "New content for decomposition",
                "classification": {
                    "para_bucket": "Resources",
                    "domain": "Other",
                    "subdomain": "",
                    "tags": [],
                },
            },
        }
        existing = [
            {
                "source_note": "Already Done",
                "title": "Already Done Atom",
                "content": "old",
            }
        ]

        result = decompose_and_link(
            classified,
            use_rlm=False,
            decomposer_rlm=None,
            decomposer_predict=mock_predict,
            cfg=cfg,
            existing_atoms=existing,
        )
        # Should include existing atom + new atoms
        sources = {a.get("source_note") for a in result}
        assert "Already Done" in sources  # carried forward
        # mock_predict should only be called for "New Note"
        assert mock_predict.call_count >= 1

    def test_checkpoint_file_written(self, tmp_path, monkeypatch):
        import zettelvault.decompose as decompose_mod

        atomic_cache = tmp_path / "atomic.json"
        fallback_log = tmp_path / "fallback.json"
        monkeypatch.setattr(decompose_mod, "ATOMIC_CACHE", atomic_cache)
        monkeypatch.setattr(decompose_mod, "FALLBACK_LOG", fallback_log)

        valid_output = (
            "Title: Checkpoint Test Note Title\n"
            "Tags: tag1\n"
            "Body:\n"
            "This is real content with enough characters for validation purposes. "
            "Multiple sentences."
        )

        mock_predict = MagicMock()
        mock_predict.return_value = MagicMock(decomposed=valid_output)
        cfg = {
            "pipeline": {
                "max_retries": 1,
                "retry_temp_start": 0.1,
                "retry_temp_step": 0.3,
                "max_input_chars": 8000,
                "concept_min_word_len": 4,
                "related_top_n": 20,
            }
        }

        classified = {
            "Test Note": {
                "content": "Content here",
                "classification": {
                    "para_bucket": "Resources",
                    "domain": "Other",
                    "subdomain": "",
                    "tags": [],
                },
            },
        }

        decompose_and_link(
            classified,
            use_rlm=False,
            decomposer_rlm=None,
            decomposer_predict=mock_predict,
            cfg=cfg,
        )
        # Checkpoint file should have been written
        assert atomic_cache.exists()
        data = json.loads(atomic_cache.read_text())
        assert len(data) >= 1


# ============================================================================
# Writing: _build_content
# ============================================================================


class TestBuildContent:
    def _make_note(self, **overrides):
        base = {
            "title": "Test Note",
            "content": "Some atomic idea.",
            "tags": ["test", "zettel"],
            "domain": "Engineering",
            "subdomain": "Python",
            "source_note": "Original Note",
            "links": ["[[Other Note]]"],
        }
        base.update(overrides)
        return base

    def test_starts_with_frontmatter(self):
        content = _build_content(self._make_note())
        assert content.startswith("---\n")
        assert "type: zettel" in content

    def test_has_domain_and_subdomain(self):
        content = _build_content(
            self._make_note(domain="Investing", subdomain="Backtesting")
        )
        assert "domain: Investing" in content
        assert "subdomain: Backtesting" in content

    def test_tags_in_yaml(self):
        content = _build_content(self._make_note(tags=["alpha", "beta"]))
        assert "  - alpha\n" in content
        assert "  - beta\n" in content

    def test_empty_tags(self):
        content = _build_content(self._make_note(tags=[]))
        assert "  []\n" in content

    def test_title_as_heading(self):
        assert "# My Atomic Note" in _build_content(
            self._make_note(title="My Atomic Note")
        )

    def test_related_section_present(self):
        content = _build_content(self._make_note(links=["[[Note A]]", "[[Note B]]"]))
        assert "## Related" in content
        assert "[[Note A]]" in content
        assert "[[Note B]]" in content

    def test_no_related_section_when_no_links(self):
        assert "## Related" not in _build_content(self._make_note(links=[]))

    def test_original_frontmatter_merged(self):
        note = self._make_note(
            tags=["generated-tag"],
            domain="AI/ML",
            subdomain="DSPy",
            original_frontmatter={
                "aliases": ["Portfolio Opt"],
                "cssclass": "wide",
                "publish": True,
            },
        )
        content = _build_content(note)
        assert "aliases:" in content
        assert "Portfolio Opt" in content
        assert "cssclass: wide" in content
        assert "publish: true" in content

    def test_original_frontmatter_conflicts_overridden(self):
        """Generated keys (tags, domain, subdomain, source, type) always win."""
        note = self._make_note(
            tags=["generated-tag"],
            original_frontmatter={
                "tags": ["should-be-overridden"],
                "type": "should-be-overridden",
                "aliases": ["kept"],
            },
        )
        content = _build_content(note)
        assert "  - generated-tag" in content
        assert "type: zettel" in content
        assert content.count("type:") == 1
        assert "should-be-overridden" not in content
        assert "aliases:" in content


# ============================================================================
# Writing: write_note
# ============================================================================


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


class TestWriteNote:
    def test_creates_file(self, tmp_path):
        note = _make_note()
        written = write_note(tmp_path, note)
        assert written.exists()
        assert written.name == "Test Note.md"
        assert "# Test Note" in written.read_text()

    def test_correct_bucket_path(self, tmp_path):
        note = _make_note(para_bucket="Projects", subdomain="Hextropian")
        write_note(tmp_path, note)
        expected = tmp_path / "1. Projects" / "Hextropian" / "Test Note.md"
        assert expected.exists()

    def test_archive_bucket(self, tmp_path):
        note = _make_note(para_bucket="Archive", subdomain="")
        write_note(tmp_path, note)
        expected = tmp_path / "4. Archive" / "Test Note.md"
        assert expected.exists()

    def test_no_subdomain(self, tmp_path):
        note = _make_note(subdomain="")
        write_note(tmp_path, note)
        expected = tmp_path / "3. Resources" / "Test Note.md"
        assert expected.exists()

    def test_collision_handling(self, tmp_path):
        note = _make_note(subdomain="")
        write_note(tmp_path, note)
        write_note(tmp_path, note)
        files = list((tmp_path / "3. Resources").rglob("*.md"))
        assert len(files) == 2
        names = {f.name for f in files}
        assert "Test Note.md" in names
        assert "Test Note_1.md" in names

    def test_triple_collision(self, tmp_path):
        note = _make_note(subdomain="")
        write_note(tmp_path, note)
        write_note(tmp_path, note)
        write_note(tmp_path, note)
        files = list((tmp_path / "3. Resources").rglob("*.md"))
        assert len(files) == 3
        names = {f.name for f in files}
        assert "Test Note_2.md" in names

    def test_unsafe_title_characters(self, tmp_path):
        note = _make_note(title="Note: With/Bad Chars", subdomain="")
        written = write_note(tmp_path, note)
        assert written.exists()
        assert "/" not in written.name
        assert ":" not in written.name


# ============================================================================
# Writing: write_moc
# ============================================================================


class TestWriteMoc:
    def test_creates_one_file_per_domain(self, tmp_path):
        notes = [
            {"title": "Note A", "domain": "AI/ML"},
            {"title": "Note B", "domain": "AI/ML"},
            {"title": "Note C", "domain": "Investing"},
        ]
        write_moc(tmp_path, notes)
        assert (tmp_path / "MOC" / "AI-ML.md").exists()
        assert (tmp_path / "MOC" / "Investing.md").exists()

    def test_contains_wikilinks(self, tmp_path):
        notes = [{"title": "Backtesting", "domain": "Investing"}]
        write_moc(tmp_path, notes)
        content = (tmp_path / "MOC" / "Investing.md").read_text()
        assert "[[Backtesting]]" in content

    def test_has_moc_frontmatter(self, tmp_path):
        notes = [{"title": "Note", "domain": "Engineering"}]
        write_moc(tmp_path, notes)
        content = (tmp_path / "MOC" / "Engineering.md").read_text()
        assert "type: moc" in content
        assert "  - moc" in content

    def test_deduplicates_titles(self, tmp_path):
        notes = [
            {"title": "Same Note", "domain": "AI/ML"},
            {"title": "Same Note", "domain": "AI/ML"},
        ]
        write_moc(tmp_path, notes)
        content = (tmp_path / "MOC" / "AI-ML.md").read_text()
        assert content.count("[[Same Note]]") == 1

    def test_domain_with_unsafe_chars(self, tmp_path):
        notes = [{"title": "Note", "domain": "AI/ML"}]
        write_moc(tmp_path, notes)
        assert (tmp_path / "MOC" / "AI-ML.md").exists()


# ============================================================================
# Link Resolution: resolve_links -- CRITICAL, previously had ZERO tests
# ============================================================================


class TestResolveLinks:
    def _create_note(self, dest: Path, folder: str, filename: str, content: str):
        """Helper: create a .md note file in the destination vault."""
        path = dest / folder / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def test_case_insensitive_match(self, tmp_path):
        """[[note a]] should resolve to Note A.md via case-insensitive matching."""
        self._create_note(
            tmp_path,
            "3. Resources",
            "Note A.md",
            "---\ntype: zettel\n---\n\n# Note A\n\nSome content.",
        )
        ref_path = self._create_note(
            tmp_path,
            "3. Resources",
            "Referrer.md",
            "---\ntype: zettel\n---\n\n# Referrer\n\nSee [[note a]] for details.",
        )

        resolve_links(tmp_path)

        updated = ref_path.read_text()
        assert "[[Note A]]" in updated
        assert "[[note a]]" not in updated

    def test_fuzzy_match_above_threshold(self, tmp_path):
        """[[Modren Portfolio Theroy]] should fuzzy-match to Modern Portfolio Theory.md."""
        self._create_note(
            tmp_path,
            "3. Resources",
            "Modern Portfolio Theory.md",
            "---\ntype: zettel\n---\n\n# Modern Portfolio Theory\n\nContent.",
        )
        ref_path = self._create_note(
            tmp_path,
            "3. Resources",
            "Referrer.md",
            "---\ntype: zettel\n---\n\n# Referrer\n\nSee [[Modren Portfolio Theroy]].",
        )

        resolve_links(tmp_path)

        updated = ref_path.read_text()
        assert "[[Modern Portfolio Theory]]" in updated

    def test_fuzzy_match_below_threshold_no_match(self, tmp_path):
        """A very dissimilar link should not match anything."""
        self._create_note(
            tmp_path,
            "3. Resources",
            "Quantum Computing.md",
            "---\ntype: zettel\n---\n\n# Quantum Computing\n\nContent.",
        )
        ref_path = self._create_note(
            tmp_path,
            "3. Resources",
            "Referrer.md",
            "---\ntype: zettel\n---\n\n# Referrer\n\nSee [[Banana Pudding Recipe]].",
        )

        resolve_links(tmp_path)

        updated = ref_path.read_text()
        # Should have been converted to plain text (dead link removal, < 3 refs)
        assert "[[Banana Pudding Recipe]]" not in updated
        assert "Banana Pudding Recipe" in updated

    def test_stub_creation_when_orphan_has_three_or_more_references(self, tmp_path):
        """An orphan link referenced by 3+ files should create a stub note."""
        for i in range(3):
            self._create_note(
                tmp_path,
                "3. Resources",
                f"Referrer{i}.md",
                f"---\ntype: zettel\n---\n\n# Referrer{i}\n\nSee [[Missing Concept]].",
            )

        resolve_links(tmp_path)

        # A stub note should have been created
        stub_files = list(tmp_path.rglob("Missing Concept.md"))
        assert len(stub_files) == 1
        stub_content = stub_files[0].read_text()
        assert "type: stub" in stub_content
        assert "## Referenced by" in stub_content

    def test_dead_link_removal_with_few_references(self, tmp_path):
        """Orphan links with 1-2 references should be converted to plain text."""
        ref_path = self._create_note(
            tmp_path,
            "3. Resources",
            "Referrer.md",
            "---\ntype: zettel\n---\n\n# Referrer\n\nSee [[Nonexistent Note]] here.",
        )

        resolve_links(tmp_path)

        updated = ref_path.read_text()
        assert "[[Nonexistent Note]]" not in updated
        assert "Nonexistent Note" in updated

    def test_no_orphans_no_modifications(self, tmp_path):
        """When all links resolve, no files should be modified."""
        self._create_note(
            tmp_path,
            "3. Resources",
            "Note A.md",
            "---\ntype: zettel\n---\n\n# Note A\n\nSee [[Note B]].",
        )
        self._create_note(
            tmp_path,
            "3. Resources",
            "Note B.md",
            "---\ntype: zettel\n---\n\n# Note B\n\nSee [[Note A]].",
        )

        resolve_links(tmp_path)

        # Both should still have their original links
        a_content = (tmp_path / "3. Resources" / "Note A.md").read_text()
        assert "[[Note B]]" in a_content
        b_content = (tmp_path / "3. Resources" / "Note B.md").read_text()
        assert "[[Note A]]" in b_content

    def test_mixed_resolution(self, tmp_path):
        """Mix of resolved, stub, and dead link handling in the same vault."""
        # Target that exists (with case mismatch)
        self._create_note(
            tmp_path,
            "3. Resources",
            "Real Note.md",
            "---\ntype: zettel\n---\n\n# Real Note\n\nContent.",
        )

        # File with case-insensitive match + dead link
        ref1 = self._create_note(
            tmp_path,
            "3. Resources",
            "Referrer1.md",
            "---\ntype: zettel\n---\n\nSee [[real note]] and [[Dead Link]].",
        )

        # Files referencing a stub candidate (3 refs)
        for i in range(2, 5):
            self._create_note(
                tmp_path,
                "3. Resources",
                f"Referrer{i}.md",
                "---\ntype: zettel\n---\n\nSee [[Stub Candidate]].",
            )

        resolve_links(tmp_path)

        # Case-insensitive match resolved
        r1 = ref1.read_text()
        assert "[[Real Note]]" in r1

        # Dead link removed (1 ref)
        assert "[[Dead Link]]" not in r1
        assert "Dead Link" in r1

        # Stub created for 3-ref orphan
        stub_files = list(tmp_path.rglob("Stub Candidate.md"))
        assert len(stub_files) == 1
        assert "type: stub" in stub_files[0].read_text()

    def test_obsidian_dir_ignored(self, tmp_path):
        """Files inside .obsidian should not be scanned."""
        obsidian = tmp_path / ".obsidian"
        obsidian.mkdir()
        (obsidian / "config.md").write_text("[[Some Link]] in obsidian config.")

        self._create_note(
            tmp_path,
            "3. Resources",
            "Note.md",
            "---\ntype: zettel\n---\n\n# Note\n\nContent.",
        )

        # Should not crash or count .obsidian links
        resolve_links(tmp_path)


# ============================================================================
# Utility Functions: _safe_filename
# ============================================================================


class TestSafeFilename:
    def test_clean_title(self):
        assert _safe_filename("Normal Title") == "Normal Title"

    def test_strips_colon(self):
        assert _safe_filename("Title: Subtitle") == "Title- Subtitle"

    def test_strips_slash(self):
        assert _safe_filename("AI/ML Overview") == "AI-ML Overview"

    def test_strips_backslash(self):
        assert _safe_filename("Path\\Note") == "Path-Note"

    def test_strips_leading_dots(self):
        assert _safe_filename("...hidden") == "hidden"

    def test_empty_input(self):
        assert _safe_filename("") == "Untitled"

    def test_all_unsafe(self):
        assert _safe_filename(":::") == "Untitled"

    def test_question_mark(self):
        result = _safe_filename("What is AI?")
        assert "?" not in result

    def test_angle_brackets(self):
        result = _safe_filename("<tag>content</tag>")
        assert "<" not in result
        assert ">" not in result

    def test_pipe_character(self):
        result = _safe_filename("Option A | Option B")
        assert "|" not in result


# ============================================================================
# Utility Functions: SafeEncoder
# ============================================================================


class TestSafeEncoder:
    def test_datetime_date_serialization(self):
        data = {"date": datetime.date(2024, 6, 15)}
        result = json.dumps(data, cls=SafeEncoder)
        assert '"2024-06-15"' in result

    def test_datetime_datetime_serialization(self):
        data = {"timestamp": datetime.datetime(2024, 6, 15, 10, 30, 0)}
        result = json.dumps(data, cls=SafeEncoder)
        assert '"2024-06-15T10:30:00"' in result

    def test_regular_types_unchanged(self):
        data = {"string": "hello", "number": 42, "list": [1, 2, 3]}
        result = json.dumps(data, cls=SafeEncoder)
        parsed = json.loads(result)
        assert parsed == data

    def test_unsupported_type_raises(self):
        data = {"obj": object()}
        with pytest.raises(TypeError):
            json.dumps(data, cls=SafeEncoder)


# ============================================================================
# Utility Functions: _fmt_duration
# ============================================================================


class TestFmtDuration:
    def test_seconds_only(self):
        assert _fmt_duration(45) == "45s"
        assert _fmt_duration(0) == "0s"

    def test_minutes_and_seconds(self):
        assert _fmt_duration(90) == "1m30s"
        assert _fmt_duration(125) == "2m05s"

    def test_hours_and_minutes(self):
        assert _fmt_duration(3700) == "1h01m"
        assert _fmt_duration(7260) == "2h01m"

    def test_fractional_seconds(self):
        result = _fmt_duration(30.7)
        assert result == "31s"


# ============================================================================
# Utility Functions: _progress_line
# ============================================================================


class TestProgressLine:
    def test_formatting(self):
        import time as _time

        t0 = _time.time() - 10  # 10 seconds ago
        line = _progress_line(5, 10, t0, "classify", "AI/ML Note")
        assert "[classify]" in line
        assert "5" in line
        assert "10" in line
        assert "50.0%" in line
        assert "AI/ML Note" in line

    def test_no_detail(self):
        import time as _time

        t0 = _time.time()
        line = _progress_line(1, 1, t0, "test")
        assert "[test]" in line
        assert "--" not in line

    def test_zero_total(self):
        import time as _time

        t0 = _time.time()
        # Should not divide by zero
        line = _progress_line(0, 0, t0, "test")
        assert "0.0%" in line


# ============================================================================
# Integration Tests (require OPENROUTER_API_KEY)
# ============================================================================


@pytest.mark.integration
class TestClassifyNoteIntegration:
    def test_classify_real_note(self):
        """Classifies a short note using the real LLM."""
        from zettelvault.config import load_config
        from zettelvault.pipeline import Pipeline

        cfg = load_config()
        pipeline = Pipeline(cfg)
        pipeline.init_lm(use_rlm=False)

        result = classify_note(
            title="DSPy",
            content=(
                "DSPy is a framework for programming with language models. "
                "It uses signatures to define typed input/output contracts and "
                "provides optimizers like BootstrapFewShot and MIPROv2."
            ),
            classifier=pipeline.classifier,
            cfg=cfg,
        )
        assert result["para_bucket"] in {"Projects", "Areas", "Resources", "Archive"}
        assert result["domain"]
        assert isinstance(result["tags"], list)
        assert len(result["tags"]) > 0

    def test_classify_travel_note(self):
        """Travel content should map to Resources or Areas, with Travel domain."""
        from zettelvault.config import load_config
        from zettelvault.pipeline import Pipeline

        cfg = load_config()
        pipeline = Pipeline(cfg)
        pipeline.init_lm(use_rlm=False)

        result = classify_note(
            title="Bidwell Hotel",
            content=(
                "The Bidwell Hotel in Portland, Oregon. Great location near Powell's Books. "
                "Comfortable rooms, friendly staff. Good base for exploring the city."
            ),
            classifier=pipeline.classifier,
            cfg=cfg,
        )
        assert result["para_bucket"] in {"Resources", "Areas"}
        assert "Travel" in result["domain"] or "travel" in [
            t.lower() for t in result["tags"]
        ]


@pytest.mark.integration
class TestDecomposeNoteIntegration:
    def test_decompose_predict_mode(self):
        """Decompose a note using Predict (no RLM). Returns 4-tuple."""
        from zettelvault.config import load_config
        from zettelvault.pipeline import Pipeline

        cfg = load_config()
        pipeline = Pipeline(cfg)
        pipeline.init_lm(use_rlm=False)

        classified_note = {
            "content": (
                "---\ntags: [ai, investing]\n---\n\n"
                "# AI Portfolio Optimizer\n\n"
                "This project combines portfolio optimization with ML risk management. "
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
        result = decompose_note(
            "AI Portfolio Optimizer",
            classified_note,
            related,
            use_rlm=pipeline.use_rlm,
            decomposer_rlm=pipeline.decomposer_rlm,
            decomposer_predict=pipeline.decomposer_predict,
            cfg=cfg,
        )
        assert len(result) == 4, f"Expected 4-tuple, got {len(result)}-tuple"
        atoms, rlm_iters, rlm_subs, method = result
        assert len(atoms) >= 1
        for atom in atoms:
            assert atom["title"]
            assert atom["content"]
            assert atom["para_bucket"] == "Projects"
            assert atom["source_note"] == "AI Portfolio Optimizer"
        assert rlm_iters == 0  # Predict mode
        assert method in ("predict", "passthrough")


# ============================================================================
# Sample Vault
# ============================================================================

# -- Realistic markdown fixtures for testing ----------------------------------

_PROSE_NOTE = """\
---
title: Deep Work
tags: [productivity, focus]
---

# Deep Work

Deep work is the ability to focus without distraction on a cognitively
demanding task. It is a skill that allows you to quickly master complicated
information and produce better results in less time.

Cal Newport argues that the ability to perform deep work is becoming
increasingly rare at exactly the same time it is becoming increasingly
valuable in our economy. As a consequence, the few who cultivate this
skill will thrive.

The key insight is that depth is more important than breadth when it
comes to knowledge work. Shallow tasks like email, meetings, and social
media consume enormous amounts of time while producing little value.
"""

_BULLET_NOTE = """\
- Buy groceries
- Clean the kitchen
- Fix the leaky faucet
- Organize the garage
- Take out recycling
- Water the plants
- Feed the cat
- Walk the dog
- Do laundry
- Vacuum living room
"""

_HEADING_NOTE = """\
# Project Plan

## Phase 1: Research

## Phase 2: Design

## Phase 3: Implementation

## Phase 4: Testing

## Phase 5: Deployment

Some brief notes on each phase.
"""

_MIXED_NOTE = """\
---
title: Meeting Notes
---

# Weekly Standup

## Updates

- Alice: working on feature X
- Bob: fixing bug Y
- Carol: reviewed PR #42

## Action Items

- Deploy staging by Friday
- Update [[Code Review Process]]
- Review [[Sprint Planning]]
- Schedule retro

## Discussion

We discussed priorities.

```python
def hello():
    print("world")
```

#meeting #standup
"""

_WIKILINK_HEAVY = """\
This note references [[Note A]], [[Note B]], [[Note C]], and [[Note D]].
It also mentions [[Note E]] in passing.
A short note with many cross-references.
"""

_CODE_NOTE = """\
# Python Tips

Some useful patterns:

```python
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
```

Another example:

```javascript
const greet = (name) => `Hello, ${name}!`;
```

Two code blocks above.
"""

_TAG_NOTE = """\
This is a note about #productivity and #focus.
It also touches on #deep-work concepts.
No code blocks here, just plain text with inline tags.
"""

_TINY_NOTE = "Short."

_LARGE_NOTE = "A" * 5000 + "\n\nThis is a very large note with lots of content.\n"

_FRONTMATTER_ONLY = """\
---
title: Empty Body
tags: [test]
---
"""


class TestExtractFeatures:
    """Test feature extraction from markdown content."""

    def test_prose_note_features(self):
        features = extract_features(_PROSE_NOTE)
        assert features["char_count"] == len(_PROSE_NOTE)
        assert features["has_frontmatter"] is True
        assert features["heading_count"] == 1  # "# Deep Work"
        assert features["bullet_count"] == 0
        assert features["wikilink_count"] == 0
        assert features["codeblock_count"] == 0

    def test_bullet_note_features(self):
        features = extract_features(_BULLET_NOTE)
        assert features["has_frontmatter"] is False
        assert features["bullet_count"] == 10
        assert features["heading_count"] == 0

    def test_heading_note_features(self):
        features = extract_features(_HEADING_NOTE)
        assert features["heading_count"] == 6  # 1 h1 + 5 h2
        assert features["bullet_count"] == 0

    def test_mixed_note_features(self):
        features = extract_features(_MIXED_NOTE)
        assert features["has_frontmatter"] is True
        assert features["heading_count"] >= 2
        assert features["bullet_count"] >= 2
        assert (
            features["wikilink_count"] == 2
        )  # Code Review Process, Sprint Planning (inside bullets)
        assert features["codeblock_count"] == 1

    def test_wikilink_counting(self):
        features = extract_features(_WIKILINK_HEAVY)
        assert features["wikilink_count"] == 5

    def test_codeblock_counting(self):
        features = extract_features(_CODE_NOTE)
        assert features["codeblock_count"] == 2

    def test_tag_counting(self):
        features = extract_features(_TAG_NOTE)
        assert features["tag_count"] == 3  # productivity, focus, deep-work

    def test_tags_not_counted_in_code_blocks(self):
        content = "Some text\n```\n#not-a-tag\n```\n#real-tag"
        features = extract_features(content)
        assert features["tag_count"] == 1

    def test_tiny_note(self):
        features = extract_features(_TINY_NOTE)
        assert features["char_count"] == len(_TINY_NOTE)
        assert features["has_frontmatter"] is False

    def test_frontmatter_only_note(self):
        features = extract_features(_FRONTMATTER_ONLY)
        assert features["has_frontmatter"] is True
        assert features["heading_count"] == 0


class TestClassifyStructure:
    """Test structure classification (bullet-heavy, heading-heavy, prose-heavy, mixed)."""

    def _default_cfg(self):
        return {
            "sample": {
                "bullet_heavy_threshold": 0.40,
                "heading_heavy_threshold": 0.15,
                "prose_heavy_threshold": 0.70,
            }
        }

    def test_bullet_heavy(self):
        features = extract_features(_BULLET_NOTE)
        total_lines = len(_BULLET_NOTE.splitlines())
        result = classify_structure(features, total_lines, self._default_cfg())
        assert result == "bullet-heavy"

    def test_heading_heavy(self):
        features = extract_features(_HEADING_NOTE)
        total_lines = len(_HEADING_NOTE.splitlines())
        result = classify_structure(features, total_lines, self._default_cfg())
        assert result == "heading-heavy"

    def test_prose_heavy(self):
        features = extract_features(_PROSE_NOTE)
        total_lines = len(_PROSE_NOTE.splitlines())
        result = classify_structure(features, total_lines, self._default_cfg())
        assert result == "prose-heavy"

    def test_mixed(self):
        features = extract_features(_MIXED_NOTE)
        total_lines = len(_MIXED_NOTE.splitlines())
        result = classify_structure(features, total_lines, self._default_cfg())
        assert result == "mixed"

    def test_zero_lines(self):
        features = {"bullet_count": 0, "heading_count": 0}
        result = classify_structure(features, 0, self._default_cfg())
        assert result == "mixed"

    def test_threshold_from_config(self):
        """Custom config thresholds are honored."""
        cfg = {
            "sample": {
                "bullet_heavy_threshold": 0.01,
                "heading_heavy_threshold": 0.15,
                "prose_heavy_threshold": 0.70,
            }
        }
        # Even 1 bullet in 10 lines => 10% > 1% threshold
        features = {"bullet_count": 1, "heading_count": 0}
        result = classify_structure(features, 10, cfg)
        assert result == "bullet-heavy"


class TestSizeBucketing:
    """Test size quartile computation and assignment."""

    def test_four_notes(self):
        counts = [100, 200, 300, 400]
        buckets = compute_size_buckets(counts)
        assert len(buckets) == 4
        assert "Q1" in buckets
        assert "Q4" in buckets

    def test_many_notes(self):
        counts = list(range(100, 2100, 100))  # 100, 200, ..., 2000
        buckets = compute_size_buckets(counts)
        assert len(buckets) == 4
        # Q1 should contain the smallest notes
        assert buckets["Q1"][0] == 100

    def test_fewer_than_four(self):
        counts = [100, 200]
        buckets = compute_size_buckets(counts)
        assert len(buckets) == 2

    def test_empty(self):
        buckets = compute_size_buckets([])
        assert buckets == {}

    def test_assign_bucket(self):
        buckets = {"Q1": (0, 100), "Q2": (101, 200), "Q3": (201, 300), "Q4": (301, 400)}
        assert assign_size_bucket(50, buckets) == "Q1"
        assert assign_size_bucket(150, buckets) == "Q2"
        assert assign_size_bucket(350, buckets) == "Q4"

    def test_assign_bucket_edge(self):
        """Value exactly at boundary goes to the matching bucket."""
        buckets = {"Q1": (0, 100), "Q2": (101, 200)}
        assert assign_size_bucket(100, buckets) == "Q1"
        assert assign_size_bucket(101, buckets) == "Q2"


class TestGreedySelection:
    """Test the greedy set-cover selection algorithm."""

    def _make_note(
        self,
        title,
        char_count=500,
        structure="mixed",
        has_fm=False,
        has_wl=False,
        has_cb=False,
        has_tags=False,
    ):
        features = {
            "char_count": char_count,
            "has_frontmatter": has_fm,
            "heading_count": 0,
            "bullet_count": 0,
            "wikilink_count": 1 if has_wl else 0,
            "codeblock_count": 1 if has_cb else 0,
            "tag_count": 1 if has_tags else 0,
        }
        size_bucket = "Q2"
        slots = _coverage_slots(features, size_bucket, structure)
        return {
            "title": title,
            "content": "x" * char_count,
            "features": features,
            "structure": structure,
            "size_bucket": size_bucket,
            "slots": slots,
        }

    def test_covers_all_axes(self):
        """Selection should cover different structure types and features."""
        notes = [
            self._make_note("BulletNote", structure="bullet-heavy"),
            self._make_note("HeadingNote", structure="heading-heavy"),
            self._make_note("ProseNote", structure="prose-heavy"),
            self._make_note("MixedNote", structure="mixed"),
            self._make_note("FMNote", has_fm=True),
            self._make_note("WLNote", has_wl=True),
            self._make_note("CBNote", has_cb=True),
            self._make_note("TagNote", has_tags=True),
        ]
        selected = greedy_select(notes, 8, 500)
        assert len(selected) == 8
        structures = {n["structure"] for n in selected}
        assert structures == {"bullet-heavy", "heading-heavy", "prose-heavy", "mixed"}

    def test_determinism(self):
        """Same input always produces the same output."""
        notes = [
            self._make_note("Alpha", char_count=100, structure="prose-heavy"),
            self._make_note("Beta", char_count=200, structure="bullet-heavy"),
            self._make_note("Gamma", char_count=300, structure="heading-heavy"),
            self._make_note("Delta", char_count=400, structure="mixed"),
            self._make_note("Epsilon", char_count=500, has_fm=True),
        ]
        result1 = greedy_select([dict(n, slots=set(n["slots"])) for n in notes], 3, 300)
        result2 = greedy_select([dict(n, slots=set(n["slots"])) for n in notes], 3, 300)
        assert [n["title"] for n in result1] == [n["title"] for n in result2]

    def test_fewer_notes_than_sample_size(self):
        """When vault has fewer notes than sample_size, select all."""
        notes = [
            self._make_note("Only1"),
            self._make_note("Only2"),
        ]
        selected = greedy_select(notes, 10, 500)
        assert len(selected) == 2
        for n in selected:
            assert "all notes selected" in n["selection_reason"]

    def test_selection_reason_present(self):
        """Every selected note should have a selection_reason."""
        notes = [
            self._make_note("A", structure="prose-heavy"),
            self._make_note("B", structure="bullet-heavy"),
            self._make_note("C", structure="heading-heavy"),
        ]
        selected = greedy_select(notes, 3, 500)
        for n in selected:
            assert "selection_reason" in n
            assert n["selection_reason"]

    def test_tie_break_prefers_median(self):
        """When two notes cover the same slots, prefer the one closer to median."""
        notes = [
            self._make_note("Far", char_count=1000, structure="mixed"),
            self._make_note("Close", char_count=500, structure="mixed"),
        ]
        # Both have the same slots; median is 500
        selected = greedy_select(notes, 1, 500)
        assert selected[0]["title"] == "Close"

    def test_tie_break_alphabetical(self):
        """When distance to median is equal, prefer alphabetically first."""
        notes = [
            self._make_note("Zebra", char_count=500, structure="mixed"),
            self._make_note("Alpha", char_count=500, structure="mixed"),
        ]
        selected = greedy_select(notes, 1, 500)
        assert selected[0]["title"] == "Alpha"

    def test_size_diversity_phase(self):
        """After coverage saturates, remaining picks should maximize size diversity."""
        # All same structure/features, different sizes
        notes = [
            self._make_note("Tiny", char_count=10, structure="mixed"),
            self._make_note("Small", char_count=100, structure="mixed"),
            self._make_note("Medium", char_count=500, structure="mixed"),
            self._make_note("Large", char_count=1000, structure="mixed"),
            self._make_note("Huge", char_count=5000, structure="mixed"),
        ]
        # First pick covers 'structure:mixed' and 'size:Q2'
        # After that, all remaining have same slots -> size diversity kicks in
        selected = greedy_select(notes, 3, 500)
        assert len(selected) == 3
        sizes = [n["features"]["char_count"] for n in selected]
        # Should not pick three notes with same char_count
        assert len(set(sizes)) == 3


class TestSampleVault:
    """Test the end-to-end sample_vault function with mocked vault I/O."""

    def _mock_notes(self):
        """Return a dict mapping titles to content for a diverse mock vault."""
        return {
            "Deep Work": _PROSE_NOTE,
            "Shopping List": _BULLET_NOTE,
            "Project Plan": _HEADING_NOTE,
            "Meeting Notes": _MIXED_NOTE,
            "Cross References": _WIKILINK_HEAVY,
            "Python Tips": _CODE_NOTE,
            "Tagged Note": _TAG_NOTE,
            "Tiny": _TINY_NOTE,
            "Large Document": _LARGE_NOTE,
            "Empty Body Note": _FRONTMATTER_ONLY,
            "Extra Prose 1": "A long prose paragraph. " * 20,
            "Extra Prose 2": "Another long paragraph with different content. " * 15,
            "Extra Bullet": "- item 1\n- item 2\n- item 3\n- item 4\n- item 5\n- item 6\n- item 7\n- item 8\n",
        }

    def _default_cfg(self):
        return {
            "sample": {
                "size": 10,
                "bullet_heavy_threshold": 0.40,
                "heading_heavy_threshold": 0.15,
                "prose_heavy_threshold": 0.70,
            }
        }

    @patch("zettelvault.sample.read_note")
    @patch("zettelvault.sample.list_vault_notes")
    def test_output_files_created(self, mock_list, mock_read, tmp_path):
        notes = self._mock_notes()
        mock_list.return_value = sorted(notes.keys())
        mock_read.side_effect = lambda vault, title: notes.get(title, "")

        dest = sample_vault(
            source_vaults=["TestVault"],
            cfg=self._default_cfg(),
            sample_size=5,
            output_dir=str(tmp_path / "_sample"),
        )
        md_files = list(dest.glob("*.md"))
        assert len(md_files) == 5

    @patch("zettelvault.sample.read_note")
    @patch("zettelvault.sample.list_vault_notes")
    def test_manifest_generated(self, mock_list, mock_read, tmp_path):
        notes = self._mock_notes()
        mock_list.return_value = sorted(notes.keys())
        mock_read.side_effect = lambda vault, title: notes.get(title, "")

        dest = sample_vault(
            source_vaults=["TestVault"],
            cfg=self._default_cfg(),
            sample_size=5,
            output_dir=str(tmp_path / "_sample"),
        )
        manifest_path = dest / "_sample_manifest.json"
        assert manifest_path.exists()

        manifest = json.loads(manifest_path.read_text())
        assert manifest["source_vaults"] == ["TestVault"]
        assert manifest["total_notes"] == len(notes)
        assert manifest["sample_size"] == 5
        assert len(manifest["notes"]) == 5

        # Each note in manifest has required fields
        for note in manifest["notes"]:
            assert "title" in note
            assert "features" in note
            assert "structure" in note
            assert "size_bucket" in note
            assert "selection_reason" in note

    @patch("zettelvault.sample.read_note")
    @patch("zettelvault.sample.list_vault_notes")
    def test_deterministic(self, mock_list, mock_read, tmp_path):
        """Running twice with same input produces identical results."""
        notes = self._mock_notes()
        mock_list.return_value = sorted(notes.keys())
        mock_read.side_effect = lambda vault, title: notes.get(title, "")

        dest1 = sample_vault(
            source_vaults=["TestVault"],
            cfg=self._default_cfg(),
            sample_size=5,
            output_dir=str(tmp_path / "_sample1"),
        )
        dest2 = sample_vault(
            source_vaults=["TestVault"],
            cfg=self._default_cfg(),
            sample_size=5,
            output_dir=str(tmp_path / "_sample2"),
        )
        files1 = sorted(f.name for f in dest1.glob("*.md"))
        files2 = sorted(f.name for f in dest2.glob("*.md"))
        assert files1 == files2

    @patch("zettelvault.sample.read_note")
    @patch("zettelvault.sample.list_vault_notes")
    def test_fewer_notes_than_sample_warns(
        self, mock_list, mock_read, tmp_path, capsys
    ):
        notes = {"Only Note": "Just one note."}
        mock_list.return_value = sorted(notes.keys())
        mock_read.side_effect = lambda vault, title: notes.get(title, "")

        dest = sample_vault(
            source_vaults=["TinyVault"],
            cfg=self._default_cfg(),
            sample_size=10,
            output_dir=str(tmp_path / "_sample"),
        )
        md_files = list(dest.glob("*.md"))
        assert len(md_files) == 1

    @patch("zettelvault.sample.read_note")
    @patch("zettelvault.sample.list_vault_notes")
    def test_content_preserved(self, mock_list, mock_read, tmp_path):
        """Written .md files contain the original note content."""
        notes = {"Test Note": "Original content here.\n\nSecond paragraph."}
        mock_list.return_value = ["Test Note"]
        mock_read.side_effect = lambda vault, title: notes.get(title, "")

        dest = sample_vault(
            source_vaults=["V"],
            cfg=self._default_cfg(),
            sample_size=10,
            output_dir=str(tmp_path / "_sample"),
        )
        out_file = dest / "Test Note.md"
        assert out_file.exists()
        assert out_file.read_text(encoding="utf-8") == notes["Test Note"]

    @patch("zettelvault.sample.read_note")
    @patch("zettelvault.sample.list_vault_notes")
    def test_vault_dir_uses_first_vault_name(self, mock_list, mock_read, tmp_path):
        notes = {"A": "Content A"}
        mock_list.return_value = ["A"]
        mock_read.side_effect = lambda vault, title: notes.get(title, "")

        dest = sample_vault(
            source_vaults=["My Vault"],
            cfg=self._default_cfg(),
            sample_size=10,
            output_dir=str(tmp_path / "_sample"),
        )
        assert dest.name == "My_Vault"

    @patch("zettelvault.sample.read_note")
    @patch("zettelvault.sample.list_vault_notes")
    def test_empty_vault(self, mock_list, mock_read, tmp_path):
        mock_list.return_value = []

        dest = sample_vault(
            source_vaults=["Empty"],
            cfg=self._default_cfg(),
            output_dir=str(tmp_path / "_sample"),
        )
        # Should not crash, no files written
        assert not list(dest.glob("*.md")) if dest.exists() else True
