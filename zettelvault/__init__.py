"""ZettelVault -- Transform an Obsidian vault into PARA + Zettelkasten structure.

Pipeline:
  1. Read all notes from source vault via vlt
  2. Classify each note (PARA bucket + domain) via dspy.Predict
  3. Build concept index (Python), then decompose each note via dspy.RLM
     (falls back to dspy.Predict if RLM is unavailable)
  4. Write atomic notes to destination vault (filesystem)
  5. Resolve links -- fix orphan [[wikilinks]] via fuzzy match, stub creation, or removal
"""

from .classify import (
    ClassifyNote,
    build_concept_index,
    classify_note,
    find_related,
)
from .config import (
    ATOMIC_CACHE,
    BUCKET_FOLDERS,
    CLASSIFIED_CACHE,
    FALLBACK_LOG,
    PARA_FOLDERS,
    config_get,
    deep_merge,
    load_config,
)
from .decompose import (
    DecomposeNote,
    decompose_and_link,
    decompose_note,
    is_valid_output,
    parse_atoms,
)
from .pipeline import (
    Pipeline,
    _fmt_duration,  # noqa: F401 - re-exported for tests
    _progress_line,  # noqa: F401 - re-exported for tests
)
from .resolve import resolve_links
from .sample import sample_vault
from .sanitize import (
    SafeEncoder,
    WL_CLOSE,
    WL_OPEN,
    extract_frontmatter,
    restore_wikilinks,
    sanitize_content,
)
from .vault_io import (
    copy_obsidian_config,
    list_vault_notes,
    read_note,
    resolve_vault_path,
    vlt_run,
)
from .writer import (
    _build_content,  # noqa: F401 - re-exported for tests
    _safe_filename,  # noqa: F401 - re-exported for tests
    write_moc,
    write_note,
)

__all__ = [
    # config
    "load_config",
    "deep_merge",
    "config_get",
    "PARA_FOLDERS",
    "BUCKET_FOLDERS",
    "CLASSIFIED_CACHE",
    "ATOMIC_CACHE",
    "FALLBACK_LOG",
    # vault_io
    "vlt_run",
    "resolve_vault_path",
    "list_vault_notes",
    "read_note",
    "copy_obsidian_config",
    # sanitize
    "WL_OPEN",
    "WL_CLOSE",
    "SafeEncoder",
    "sanitize_content",
    "restore_wikilinks",
    "extract_frontmatter",
    # classify
    "ClassifyNote",
    "classify_note",
    "build_concept_index",
    "find_related",
    # decompose
    "DecomposeNote",
    "is_valid_output",
    "parse_atoms",
    "decompose_note",
    "decompose_and_link",
    # writer
    "write_note",
    "write_moc",
    # resolve
    "resolve_links",
    # sample
    "sample_vault",
    # pipeline
    "Pipeline",
]
