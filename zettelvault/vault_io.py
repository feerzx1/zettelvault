"""Vault I/O operations via the vlt CLI.

Provides subprocess wrappers for reading vault contents and managing
the .obsidian configuration directory.
"""

import json
import shutil
import subprocess
from pathlib import Path


def vlt_run(vault: str, *args: str) -> str:
    """Execute a vlt CLI command against a named vault."""
    cmd = ["vlt", f"vault={vault}", *args]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"vlt [{' '.join(args[:2])}]: {r.stderr.strip()}")
    return r.stdout.strip()


def resolve_vault_path(vault_name: str) -> Path | None:
    """Resolve a vault name to its filesystem path via vlt."""
    try:
        raw = subprocess.run(
            ["vlt", "vaults", "--json"],
            capture_output=True,
            text=True,
        ).stdout
        for v in json.loads(raw):
            if v.get("name") == vault_name:
                return Path(v["path"])
    except Exception:
        pass
    return None


def list_vault_notes(vault: str) -> list[str]:
    """List all markdown note titles in a vault."""
    raw = vlt_run(vault, "files", "--json")
    try:
        entries = json.loads(raw)
    except json.JSONDecodeError:
        entries = [line.strip() for line in raw.splitlines() if line.strip()]

    titles = []
    for entry in entries:
        path = entry if isinstance(entry, str) else entry.get("path", "")
        if path.endswith(".md"):
            titles.append(Path(path).stem)
    return titles


def read_note(vault: str, title: str) -> str:
    """Read a note's content from a vault, returning empty string on error."""
    try:
        return vlt_run(vault, "read", f"file={title}")
    except RuntimeError:
        return ""


def copy_obsidian_config(source_vault: str, dest_path: Path):
    """Copy .obsidian directory from source vault to destination."""
    src_path = resolve_vault_path(source_vault)
    if not src_path:
        print(
            f"      Could not resolve path for vault '{source_vault}', skipping .obsidian copy"
        )
        return

    obsidian_dir = src_path / ".obsidian"
    if not obsidian_dir.is_dir():
        print("      No .obsidian directory in source vault")
        return

    dest_obsidian = dest_path / ".obsidian"
    if dest_obsidian.exists():
        print("      .obsidian already exists in destination, skipping copy")
        return

    shutil.copytree(obsidian_dir, dest_obsidian)
    print(f"      Copied .obsidian ({sum(1 for _ in dest_obsidian.rglob('*'))} files)")
