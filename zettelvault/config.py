"""Configuration loading and access for ZettelVault.

Loads YAML configuration from config.yaml (defaults), config.local.yaml
(user overrides), and an optional explicit path. Provides dotted-key
access into the merged config tree.
"""

from pathlib import Path

import yaml


_CONFIG_DEFAULT = Path("config.yaml")
_CONFIG_LOCAL = Path("config.local.yaml")


def load_config(path: Path | None = None) -> dict:
    """Load config from YAML, merging defaults with overrides.

    Loads config.yaml first, overlays config.local.yaml if present,
    then overlays the explicit path if given.
    """
    cfg: dict = {}
    for p in [_CONFIG_DEFAULT, _CONFIG_LOCAL]:
        if p.exists():
            with open(p) as f:
                override = yaml.safe_load(f)
                if isinstance(override, dict):
                    deep_merge(cfg, override)
    if path and path.exists():
        with open(path) as f:
            override = yaml.safe_load(f)
            if isinstance(override, dict):
                deep_merge(cfg, override)
    return cfg


def deep_merge(base: dict, override: dict):
    """Merge override into base, recursing into nested dicts."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = value


def config_get(cfg: dict, key: str, default=None):
    """Read a dotted config key like 'model.max_tokens' from a config dict."""
    node = cfg
    for part in key.split("."):
        if isinstance(node, dict):
            node = node.get(part)
        else:
            return default
        if node is None:
            return default
    return node


# ---- Constants shared across modules ----------------------------------------

PARA_FOLDERS = ["1. Projects", "2. Areas", "3. Resources", "4. Archive", "MOC"]

BUCKET_FOLDERS = {
    "Projects": "1. Projects",
    "Areas": "2. Areas",
    "Resources": "3. Resources",
    "Archive": "4. Archive",
}

CLASSIFIED_CACHE = Path("classified_notes.json")
ATOMIC_CACHE = Path("atomic_notes.json")
FALLBACK_LOG = Path("fallback_notes.json")
