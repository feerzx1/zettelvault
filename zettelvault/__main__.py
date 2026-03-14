"""CLI entry point for ZettelVault.

Usage:
  python -m zettelvault Personal3 ./ZettelVault1
  python -m zettelvault Personal3 ./ZettelVault1 --no-rlm
  python -m zettelvault Personal3 ./ZettelVault1 --limit 4
  python -m zettelvault Personal3 ./ZettelVault1 --skip-classification
"""

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from .classify import classify_note
from .config import (
    ATOMIC_CACHE,
    CLASSIFIED_CACHE,
    PARA_FOLDERS,
    config_get,
    load_config,
)
from .decompose import decompose_and_link
from .pipeline import Pipeline, _progress_line
from .resolve import resolve_links
from .sample import sample_vault
from .sanitize import SafeEncoder
from .vault_io import copy_obsidian_config, list_vault_notes, read_note
from .writer import write_moc, write_note

from pricing import CostTracker


def main():
    """Run the full ZettelVault pipeline."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Transform an Obsidian vault into PARA + Zettelkasten structure"
    )
    parser.add_argument(
        "source_vault", nargs="+", help="Source vault name(s) (as known to vlt)"
    )
    parser.add_argument(
        "dest_vault",
        nargs="?",
        default=None,
        help="Destination vault path (absolute or ~/...; optional with --sample)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="No file writes; preview only"
    )
    parser.add_argument(
        "--no-rlm",
        action="store_true",
        help="Disable RLM; use dspy.Predict for decomposition",
    )
    parser.add_argument(
        "--skip-classification",
        action="store_true",
        help=f"Load pre-classified notes from {CLASSIFIED_CACHE}",
    )
    parser.add_argument(
        "--skip-decomposition",
        action="store_true",
        help=f"Load atomic notes from {ATOMIC_CACHE} (implies --skip-classification)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        metavar="N",
        help="Process only the first N notes (0 = all)",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Select representative notes from source vault(s) for pipeline preview",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        metavar="N",
        help="Number of notes to sample (default: from config, fallback 10)",
    )
    parser.add_argument(
        "--sample-dir",
        type=str,
        default=None,
        metavar="PATH",
        help="Output directory for sample notes (default: ./_sample)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="FILE",
        help="Path to config YAML (default: config.yaml + config.local.yaml)",
    )
    args = parser.parse_args()

    # argparse nargs="+" followed by nargs="?" is greedy -- the "+" always
    # consumes every positional, leaving dest_vault=None even when a
    # destination was supplied.  Fix: pop the last source_vault element as
    # the dest when dest_vault was not captured.
    if args.dest_vault is None and len(args.source_vault) >= 2:
        args.dest_vault = args.source_vault.pop()

    # -- Sample mode (early exit, no LM needed) --------------------------------
    cfg = load_config(args.config)

    if args.sample:
        sample_vault(
            source_vaults=args.source_vault,
            cfg=cfg,
            sample_size=args.sample_size,
            output_dir=args.sample_dir,
        )
        return

    # -- Normal pipeline: dest_vault is required --------------------------------
    if args.dest_vault is None:
        parser.error("dest_vault is required when not using --sample")

    dest = Path(args.dest_vault).expanduser().resolve()

    # -- Initialize ---------------------------------------------------------------
    pipeline = Pipeline(cfg)

    model_id = config_get(cfg, "model.id", "qwen/qwen3.5-35b-a3b")
    tracker = CostTracker(model_id)
    pipeline.init_lm(use_rlm=not args.no_rlm)

    # -- Step 1: Read -------------------------------------------------------------
    title_vault: dict[str, str] = {}
    for vault_name in args.source_vault:
        print(f"[1/5] Reading '{vault_name}'...")
        for t in list_vault_notes(vault_name):
            if t not in title_vault:
                title_vault[t] = vault_name
    titles = list(title_vault.keys())
    if args.limit:
        titles = titles[: args.limit]
        title_vault = {t: title_vault[t] for t in titles}
        print(f"      {len(titles)} notes (limited to {args.limit})")
    else:
        for v in args.source_vault:
            count = sum(1 for tv in title_vault.values() if tv == v)
            print(f"      {v}: {count} notes")
        print(f"      {len(titles)} total notes")

    # -- Step 2: Classify ---------------------------------------------------------
    classify_checkpoint = config_get(cfg, "pipeline.classify_checkpoint", 50)
    skip_cls = args.skip_classification or args.skip_decomposition

    if skip_cls and CLASSIFIED_CACHE.exists():
        print(f"[2/5] Loading pre-classified notes from {CLASSIFIED_CACHE}")
        classified = json.loads(CLASSIFIED_CACHE.read_text())
        if args.limit:
            classified = dict(list(classified.items())[: args.limit])
    else:
        classified = {}
        if CLASSIFIED_CACHE.exists():
            classified = json.loads(CLASSIFIED_CACHE.read_text())

        new_titles = [t for t in titles if t not in classified]
        cached_titles = [t for t in titles if t in classified]

        if cached_titles:
            print(
                f"[2/5] Classifying notes (PARA + domain)... "
                f"({len(cached_titles)} cached, {len(new_titles)} new)"
            )
        else:
            print("[2/5] Classifying notes (PARA + domain)...")

        if new_titles:
            cls_total = len(new_titles)
            cls_t0 = time.time()
            cls_milestone = max(1, cls_total // 10)
            cls_interval = min(10, cls_milestone)

            with tracker.phase("classification"):
                for cls_i, title in enumerate(new_titles, 1):
                    content = read_note(
                        title_vault.get(title, args.source_vault[0]), title
                    )
                    if not content:
                        continue
                    result = classify_note(
                        title,
                        content,
                        classifier=pipeline.classifier,
                        cfg=cfg,
                    )
                    classified[title] = {"content": content, "classification": result}
                    bucket = result["para_bucket"] or "?"
                    domain = result["domain"] or "?"

                    if cls_i % cls_interval == 0 or cls_i == cls_total:
                        print(
                            _progress_line(
                                cls_i,
                                cls_total,
                                cls_t0,
                                "classify",
                                f"[{bucket}] [{domain}] {title[:35]}",
                            ),
                            flush=True,
                        )

                    if cls_i % classify_checkpoint == 0:
                        CLASSIFIED_CACHE.write_text(
                            json.dumps(classified, indent=2, cls=SafeEncoder)
                        )

            CLASSIFIED_CACHE.write_text(
                json.dumps(classified, indent=2, cls=SafeEncoder)
            )
            print(f"      Saved to {CLASSIFIED_CACHE}", flush=True)
        else:
            print("      All notes already classified")

        classified = {t: classified[t] for t in titles if t in classified}

    # -- Step 3: Decompose + link -------------------------------------------------
    if args.skip_decomposition and ATOMIC_CACHE.exists():
        print(f"[3/5] Loading atomic notes from {ATOMIC_CACHE}")
        atomic = json.loads(ATOMIC_CACHE.read_text())
    else:
        existing_atoms = None
        if ATOMIC_CACHE.exists():
            existing_atoms = json.loads(ATOMIC_CACHE.read_text())

        print("[3/5] Decomposing and cross-linking...")
        with tracker.phase("decomposition") as phase:
            atomic = decompose_and_link(
                classified,
                use_rlm=pipeline.use_rlm,
                decomposer_rlm=pipeline.decomposer_rlm,
                decomposer_predict=pipeline.decomposer_predict,
                cfg=cfg,
                phase_usage=phase,
                existing_atoms=existing_atoms,
                progress_line_fn=_progress_line,
            )

        if not atomic:
            print("ERROR: Decomposition returned no notes.", file=sys.stderr)
            sys.exit(1)

        new_count = len(atomic) - (len(existing_atoms) if existing_atoms else 0)
        print(f"      {len(atomic)} total atomic notes ({new_count} new)")
        ATOMIC_CACHE.write_text(json.dumps(atomic, indent=2, cls=SafeEncoder))
        print(f"      Saved to {ATOMIC_CACHE}")

    # -- Step 4: Write ------------------------------------------------------------
    if args.dry_run:
        print("[DRY RUN] Sample output (first 10 notes):")
        for n in atomic[:10]:
            print(
                f"  [{n.get('para_bucket', '?'):8}] "
                f"[{n.get('domain', '?'):15}] {n.get('title', '?')}"
            )
        domains = {n.get("domain") for n in atomic}
        print(f"  ... {len(atomic)} total notes across {len(domains)} domains")
        tracker.report()
        return

    print(f"[4/5] Writing to {dest}...")
    dest.mkdir(parents=True, exist_ok=True)
    for folder in PARA_FOLDERS:
        (dest / folder).mkdir(exist_ok=True)

    copy_obsidian_config(args.source_vault[0], dest)

    for note in atomic:
        write_note(dest, note)

    write_moc(dest, atomic)

    # -- Step 5: Resolve links ----------------------------------------------------
    print("[5/5] Resolving orphan links...")
    resolve_links(dest, cfg=cfg)

    domains = {n.get("domain") for n in atomic}
    print(f"Done. {len(atomic)} notes + {len(domains)} MOC pages written to {dest}")
    tracker.report()


if __name__ == "__main__":
    main()
