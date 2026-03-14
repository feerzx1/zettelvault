.DEFAULT_GOAL := help

PYTHON  := uv run --env-file .env -- python
SOURCE  := Personal3
DEST    := $(HOME)/Library/Mobile\ Documents/iCloud~md~obsidian/Documents/ZettelVault1
LIMIT   := 0
CONFIG  :=

# ── Derived ──────────────────────────────────────────────────────────────────
_LIMIT_FLAG  := $(if $(filter-out 0,$(LIMIT)),--limit $(LIMIT),)
_CONFIG_FLAG := $(if $(CONFIG),--config $(CONFIG),)
_FLAGS       := $(_LIMIT_FLAG) $(_CONFIG_FLAG)

# ── Help ─────────────────────────────────────────────────────────────────────
help:
	@echo "ZettelVault -- Transform Obsidian vaults into PARA + Zettelkasten"
	@echo ""
	@echo "Pipeline targets:"
	@echo "  run             Full pipeline (read -> classify -> decompose -> write)"
	@echo "  dry-run         Classify + decompose, preview only (no file writes)"
	@echo "  resume          Skip classification, reuse classified_notes.json"
	@echo "  resume-all      Skip classify + decompose, reuse atomic_notes.json"
	@echo "  reprocess       Re-run only the notes that fell back to Predict"
	@echo ""
	@echo "Housekeeping:"
	@echo "  status          Show progress of caches and current run"
	@echo "  clean           Remove all caches (classified, atomic, fallback)"
	@echo "  clean-all       Remove caches + destination vault contents"
	@echo "  install         Create venv and install dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  test            Unit tests (no API key needed)"
	@echo "  test-all        Unit + integration tests (needs API key)"
	@echo "  lint            Run ruff linter"
	@echo ""
	@echo "Variables (override on command line):"
	@echo "  SOURCE    Source vault name(s), space-separated  [$(SOURCE)]"
	@echo "  DEST      Destination vault path                 [ZettelVault1 in iCloud]"
	@echo "  LIMIT     Process only first N notes (0=all)     [$(LIMIT)]"
	@echo "  CONFIG    Path to config YAML override           [auto-detect]"
	@echo ""
	@echo "Examples:"
	@echo "  make run SOURCE=\"Personal2 Personal3\""
	@echo "  make dry-run LIMIT=10"
	@echo "  make run CONFIG=config.local.yaml"

# ── Pipeline ─────────────────────────────────────────────────────────────────
run:
	$(PYTHON) zettelvault_dspy.py $(SOURCE) "$(DEST)" $(_FLAGS)

dry-run:
	$(PYTHON) zettelvault_dspy.py $(SOURCE) "$(DEST)" --dry-run $(_FLAGS)

resume:
	$(PYTHON) zettelvault_dspy.py $(SOURCE) "$(DEST)" --skip-classification $(_FLAGS)

resume-all:
	$(PYTHON) zettelvault_dspy.py $(SOURCE) "$(DEST)" --skip-decomposition $(_FLAGS)

reprocess:
	@if [ ! -f fallback_notes.json ]; then \
		echo "No fallback_notes.json found -- nothing to reprocess."; \
		exit 0; \
	fi
	@count=$$($(PYTHON) -c "import json; print(len(json.load(open('fallback_notes.json'))))"); \
	echo "Reprocessing $$count notes that fell back to Predict..."; \
	$(PYTHON) zettelvault_dspy.py $(SOURCE) "$(DEST)" $(_FLAGS)

# ── Status ───────────────────────────────────────────────────────────────────
status:
	@echo "=== ZettelVault migration status ==="
	@if [ -f classified_notes.json ]; then \
		count=$$($(PYTHON) -c "import json; print(len(json.load(open('classified_notes.json'))))"); \
		echo "Classified:  $$count notes (classified_notes.json)"; \
	else \
		echo "Classified:  not started"; \
	fi
	@if [ -f atomic_notes.json ]; then \
		count=$$($(PYTHON) -c "import json; print(len(json.load(open('atomic_notes.json'))))"); \
		sources=$$($(PYTHON) -c "import json; d=json.load(open('atomic_notes.json')); print(len(set(a.get('source_note','') for a in d)))"); \
		echo "Decomposed:  $$count atoms from $$sources source notes (atomic_notes.json)"; \
	else \
		echo "Decomposed:  not started"; \
	fi
	@if [ -f fallback_notes.json ]; then \
		count=$$($(PYTHON) -c "import json; print(len(json.load(open('fallback_notes.json'))))"); \
		echo "Fallbacks:   $$count notes (fallback_notes.json)"; \
	else \
		echo "Fallbacks:   none"; \
	fi
	@if [ -f migration_log.txt ]; then \
		echo "---"; \
		echo "Latest log:"; \
		tail -c 300 migration_log.txt; \
		echo ""; \
	fi

# ── Housekeeping ─────────────────────────────────────────────────────────────
clean:
	rm -f classified_notes.json atomic_notes.json fallback_notes.json migration_log.txt
	@echo "Caches cleared."

clean-all: clean
	@echo "Clearing destination vault contents (preserving .obsidian)..."
	find "$(DEST)" -name '*.md' -not -path '*/.obsidian/*' -delete 2>/dev/null || true
	@echo "Done."

# ── Dev ──────────────────────────────────────────────────────────────────────
install:
	uv sync
	@echo "Environment ready. Copy config.yaml to config.local.yaml and edit as needed."

test:
	uv run --env-file .env -- pytest tests/ -v -m "not integration"

test-all:
	uv run --env-file .env -- pytest tests/ -v

lint:
	uv run -- ruff check zettelvault_dspy.py

.PHONY: help run dry-run resume resume-all reprocess status clean clean-all install test test-all lint
