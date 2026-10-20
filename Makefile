# TrustFin repository — offline checks (Phase-5A, docs/next_phases_roadmap.md §5/§11).
# Run from the repository root:  make test
#
# The interpreter is discovered by trust_fin_extra_phase/tools/paths.py —
# no hard-coded machine paths:
#   $TRUSTFIN_PYTHON  >  ~/trustfin-slm/.venv/bin/python  >  python3
PY := $(shell python3 trust_fin_extra_phase/tools/paths.py --python 2>/dev/null || echo python3)

.PHONY: help test test-verbose hashes verify-hashes verify-phase4 docker-test env

help:
	@echo "TrustFin make targets:"
	@echo "  make test           offline test suite (schemas, frozen hashes, safety regression,"
	@echo "                      redaction, abstention, protected-test access, path hygiene)"
	@echo "  make verify-hashes  check frozen Phase-2.1/3/4 artifacts against recorded SHA-256s"
	@echo "  make hashes         re-record the frozen manifests (only after deliberate changes)"
	@echo "  make verify-phase4  Phase-4 freeze status (10 outputs, FROZEN)"
	@echo "  make docker-test    build + run the CPU-only test container"
	@echo "  make env            resolved interpreter and key pinned versions"
	@echo "interpreter: $(PY)"

test:
	$(PY) -m pytest trust_fin_extra_phase/tests -q

test-verbose:
	$(PY) -m pytest trust_fin_extra_phase/tests -q -rx

hashes:
	$(PY) trust_fin_extra_phase/tools/record_frozen_hashes.py

verify-hashes:
	$(PY) trust_fin_extra_phase/tools/record_frozen_hashes.py --check

verify-phase4:
	python3 trust_fin_extra_phase/tools/phase4_shard.py verify

docker-test:
	docker build -t trustfin-test . && docker run --rm trustfin-test

env:
	@echo "interpreter: $(PY)"
	@$(PY) -V
	@$(PY) -c "import pytest, jsonschema; print('pytest', pytest.__version__, '| jsonschema', jsonschema.__version__)" 2>/dev/null || \
		echo "(pytest not installed in this interpreter — pip install -r trust_fin_extra_phase/requirements-test.txt)"
