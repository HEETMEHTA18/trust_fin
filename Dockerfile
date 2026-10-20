# TrustFin offline test image (Phase-5A, roadmap §11 step 3).
# CPU-only, no model weights, no network access at runtime.
# Build+run:  make docker-test        (or: docker build -t trustfin-test . && docker run --rm trustfin-test)
#
# The image copies only the pure-code test surface (see .dockerignore): the
# frozen artifacts, weights, notebooks and papers stay on the workstation.
# Artifact-dependent tests (frozen hashes, release manifest, protected-test
# freeze manifest) skip themselves gracefully here; run `make test` on the
# workstation for the full 37-test gate.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /repo

COPY trust_fin_extra_phase/requirements-test.txt trust_fin_extra_phase/requirements-test.txt
RUN pip install --no-cache-dir -r trust_fin_extra_phase/requirements-test.txt

# repository-root marker for trust_fin_extra_phase/tools/paths.py discovery
COPY trust_fin_extra_phase/PLAN.md trust_fin_extra_phase/PLAN.md
COPY trust_fin_extra_phase/tools trust_fin_extra_phase/tools
COPY trust_fin_extra_phase/runtime trust_fin_extra_phase/runtime
COPY trust_fin_extra_phase/scripts trust_fin_extra_phase/scripts
COPY trust_fin_extra_phase/tests trust_fin_extra_phase/tests
COPY trust_fin_extra_phase/data/safety trust_fin_extra_phase/data/safety
COPY Makefile Makefile

CMD ["python3", "-m", "pytest", "trust_fin_extra_phase/tests", "-q"]
