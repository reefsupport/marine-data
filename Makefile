.PHONY: ci lint check taxonomy-check labels-check docs docs-check gates test e2e image image-check

# `ci` is the exact sequence the GitHub Actions "Test" step runs — kept here so a
# contributor can reproduce a CI failure locally with one command, and so the
# workflow itself has no logic beyond `make ci` (D12, 5star-rubric WP-4). `docs-check` and
# `gates` sit before `test` so a failure in the long suite cannot hide them.
ci: lint check taxonomy-check labels-check docs-check gates test e2e

lint:
	uv run ruff check .
	uv run ruff format --check .

# Registry validation (see CONTRIBUTING.md "Adding a source"): every source must
# resolve to a known legal_basis/tier under the research profile. Denied sources
# are expected and do not fail the build — only a crash (bad model, unknown
# schema, etc.) does.
check:
	uv run marinedata check --profile research

# D-Q scoped label gate (WP-7c): fails only for staged or released sources; `--strict` lists
# the rest (registered-but-not-staged sources are reported, never fail `make ci`).
taxonomy-check:
	uv run marinedata taxonomy check

# WP-U2 crosswalk contract: every id in registry/sources AND registry/ingest-specs. Report
# only (non-strict, exit 0): `--strict` or `--release-sources FILE` turn the 0.95 floor into
# a failing gate for those ids.
labels-check:
	uv run marinedata labels check

# docs/sources.md and docs/configs.md are generated from registry/ and hf_export: never edit
# them by hand. `docs-check` renders into a temp dir and fails when the committed files differ.
docs:
	uv run python scripts/gen_sources_docs.py

docs-check:
	@tmp=$$(mktemp -d) && uv run python scripts/gen_sources_docs.py --out $$tmp > /dev/null && \
	  { diff -u docs/sources.md $$tmp/sources.md && diff -u docs/configs.md $$tmp/configs.md; } ; \
	  rc=$$?; rm -rf $$tmp; [ $$rc -eq 0 ] || echo "docs are stale: run 'make docs' and commit"; \
	  exit $$rc

# Release gates: their unit tests, then the masks gate on a committed 2-image build
# (tests/fixtures/masks_gate_ok). The gate writes GATES.json next to the build, so it runs on a copy.
gates:
	uv run pytest -q tests/test_gate.py tests/test_masks_gate.py tests/test_check_report.py tests/test_release_lock.py
	@d=$$(mktemp -d) && cp -R tests/fixtures/masks_gate_ok/. $$d && \
	  uv run marinedata release gate --masks $$d ; rc=$$?; rm -rf $$d; exit $$rc

test:
	uv run pytest -q

# The D12 e2e fixture (tests/test_e2e_release.py) already runs as part of `test`
# above; it is called out again here, standalone, because the CI sequence in the
# brief names it as its own step and because running it in isolation makes its
# <60s, network-free budget easy to eyeball independent of the full suite.
e2e:
	uv run pytest -q tests/test_e2e_release.py

# Release-build image (Dockerfile; deploy/release-vm/ runs it on a temporary VM). Not part of
# `ci`: it needs a Docker daemon. `image-check` fails when the image is over IMAGE_MAX_BYTES.
IMAGE ?= marinedata
IMAGE_TAG ?= $(shell git describe --tags --always --dirty)
IMAGE_MAX_BYTES ?= 1500000000

image:
	docker build -t $(IMAGE):$(IMAGE_TAG) .

image-check:
	@docker image inspect $(IMAGE):$(IMAGE_TAG) > /dev/null 2>&1 || \
	  { echo "image-check: $(IMAGE):$(IMAGE_TAG) not built: run 'make image' first"; exit 1; }
	docker run --rm $(IMAGE):$(IMAGE_TAG) --version
	@docker run --rm $(IMAGE):$(IMAGE_TAG) check --profile research > /dev/null && \
	  echo "image-check: registry check ok (research profile, repo registry, no bucket)"
	@size=$$(docker image inspect --format '{{.Size}}' $(IMAGE):$(IMAGE_TAG)) && \
	  echo "image-check: $$size bytes (cap $(IMAGE_MAX_BYTES))" && [ $$size -le $(IMAGE_MAX_BYTES) ]
