.PHONY: ci lint check test e2e

# `ci` is the exact sequence the GitHub Actions "Test" step runs — kept here so a
# contributor can reproduce a CI failure locally with one command, and so the
# workflow itself has no logic beyond `make ci` (D12, 5star-rubric WP-4).
ci: lint check test e2e

lint:
	uv run ruff check .
	uv run ruff format --check .

# Registry validation (see CONTRIBUTING.md "Adding a source"): every source must
# resolve to a known legal_basis/tier under the research profile. Denied sources
# are expected and do not fail the build — only a crash (bad model, unknown
# schema, etc.) does.
check:
	uv run marinedata check --profile research

test:
	uv run pytest -q

# The D12 e2e fixture (tests/test_e2e_release.py) already runs as part of `test`
# above; it is called out again here, standalone, because the CI sequence in the
# brief names it as its own step and because running it in isolation makes its
# <60s, network-free budget easy to eyeball independent of the full suite.
e2e:
	uv run pytest -q tests/test_e2e_release.py
