.PHONY: taxonomy-check

# D-Q scoped label gate (WP-7c): fails only for staged or released sources; `--strict` lists
# the rest. At the integration merge, WP-4's `ci` target must call it (ci: ... taxonomy-check ...).
taxonomy-check:
	uv run marinedata taxonomy check
