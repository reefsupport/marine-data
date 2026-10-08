"""``superseded_by`` (P1.2): the target exists and is itself canonical (no chains, no cycles)."""

from marinedata import Registry


def test_superseded_targets_exist_and_are_canonical() -> None:
    reg = Registry.load()
    by_id = {s.id: s for s in reg.sources}
    for s in reg.sources:
        if s.superseded_by is None:
            continue
        assert s.superseded_by in by_id, (s.id, s.superseded_by)
        assert by_id[s.superseded_by].superseded_by is None, (s.id, "chain", s.superseded_by)
