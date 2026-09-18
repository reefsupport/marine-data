"""``RegistryError`` coverage for ``Registry._check_references``.

No test exercised this function directly before D3h — the real registry passing
``Registry.load()`` was the only coverage the ``loader.schema_id``/``crosswalk_id``
checks had. This adds a control case (the real registry resolves cleanly) and the
``annotations.schema_id`` check added alongside it.
"""

from __future__ import annotations

import pytest
from conftest import make_source

from marinedata import Registry
from marinedata.enums import AnnotationKind
from marinedata.models import Annotation
from marinedata.registry import RegistryError


def test_registry_loads() -> None:
    """The real registry resolves every loader schema, crosswalk and annotation
    schema reference — the control case for the checks in ``_check_references``."""
    Registry.load()


def test_unknown_annotations_schema_id_raises() -> None:
    """An ``annotations.schema_id`` with no matching registered schema must fail at
    load, the same way an unknown ``loader.schema_id`` already does — not silently,
    the way it did before D3h (see the deolho-21 stub commit)."""
    src = make_source(
        layout="image-folder",
        annotations=(Annotation(kind=AnnotationKind.DENSE_MASK, schema_id="no-such-schema"),),
    )
    with pytest.raises(RegistryError, match=r"fixture.*annotations\.schema_id.*no-such-schema"):
        Registry._check_references({src.id: src}, schemas={}, crosswalks={})
