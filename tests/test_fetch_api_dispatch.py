"""``method: api`` dispatch — no network.

Every ``api`` source now declares its own ``access.params.client``; dispatch reads
that key rather than assuming FathomNet. The bug this closes: before this, every
``api`` source — obis, allen-coral-atlas, copernicus-globcolour, and (once re-valued)
coralnet and atlantis-synthetic-depth — was silently routed to ``fetch_fathomnet``
because ``_fetch_api`` mapped the whole ``AccessMethod.API`` value to that one client.
"""

from __future__ import annotations

import pytest
from conftest import make_source

from marinedata import Registry
from marinedata import fetch as fetch_module
from marinedata.enums import AccessMethod
from marinedata.fetch import _API_CLIENTS, FetchError, FetchResult, _fetch_api


def _api_source(client: str | None, source_id: str = "fixture"):
    base = make_source("image-mask-pairs", source_id=source_id)
    params = {} if client is None else {"client": client}
    access = base.access.model_copy(update={"method": AccessMethod.API, "params": params})
    return base.model_copy(update={"access": access})


def test_fetch_api_dispatches_on_client_param(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def _fake(source, root, limit):
        calls.append(source.id)
        return FetchResult(source_id=source.id, root=root, items=1, method="api", truncated=False)

    monkeypatch.setattr(fetch_module, "_API_CLIENTS", {"stub-client": _fake})
    source = _api_source("stub-client", source_id="stub-source")

    result = _fetch_api(source, tmp_path, 10)

    assert calls == ["stub-source"]
    assert result.root == tmp_path


def test_fetch_api_missing_client_raises(tmp_path) -> None:
    source = _api_source(None, source_id="no-client-source")
    with pytest.raises(FetchError, match="no-client-source"):
        _fetch_api(source, tmp_path, 10)


def test_fetch_api_unknown_client_raises(tmp_path) -> None:
    source = _api_source("made-up-client", source_id="unknown-client-source")
    with pytest.raises(FetchError, match="made-up-client"):
        _fetch_api(source, tmp_path, 10)


def test_fetch_api_unimplemented_client_raises_not_implemented(tmp_path) -> None:
    source = _api_source("obis", source_id="obis-fixture")
    with pytest.raises(NotImplementedError, match="obis"):
        _fetch_api(source, tmp_path, 10)


def test_every_api_source_declares_known_client(registry: Registry) -> None:
    for source in registry:
        if source.access.method is not AccessMethod.API:
            continue
        client = source.access.params.get("client")
        assert client is not None, f"{source.id}: no access.params.client declared"
        assert client in _API_CLIENTS, f"{source.id}: unknown client {client!r}"


def test_only_fathomnet_source_uses_fathomnet_client(registry: Registry) -> None:
    fathomnet_sources = [
        source.id
        for source in registry
        if source.access.method is AccessMethod.API
        and source.access.params.get("client") == "fathomnet"
    ]
    assert fathomnet_sources == ["fathomnet"]


def test_coralnet_and_atlantis_are_api_method(registry: Registry) -> None:
    coralnet = registry.source("coralnet")
    atlantis = registry.source("atlantis-synthetic-depth")

    assert coralnet.access.method is AccessMethod.API
    assert coralnet.access.params.get("client") == "coralnet"
    assert atlantis.access.method is AccessMethod.API
    assert atlantis.access.params.get("client") == "atlantis"
