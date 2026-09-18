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
from marinedata.fetch import (
    _API_CLIENTS,
    _IMPLEMENTED_API_CLIENTS,
    FetchError,
    FetchResult,
    _fetch_api,
    auto_fetchable,
)


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


def test_auto_fetchable_true_for_fathomnet(registry: Registry) -> None:
    assert auto_fetchable(registry.source("fathomnet"))


def test_auto_fetchable_false_for_stubbed_api_clients(registry: Registry) -> None:
    stubbed = {
        source.id
        for source in registry
        if source.access.method is AccessMethod.API and not auto_fetchable(source)
    }
    assert stubbed == {
        "obis",
        "allen-coral-atlas",
        "copernicus-globcolour",
        "coralnet",
        "atlantis-synthetic-depth",
    }


def test_auto_fetchable_false_for_missing_or_unknown_client_does_not_raise(
    tmp_path,
) -> None:
    missing = _api_source(None, source_id="no-client-source")
    unknown = _api_source("made-up-client", source_id="unknown-client-source")

    assert auto_fetchable(missing) is False
    assert auto_fetchable(unknown) is False


def test_auto_fetchable_unchanged_for_non_api_methods() -> None:
    http_source = make_source("image-mask-pairs", source_id="http-fixture")
    assert auto_fetchable(http_source)

    gated_source = http_source.model_copy(
        update={"access": http_source.access.model_copy(update={"gated": True, "notes": "x"})}
    )
    assert not auto_fetchable(gated_source)


def test_implemented_api_clients_is_subset_of_api_clients() -> None:
    assert set(_API_CLIENTS) >= _IMPLEMENTED_API_CLIENTS
