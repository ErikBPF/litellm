"""Per-request multi-tenant credential routing (V1 parity)."""

import base64
import os
import sys

sys.path.insert(0, os.path.abspath("../../../.."))

from opentelemetry.trace import NoOpTracer

from litellm.integrations.otel.model.config import (
    ExporterSpec,
    OpenTelemetryV2Config,
    TenantRoute,
)
from litellm.integrations.otel.presets import dynamic_otlp_route
from litellm.integrations.otel.plumbing.routing import TenantTracerCache


def _cache(callback_name, exporters=None):
    cfg = OpenTelemetryV2Config(exporters=exporters or [ExporterSpec(kind="in_memory")])
    return TenantTracerCache(cfg, callback_name, "litellm")


# --- header builders mirror the V1 construct_dynamic_otel_headers overrides --- #


def _headers(callback_name, params):
    route = dynamic_otlp_route(callback_name, params)
    return None if route is None else dict(route.headers)


def test_arize_dynamic_headers():
    assert _headers("arize", {"arize_space_id": "S", "arize_api_key": "K"}) == {
        "arize-space-id": "S",
        "api_key": "K",
    }


def test_arize_space_key_overrides_space_id():
    assert _headers("arize", {"arize_space_id": "S", "arize_space_key": "SK"}) == {
        "arize-space-id": "SK"
    }


def test_langfuse_dynamic_headers_need_both_keys():
    assert dynamic_otlp_route("langfuse_otel", {"langfuse_public_key": "pk"}) is None
    headers = _headers(
        "langfuse_otel", {"langfuse_public_key": "pk", "langfuse_secret_key": "sk"}
    )
    assert headers is not None and "Authorization" in headers


def test_langfuse_dynamic_headers_carry_v4_ingestion_version():
    expected_auth = "Basic " + base64.b64encode(b"pk:sk").decode()
    assert _headers(
        "langfuse_otel", {"langfuse_public_key": "pk", "langfuse_secret_key": "sk"}
    ) == {
        "Authorization": expected_auth,
        "x-langfuse-ingestion-version": "4",
    }


# --- endpoint builders mirror the V1 construct_dynamic_otel_config override --- #


def test_langfuse_dynamic_endpoint_follows_the_key_host():
    """A key that pins its own Langfuse host must move the export destination.

    Regression for the V2 cross-host bug: the preset swapped the key's
    credentials in but left the exporter aimed at the process-wide env host, so
    a tenant's spans were POSTed to the operator's Langfuse signed with keys it
    does not know. V1 fixed this in ``construct_dynamic_otel_config``.
    """
    route = dynamic_otlp_route(
        "langfuse_otel",
        {
            "langfuse_public_key": "pk",
            "langfuse_secret_key": "sk",
            "langfuse_host": "http://team-b-langfuse:3100",
        },
    )
    assert route is not None
    assert route.endpoint == "http://team-b-langfuse:3100/api/public/otel"


def test_langfuse_dynamic_endpoint_normalizes_a_bare_host():
    route = dynamic_otlp_route(
        "langfuse_otel",
        {
            "langfuse_public_key": "pk",
            "langfuse_secret_key": "sk",
            "langfuse_host": "langfuse.internal/",
        },
    )
    assert route is not None
    assert route.endpoint == "https://langfuse.internal/api/public/otel"


def test_langfuse_dynamic_endpoint_is_none_without_a_host():
    # No host on the key means "do not move the destination": the preset's
    # env-resolved endpoint stands, matching V1's fallback to the env host.
    route = dynamic_otlp_route(
        "langfuse_otel", {"langfuse_public_key": "pk", "langfuse_secret_key": "sk"}
    )
    assert route is not None and route.endpoint is None


def test_a_host_without_both_keys_does_not_route_at_all():
    """V1 parity: ``construct_dynamic_otel_config`` returns None unless both keys
    are present, so a key naming a host but only half its credentials keeps the
    default tracer rather than aiming the operator's credentials at that host."""
    assert (
        dynamic_otlp_route(
            "langfuse_otel",
            {"langfuse_public_key": "pk", "langfuse_host": "http://team-b:3100"},
        )
        is None
    )
    assert (
        dynamic_otlp_route("langfuse_otel", {"langfuse_host": "http://team-b:3100"})
        is None
    )


def test_every_endpoint_builder_has_a_headers_builder():
    """An endpoint may only be resolved for a callback that also supplies
    credentials, so a host can never be honored with the operator's keys."""
    from litellm.integrations.otel.presets import (
        DYNAMIC_ENDPOINT_BY_CALLBACK,
        DYNAMIC_HEADERS_BY_CALLBACK,
    )

    assert set(DYNAMIC_ENDPOINT_BY_CALLBACK) <= set(DYNAMIC_HEADERS_BY_CALLBACK)


def test_non_langfuse_callbacks_have_no_dynamic_endpoint():
    # Arize and Weave carry no host in StandardCallbackDynamicParams, so they
    # keep the endpoint their preset resolved from the environment.
    for callback_name, params in (
        ("arize", {"arize_api_key": "K", "arize_space_id": "S"}),
        ("weave_otel", {"wandb_api_key": "w"}),
    ):
        route = dynamic_otlp_route(callback_name, params)
        assert route is not None and route.endpoint is None


def test_weave_dynamic_headers():
    headers = _headers("weave_otel", {"wandb_api_key": "w", "weave_project_id": "p"})
    assert headers is not None
    assert "Authorization" in headers and headers["project_id"] == "p"


def test_non_participating_callbacks_have_no_routing():
    # Phoenix subclasses the base in V1 (no override) → no dynamic routing.
    assert dynamic_otlp_route("arize_phoenix", {"arize_api_key": "K"}) is None
    assert dynamic_otlp_route("langtrace", {"arize_api_key": "K"}) is None
    assert dynamic_otlp_route(None, {"arize_api_key": "K"}) is None


def test_no_dynamic_params_is_no_routing():
    assert dynamic_otlp_route("arize", None) is None
    assert dynamic_otlp_route("arize", {}) is None


# --- TenantTracerCache routes + caches a TracerProvider per credential set --- #


def test_provider_cached_per_credential_set():
    cache = _cache("arize")
    default = NoOpTracer()
    creds_a = {"arize_space_id": "S", "arize_api_key": "K"}
    creds_b = {"arize_space_id": "S2", "arize_api_key": "K2"}

    cache.tracer_for(default, creds_a)
    cache.tracer_for(default, creds_a)  # same set → reuse, no new provider
    assert len(cache._providers) == 1
    cache.tracer_for(default, creds_b)  # new set → new provider
    assert len(cache._providers) == 2


def test_provider_cache_is_bounded_and_evicts_lru(monkeypatch):
    # The cache key derives from request-supplied dynamic credentials, so it
    # must be bounded — an unbounded cache lets a caller spawn one provider (and
    # its background exporter thread) per unique credential set. On overflow the
    # least-recently-used provider is evicted and shut down.
    from litellm.integrations.otel.plumbing import routing as routing_mod

    monkeypatch.setattr(routing_mod, "_MAX_CACHED_PROVIDERS", 2)
    shut_down = []
    monkeypatch.setattr(
        routing_mod, "_shutdown_provider", lambda p: shut_down.append(p)
    )

    cache = _cache("arize")
    default = NoOpTracer()

    def creds(space):
        return {"arize_space_id": space, "arize_api_key": "K"}

    cache.tracer_for(default, creds("1"))
    cache.tracer_for(default, creds("2"))
    cache.tracer_for(default, creds("1"))  # touch "1" → "2" is now LRU
    cache.tracer_for(default, creds("3"))  # overflow → evict "2"

    assert len(cache._providers) == 2
    assert len(shut_down) == 1  # exactly the evicted provider was shut down


def test_no_dynamic_params_uses_default_tracer():
    cache = _cache("arize")
    default = NoOpTracer()
    assert cache.tracer_for(default, {}) is default
    assert cache._providers == {}


def test_non_participating_callback_uses_default_tracer():
    cache = _cache("arize_phoenix")
    default = NoOpTracer()
    assert cache.tracer_for(default, {"arize_api_key": "K"}) is default
    assert cache._providers == {}


def test_dynamic_headers_applied_to_otlp_exporter_only():
    cache = _cache(
        "arize",
        exporters=[
            ExporterSpec(kind="otlp_http", owner="arize"),
            ExporterSpec(kind="in_memory", owner="arize"),
        ],
    )
    route = dynamic_otlp_route("arize", {"arize_space_id": "S", "arize_api_key": "K"})
    new_cfg = cache._config_with_route(route)
    otlp, in_mem = new_cfg.exporters
    # Built from the real producer, so this also pins the header order the
    # exporter receives against an accidental re-sort in the route builder.
    assert otlp.headers == "arize-space-id=S,api_key=K"
    assert in_mem.headers is None  # console/in_memory left untouched


def test_dynamic_headers_do_not_leak_to_other_owners_exporter():
    """A tenant's Arize credentials must never be stamped onto a co-configured
    exporter owned by a different backend (a self-hosted collector, Langfuse).

    Regression for the cross-backend credential leak: ``_config_with_route``
    used to rewrite the headers of every OTLP exporter, so one request carrying
    a team's Arize key clobbered the base collector's and Langfuse's headers
    with that key.
    """
    cache = _cache(
        "arize",
        exporters=[
            ExporterSpec(
                kind="otlp_http",
                endpoint="http://self-hosted-collector:4318",
                headers="x=base-collector",
                owner=None,
            ),
            ExporterSpec(
                kind="otlp_http",
                endpoint="https://cloud.langfuse.com/api/public/otel",
                headers="Authorization=Basic base-langfuse",
                owner="langfuse_otel",
            ),
            ExporterSpec(
                kind="otlp_grpc",
                endpoint="https://otlp.arize.com/v1",
                headers="space_id=base,api_key=base",
                owner="arize",
            ),
        ],
    )
    new_cfg = cache._config_with_route(
        TenantRoute(headers=(("arize-space-id", "TEAMX"), ("api_key", "TEAMX_KEY")))
    )
    by_owner = {e.owner: e.headers for e in new_cfg.exporters}
    assert by_owner["arize"] == "arize-space-id=TEAMX,api_key=TEAMX_KEY"
    assert by_owner[None] == "x=base-collector"
    assert by_owner["langfuse_otel"] == "Authorization=Basic base-langfuse"


# --- the per-request endpoint reaches the exporter, and keys the cache ------ #


def _langfuse_cache():
    return _cache(
        "langfuse_otel",
        exporters=[
            ExporterSpec(
                kind="otlp_http",
                endpoint="http://env-host:3100/api/public/otel",
                headers="Authorization=Basic env",
                owner="langfuse_otel",
            )
        ],
    )


def _exporter_endpoints(provider):
    return [
        proc.span_exporter._endpoint
        for proc in provider._active_span_processor._span_processors
        if hasattr(proc, "span_exporter") and hasattr(proc.span_exporter, "_endpoint")
    ]


def test_key_host_reaches_the_built_otlp_exporter():
    """End to end in-process: a key pinning host B must build an exporter aimed
    at host B, not at the env host the preset resolved at startup."""
    cache = _langfuse_cache()
    cache.tracer_for(
        NoOpTracer(),
        {
            "langfuse_public_key": "pk",
            "langfuse_secret_key": "sk",
            "langfuse_host": "http://team-b-langfuse:3100",
        },
    )
    provider = next(iter(cache._providers.values()))
    assert _exporter_endpoints(provider) == [
        "http://team-b-langfuse:3100/api/public/otel/v1/traces"
    ]


def test_key_without_a_host_keeps_the_env_endpoint():
    cache = _langfuse_cache()
    cache.tracer_for(
        NoOpTracer(), {"langfuse_public_key": "pk", "langfuse_secret_key": "sk"}
    )
    provider = next(iter(cache._providers.values()))
    assert _exporter_endpoints(provider) == [
        "http://env-host:3100/api/public/otel/v1/traces"
    ]


def test_same_credentials_on_different_hosts_do_not_share_a_provider():
    """The cache key must carry the endpoint, not the headers alone.

    Two keys can hold identical Langfuse credentials on different hosts (a
    self-hosted instance and its staging twin). On a headers-only key the second
    one reuses the first one's cached exporter and its spans are delivered to the
    wrong host, which is a silent cross-host misdelivery rather than a 401.
    """
    cache = _langfuse_cache()
    default = NoOpTracer()
    creds = {"langfuse_public_key": "pk", "langfuse_secret_key": "sk"}

    cache.tracer_for(default, {**creds, "langfuse_host": "http://host-a:3100"})
    cache.tracer_for(default, {**creds, "langfuse_host": "http://host-b:3100"})
    assert len(cache._providers) == 2

    endpoints = sorted(
        endpoint
        for provider in cache._providers.values()
        for endpoint in _exporter_endpoints(provider)
    )
    assert endpoints == [
        "http://host-a:3100/api/public/otel/v1/traces",
        "http://host-b:3100/api/public/otel/v1/traces",
    ]

    cache.tracer_for(default, {**creds, "langfuse_host": "http://host-a:3100"})
    assert len(cache._providers) == 2


def test_dynamic_endpoint_does_not_move_another_owners_exporter():
    """A Langfuse key's host must never repoint a co-configured exporter owned by
    a different backend."""
    cache = _cache(
        "langfuse_otel",
        exporters=[
            ExporterSpec(
                kind="otlp_http",
                endpoint="http://self-hosted-collector:4318",
                headers="x=base-collector",
                owner=None,
            ),
            ExporterSpec(
                kind="otlp_grpc",
                endpoint="https://otlp.arize.com/v1",
                headers="space_id=base",
                owner="arize",
            ),
            ExporterSpec(
                kind="otlp_http",
                endpoint="http://env-host:3100/api/public/otel",
                headers="Authorization=Basic env",
                owner="langfuse_otel",
            ),
        ],
    )
    new_cfg = cache._config_with_route(
        TenantRoute(
            headers=(("Authorization", "Basic team-b"),),
            endpoint="http://team-b-langfuse:3100/api/public/otel",
        )
    )
    by_owner = {e.owner: (e.endpoint, e.headers) for e in new_cfg.exporters}
    assert by_owner["langfuse_otel"] == (
        "http://team-b-langfuse:3100/api/public/otel",
        "Authorization=Basic team-b",
    )
    assert by_owner[None] == ("http://self-hosted-collector:4318", "x=base-collector")
    assert by_owner["arize"] == ("https://otlp.arize.com/v1", "space_id=base")
