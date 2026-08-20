"""Per-request multi-tenant tracer routing.

When a request carries team/key vendor credentials in
``standard_callback_dynamic_params``, its spans must export through a
``TracerProvider`` aimed at the ``TenantRoute`` those credentials describe.
``TenantTracerCache`` builds and caches one provider per distinct route, and
otherwise hands back the logger's default tracer. This lets a single logger fan
requests out to many tenants without needing a logger per tenant.
"""

from collections import OrderedDict
from typing import Final

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Tracer

from litellm._logging import verbose_logger
from litellm.integrations.otel.model.config import OpenTelemetryV2Config, TenantRoute
from litellm.integrations.otel.plumbing.providers import (
    build_tracer_provider,
    get_tracer,
)
from litellm.integrations.otel.presets import dynamic_otlp_route
from litellm.types.utils import StandardCallbackDynamicParams

# Exporter kinds that ignore headers — never rewritten with dynamic credentials.
_NON_OTLP_KINDS: Final = ("console", "in_memory", "inmemory", "memory")

# Cap on distinct credential-scoped providers held at once. ``dynamic_params``
# can be populated from request metadata, so an unbounded cache lets a caller
# spawn one ``TracerProvider`` (plus its ``BatchSpanProcessor`` background
# thread) per unique credential set and exhaust the proxy. The LRU bound keeps
# the working set of active tenants resident while flushing and shutting down
# evicted providers so their threads are reclaimed.
_MAX_CACHED_PROVIDERS: Final = 256


def _shutdown_provider(provider: TracerProvider) -> None:
    """Flush + stop an evicted provider's processors (reclaims their threads).

    ``TracerProvider.shutdown`` force-flushes each ``SpanProcessor`` before
    stopping it, so any spans already handed to a ``BatchSpanProcessor`` are
    exported rather than dropped. Best-effort: a shutdown failure must not break
    the request that triggered the eviction.
    """
    try:
        provider.shutdown()
    except Exception as e:  # pragma: no cover - defensive
        verbose_logger.debug("OTel V2: error shutting down evicted provider: %s", e)


class TenantTracerCache:
    """``TracerProvider`` cache keyed by the request's ``TenantRoute``."""

    def __init__(
        self,
        config: OpenTelemetryV2Config,
        callback_name: str | None,
        tracer_name: str,
    ) -> None:
        self._config = config
        self._callback_name = callback_name
        self._tracer_name = tracer_name
        self._providers: OrderedDict[TenantRoute, TracerProvider] = OrderedDict()

    def tracer_for(self, default: Tracer, dynamic_params: StandardCallbackDynamicParams | None) -> Tracer:
        """Return the tracer for this request.

        Use ``default`` unless the request's dynamic credentials require a
        route-scoped tracer, in which case build (or reuse) one. The cache is a
        bounded LRU: the least-recently-used provider is flushed and shut down on
        overflow so its exporter threads don't accumulate.
        """
        route: Final = dynamic_otlp_route(self._callback_name, dynamic_params)
        if route is None:
            return default
        provider = self._providers.get(route)
        if provider is not None:
            self._providers.move_to_end(route)
        else:
            provider = build_tracer_provider(self._config_with_route(route))
            self._providers[route] = provider
            if len(self._providers) > _MAX_CACHED_PROVIDERS:
                _, evicted = self._providers.popitem(last=False)
                _shutdown_provider(evicted)
        return get_tracer(provider, self._tracer_name)

    def _config_with_route(self, route: TenantRoute) -> OpenTelemetryV2Config:
        """Clone the config, stamping ``route`` onto the credential's own exporter.

        ``route`` carries the per-request credentials of ``self._callback_name`` (the
        integration that built this cache), so it applies only to the exporter that
        integration contributed (``spec.owner``). A request that carries one
        tenant's Arize key must never rewrite the headers of a co-configured
        Langfuse or self-hosted collector exporter, which would leak that key to a
        different backend.

        A route with no endpoint leaves the spec's own endpoint in place, so an
        integration whose dynamic params only carry credentials keeps exporting to
        the destination its preset resolved from the environment.
        """
        header_str: Final = ",".join(f"{key}={value}" for key, value in route.headers)
        exporters: Final = [
            (
                spec.model_copy(update={"headers": header_str, "endpoint": route.endpoint or spec.endpoint})
                if spec.owner == self._callback_name and spec.kind.lower() not in _NON_OTLP_KINDS
                else spec
            )
            for spec in self._config.exporters
        ]
        return self._config.model_copy(update={"exporters": exporters})
