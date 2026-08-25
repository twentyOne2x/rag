from __future__ import annotations

import hmac
import os
import re
import time
from dataclasses import dataclass
from threading import Lock
from typing import Any, Mapping

import psycopg


_USER_SCOPE_RE = re.compile(r"^usr_[0-9a-f]{64}$")
_TENANT_SCOPE_RE = re.compile(r"^ten_[0-9a-f]{64}$")
_CACHE_LOCK = Lock()
_CACHE: dict[str, tuple[float, "EntitlementScope"]] = {}


def production_runtime() -> bool:
    return os.getenv("ICMFYI_PRODUCTION") == "1" or os.getenv("ICMFYI_ENV", "").lower() == "production"


def validate_runtime_config() -> None:
    if not production_runtime():
        return
    secret = os.getenv("INTERNAL_SERVICE_SECRET", "")
    if len(secret) < 32:
        raise RuntimeError("INTERNAL_SERVICE_SECRET must be at least 32 characters")
    database_url = os.getenv("DATABASE_URL", "")
    if not database_url.startswith(("postgresql://", "postgresql+psycopg://")):
        raise RuntimeError("DATABASE_URL must use PostgreSQL in production")


def authenticate_gateway(headers: Mapping[str, str]) -> tuple[str, str]:
    expected = os.getenv("INTERNAL_SERVICE_SECRET", "")
    supplied = headers.get("x-icmfyi-internal-secret", "")
    if not expected or not hmac.compare_digest(supplied, expected):
        raise PermissionError("invalid internal service authentication")
    user_id = headers.get("x-icmfyi-user-id", "")
    tenant_id = headers.get("x-icmfyi-tenant-id", "")
    if not _USER_SCOPE_RE.fullmatch(user_id) or not _TENANT_SCOPE_RE.fullmatch(tenant_id):
        raise PermissionError("invalid trusted gateway scope")
    return user_id, tenant_id


def _connection_url() -> str:
    value = os.getenv("DATABASE_URL", "")
    return value.replace("postgresql+psycopg://", "postgresql://", 1)


@dataclass(frozen=True)
class EntitlementScope:
    names: frozenset[str]
    ids: frozenset[str]

    @property
    def empty(self) -> bool:
        return not self.names and not self.ids


def _load_scope(tenant_id: str) -> EntitlementScope:
    query = """
        SELECT sc.external_id, sc.handle, sc.display_name
          FROM tenant_channel_entitlements AS entitlement
          JOIN source_channels AS sc ON sc.id = entitlement.channel_id
         WHERE entitlement.tenant_id = %s
           AND entitlement.status = 'active'
    """
    names: set[str] = set()
    identifiers: set[str] = set()
    with psycopg.connect(_connection_url(), connect_timeout=5) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, (tenant_id,))
            for external_id, handle, display_name in cursor.fetchall():
                for value in (external_id, handle):
                    if value:
                        identifiers.add(str(value))
                for value in (handle, display_name):
                    if value:
                        names.add(str(value))
    return EntitlementScope(frozenset(names), frozenset(identifiers))


def entitlement_scope(tenant_id: str) -> EntitlementScope:
    if not production_runtime():
        return EntitlementScope(frozenset(), frozenset())
    ttl = max(1, int(os.getenv("RAG_ENTITLEMENT_CACHE_SECONDS", "30")))
    now = time.monotonic()
    with _CACHE_LOCK:
        cached = _CACHE.get(tenant_id)
        if cached and cached[0] > now:
            return cached[1]
    scope = _load_scope(tenant_id)
    with _CACHE_LOCK:
        _CACHE[tenant_id] = (now + ttl, scope)
    return scope


def enforce_namespace(requested: str | None) -> str:
    canonical = os.getenv("PINECONE_NAMESPACE", "videos").strip() or "videos"
    candidate = (requested or "").strip()
    if production_runtime() and candidate and candidate != canonical:
        raise PermissionError("namespace is outside the canonical tenant corpus")
    return canonical if production_runtime() else candidate or canonical


def tenant_channel_filter(
    raw: Mapping[str, Any] | None,
    scope: EntitlementScope,
) -> dict[str, list[str]] | None:
    if not production_runtime():
        return dict(raw) if raw else None
    if scope.empty:
        raise PermissionError("tenant has no active channel entitlements")

    requested_names = [str(value) for value in (raw or {}).get("include_names", []) if value]
    requested_ids = [str(value) for value in (raw or {}).get("include_ids", []) if value]
    has_requested_selection = bool(requested_names or requested_ids)
    include_names = sorted(
        scope.names.intersection(requested_names) if has_requested_selection else scope.names
    )
    include_ids = sorted(
        scope.ids.intersection(requested_ids) if has_requested_selection else scope.ids
    )
    if (requested_names or requested_ids) and not include_names and not include_ids:
        raise PermissionError("requested channels are outside tenant entitlements")

    result: dict[str, list[str]] = {}
    if include_names:
        result["include_names"] = include_names
    if include_ids:
        result["include_ids"] = include_ids
    for key, allowed in (("exclude_names", scope.names), ("exclude_ids", scope.ids)):
        values = [str(value) for value in (raw or {}).get(key, []) if str(value) in allowed]
        if values:
            result[key] = values
    return result


def clear_entitlement_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()
