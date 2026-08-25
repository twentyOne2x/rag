from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from src.rag_v2.tenancy import (
    EntitlementScope,
    _load_scope,
    authenticate_gateway,
    clear_entitlement_cache,
    entitlement_scope,
    enforce_namespace,
    tenant_channel_filter,
    validate_runtime_config,
)


class TenancyTests(unittest.TestCase):
    def production_env(self):
        return patch.dict(
            os.environ,
            {
                "ICMFYI_PRODUCTION": "1",
                "INTERNAL_SERVICE_SECRET": "s" * 32,
                "DATABASE_URL": "postgresql+psycopg://user:pw@postgres/db",
                "PINECONE_NAMESPACE": "videos",
            },
            clear=True,
        )

    def test_production_runtime_contract_is_fail_closed(self):
        with self.production_env():
            validate_runtime_config()
        with patch.dict(os.environ, {"ICMFYI_PRODUCTION": "1"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "INTERNAL_SERVICE_SECRET"):
                validate_runtime_config()

    def test_gateway_auth_requires_secret_and_derived_scope(self):
        headers = {
            "x-icmfyi-internal-secret": "s" * 32,
            "x-icmfyi-user-id": "usr_" + "a" * 64,
            "x-icmfyi-tenant-id": "ten_" + "b" * 64,
        }
        with self.production_env():
            self.assertEqual(authenticate_gateway(headers), (headers["x-icmfyi-user-id"], headers["x-icmfyi-tenant-id"]))
            with self.assertRaises(PermissionError):
                authenticate_gateway({**headers, "x-icmfyi-internal-secret": "attacker"})
            with self.assertRaises(PermissionError):
                authenticate_gateway({**headers, "x-icmfyi-tenant-id": "tenant-from-caller"})
            with self.assertRaises(PermissionError):
                authenticate_gateway(
                    {
                        **headers,
                        "x-icmfyi-user-id": headers["x-icmfyi-tenant-id"],
                        "x-icmfyi-tenant-id": headers["x-icmfyi-user-id"],
                    }
                )

    def test_namespace_widening_is_rejected(self):
        with self.production_env():
            self.assertEqual(enforce_namespace(None), "videos")
            self.assertEqual(enforce_namespace("videos"), "videos")
            with self.assertRaises(PermissionError):
                enforce_namespace("other-tenant")

    def test_entitlement_filter_defaults_to_allowlist_and_intersects_requests(self):
        scope = EntitlementScope(
            names=frozenset({"Ash Robin", "Megga"}),
            ids=frozenset({"UC-ash", "megga"}),
        )
        with self.production_env():
            self.assertEqual(
                tenant_channel_filter(None, scope),
                {
                    "include_names": ["Ash Robin", "Megga"],
                    "include_ids": ["UC-ash", "megga"],
                },
            )
            self.assertEqual(
                tenant_channel_filter({"include_names": ["Megga", "Attacker"]}, scope),
                {"include_names": ["Megga"]},
            )
            with self.assertRaises(PermissionError):
                tenant_channel_filter({"include_names": ["Attacker"]}, scope)

    def test_entitlement_lookup_sets_transaction_local_rls_tenant_first(self):
        calls = []

        class Cursor:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def execute(self, query, params):
                calls.append((" ".join(query.split()), params))

            def fetchall(self):
                return [("channel-1", "creator", "Creator")]

        class Connection:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def cursor(self):
                return Cursor()

        tenant_id = "ten_" + "b" * 64
        with self.production_env(), patch(
            "src.rag_v2.tenancy.psycopg.connect", return_value=Connection()
        ):
            scope = _load_scope(tenant_id)

        self.assertIn("set_config('app.tenant_id'", calls[0][0])
        self.assertEqual(calls[0][1], (tenant_id,))
        self.assertIn("tenant_channel_entitlements", calls[1][0])
        self.assertEqual(calls[1][1], (tenant_id,))
        self.assertEqual(scope.ids, frozenset({"channel-1", "creator"}))

    def test_production_entitlements_are_immediately_visible_without_cache_invalidation(self):
        tenant_id = "ten_" + "b" * 64
        empty = EntitlementScope(frozenset(), frozenset())
        granted = EntitlementScope(frozenset({"Creator"}), frozenset({"channel-1"}))
        clear_entitlement_cache()
        with self.production_env(), patch(
            "src.rag_v2.tenancy._load_scope",
            side_effect=[empty, granted],
        ) as loader:
            self.assertTrue(entitlement_scope(tenant_id).empty)
            self.assertEqual(entitlement_scope(tenant_id), granted)
        self.assertEqual(loader.call_count, 2)

    def test_empty_entitlement_scope_is_never_cached_when_cache_is_opted_in(self):
        tenant_id = "ten_" + "b" * 64
        empty = EntitlementScope(frozenset(), frozenset())
        granted = EntitlementScope(frozenset({"Creator"}), frozenset({"channel-1"}))
        clear_entitlement_cache()
        with self.production_env(), patch.dict(
            os.environ,
            {"RAG_ENTITLEMENT_CACHE_SECONDS": "30"},
        ), patch(
            "src.rag_v2.tenancy._load_scope",
            side_effect=[empty, granted],
        ) as loader:
            self.assertTrue(entitlement_scope(tenant_id).empty)
            self.assertEqual(entitlement_scope(tenant_id), granted)
        self.assertEqual(loader.call_count, 2)


if __name__ == "__main__":
    unittest.main()
