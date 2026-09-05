"""
Tests for Authentik proxy header trust.

Authentik's Proxy Provider injects ``X-authentik-*`` headers after
authenticating the user at the outpost. LiteLLM consumes those headers
to identify dashboard users, but only after the direct TCP peer is one
of the operator-configured trusted reverse proxies
(``trusted_proxy_ranges``). Anything that maps a fixed
``litellm-``-prefixed group to a ``LitellmUserRoles`` value must happen
server-side; the proxy never asserts roles, budgets, or models.

These tests lock in:

- Trust boundary (trusted peer, untrusted peer, absent ranges).
- Identity parsing (uid preferred over username, pipe-split groups,
  unrelated groups ignored).
- Role mapping (four fixed roles, highest-privilege wins, custom
  prefix, unknown prefixed groups).
- Ordering: denial occurs before any provisioning call.
"""

import os
import sys
from collections.abc import Iterable
from unittest.mock import patch

import pytest
from fastapi import HTTPException, Request
from starlette.datastructures import Headers

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.proxy._types import LitellmUserRoles
from litellm.proxy.auth.authentik_proxy import (  # noqa: E402
    AuthentikIdentity,
    authentik_identity_from_request,
    parse_authentik_groups,
    resolve_authentik_role,
)


def _request_with_headers(
    headers: dict[str, str],
    *,
    client_host: str = "127.0.0.1",
) -> Request:
    scope = {
        "type": "http",
        "client": (client_host, 12345),
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
    }
    request = Request(scope=scope)
    request._headers = Headers(headers)
    return request


@pytest.fixture
def proxy_settings(monkeypatch):
    """Yields the auth-relevant general_settings dict for one test.

    Patches the proxy_server module's ``general_settings`` with the
    returned dict so the existing ``trusted_proxy_utils`` lookup
    fallback is exercised too.
    """

    import litellm.proxy.proxy_server as proxy_server

    def _configure(
        *,
        trusted_proxy_ranges: Iterable[str] | None = ("127.0.0.1/32",),
        authentik_group_prefix: str | None = "litellm-",
    ) -> dict[str, object]:
        settings: dict[str, object] = {}
        if trusted_proxy_ranges is not None:
            settings["trusted_proxy_ranges"] = list(trusted_proxy_ranges)
        if authentik_group_prefix is not None:
            settings["authentik_group_prefix"] = authentik_group_prefix
        monkeypatch.setattr(
            proxy_server,
            "general_settings",
            settings,
            raising=False,
        )
        return settings

    return _configure


def test_parse_authentik_groups_splits_pipe_and_filters_by_prefix() -> None:
    groups = parse_authentik_groups(
        "litellm-internal_user|admins|finance",
        "litellm-",
    )
    assert groups == ("internal_user",)


def test_parse_authentik_groups_handles_none_and_empty() -> None:
    assert parse_authentik_groups(None, "litellm-") == ()
    assert parse_authentik_groups("", "litellm-") == ()
    assert parse_authentik_groups("   ", "litellm-") == ()


def test_parse_authentik_groups_drops_unrelated_with_custom_prefix() -> None:
    groups = parse_authentik_groups(
        "litellm-internal_user|cool-team",
        "cool-",
    )
    assert groups == ("team",)


@pytest.mark.parametrize(
    ("group", "expected"),
    [
        ("proxy_admin", LitellmUserRoles.PROXY_ADMIN),
        ("proxy_admin_viewer", LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY),
        ("internal_user", LitellmUserRoles.INTERNAL_USER),
        ("internal_user_viewer", LitellmUserRoles.INTERNAL_USER_VIEW_ONLY),
    ],
)
def test_resolve_authentik_role_maps_each_fixed_group(group: str, expected: LitellmUserRoles) -> None:
    assert resolve_authentik_role((group,)) == expected


def test_resolve_authentik_role_ignores_unrelated_groups() -> None:
    assert resolve_authentik_role(("admins", "finance", "cool-team")) is None


def test_resolve_authentik_role_unknown_prefixed_group_returns_none() -> None:
    assert resolve_authentik_role(("litellm-no-such-role",)) is None


def test_resolve_authentik_role_highest_privilege_wins() -> None:
    # internal_user_viewer + proxy_admin => proxy_admin
    assert resolve_authentik_role(("internal_user_viewer", "proxy_admin")) == LitellmUserRoles.PROXY_ADMIN
    # internal_user + proxy_admin_viewer => proxy_admin_viewer
    assert resolve_authentik_role(("internal_user", "proxy_admin_viewer")) == LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY
    # internal_user_viewer + internal_user => internal_user
    assert resolve_authentik_role(("internal_user_viewer", "internal_user")) == LitellmUserRoles.INTERNAL_USER
    # proxy_admin + unrelated => proxy_admin
    assert resolve_authentik_role(("proxy_admin", "admins")) == LitellmUserRoles.PROXY_ADMIN


def test_trusted_peer_with_valid_headers_returns_identity(proxy_settings) -> None:
    settings = proxy_settings()
    request = _request_with_headers(
        {
            "x-authentik-uid": "uid-abc",
            "x-authentik-username": "alice",
            "x-authentik-email": "alice@example.com",
            "x-authentik-name": "Alice Smith",
            "x-authentik-groups": "litellm-internal_user|admins",
        }
    )

    identity = authentik_identity_from_request(request, settings)

    assert identity == AuthentikIdentity(
        user_id="uid-abc",
        email="alice@example.com",
        display_name="Alice Smith",
        role=LitellmUserRoles.INTERNAL_USER,
    )


def test_untrusted_peer_raises_trust_error_without_reading_identity(proxy_settings) -> None:
    settings = proxy_settings(trusted_proxy_ranges=["10.0.0.0/24"])
    request = _request_with_headers(
        {
            "x-authentik-uid": "uid-abc",
            "x-authentik-groups": "litellm-proxy_admin",
        },
        client_host="203.0.113.10",
    )

    with pytest.raises(ValueError, match="not trusted"):
        authentik_identity_from_request(request, settings)


def test_absent_trusted_proxy_ranges_raises_trust_error(proxy_settings) -> None:
    settings = proxy_settings(trusted_proxy_ranges=None)
    request = _request_with_headers(
        {
            "x-authentik-uid": "uid-abc",
            "x-authentik-groups": "litellm-proxy_admin",
        }
    )

    with pytest.raises(ValueError, match="trusted_proxy_ranges"):
        authentik_identity_from_request(request, settings)


def test_missing_username_and_uid_raises_identity_error(proxy_settings) -> None:
    settings = proxy_settings()
    request = _request_with_headers({"x-authentik-groups": "litellm-internal_user"})

    with pytest.raises(HTTPException) as exc:
        authentik_identity_from_request(request, settings)
    assert exc.value.status_code == 401


def test_uid_preferred_over_username_for_user_id(proxy_settings) -> None:
    settings = proxy_settings()
    request = _request_with_headers(
        {
            "x-authentik-uid": "stable-hash-1234",
            "x-authentik-username": "alice",
            "x-authentik-groups": "litellm-internal_user",
        }
    )

    identity = authentik_identity_from_request(request, settings)

    assert identity.user_id == "stable-hash-1234"


def test_username_falls_back_when_uid_absent(proxy_settings) -> None:
    settings = proxy_settings()
    request = _request_with_headers(
        {
            "x-authentik-username": "alice",
            "x-authentik-groups": "litellm-internal_user",
        }
    )

    identity = authentik_identity_from_request(request, settings)

    assert identity.user_id == "alice"


def test_unknown_prefixed_groups_yield_role_none_and_403(proxy_settings) -> None:
    settings = proxy_settings()
    request = _request_with_headers(
        {
            "x-authentik-uid": "uid-abc",
            "x-authentik-groups": "litellm-no-such-role",
        }
    )

    with pytest.raises(HTTPException) as exc:
        authentik_identity_from_request(request, settings)
    assert exc.value.status_code == 403


def test_unrelated_groups_only_returns_403(proxy_settings) -> None:
    settings = proxy_settings()
    request = _request_with_headers(
        {
            "x-authentik-uid": "uid-abc",
            "x-authentik-groups": "admins|finance",
        }
    )

    with pytest.raises(HTTPException) as exc:
        authentik_identity_from_request(request, settings)
    assert exc.value.status_code == 403


def test_custom_prefix_only_matches_that_prefix(proxy_settings) -> None:
    settings = proxy_settings(authentik_group_prefix="custom-")
    request = _request_with_headers(
        {
            "x-authentik-uid": "uid-abc",
            "x-authentik-groups": "litellm-internal_user|custom-proxy_admin",
        }
    )

    identity = authentik_identity_from_request(request, settings)

    assert identity.role == LitellmUserRoles.PROXY_ADMIN


def test_denial_occurs_before_any_provisioning_call(proxy_settings) -> None:
    """Trust check runs before any other work in the function.

    Records the order in which the trust utility and the role resolver
    are invoked. If the trust check raises, the role resolver must
    never run.
    """
    settings = proxy_settings(trusted_proxy_ranges=["10.0.0.0/24"])
    request = _request_with_headers(
        {
            "x-authentik-uid": "uid-abc",
            "x-authentik-groups": "litellm-proxy_admin",
        },
        client_host="203.0.113.10",
    )

    call_log: list[str] = []

    def record_trust(*_args, **_kwargs):
        call_log.append("trust_check")
        raise ValueError("not trusted")

    def record_resolve(*_args, **_kwargs):
        call_log.append("resolve_role")
        return LitellmUserRoles.PROXY_ADMIN

    with (
        patch(
            "litellm.proxy.auth.authentik_proxy.require_trusted_proxy_request",
            side_effect=record_trust,
        ),
        patch(
            "litellm.proxy.auth.authentik_proxy.resolve_authentik_role",
            side_effect=record_resolve,
        ),
    ):
        with pytest.raises(ValueError, match="not trusted"):
            authentik_identity_from_request(request, settings)

    assert call_log == ["trust_check"]
    assert "resolve_role" not in call_log
