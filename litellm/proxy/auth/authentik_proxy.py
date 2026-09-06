"""
Authentik proxy identity resolution.

Authentik's Proxy Provider (or any ``forward_auth``-style reverse
proxy in front of it) authenticates the user at the outpost and
injects the result into ``X-authentik-*`` request headers. LiteLLM
trusts those headers only when the direct TCP peer is one of the
operator-configured trusted reverse proxies; the trust check is
reused from ``trusted_proxy_utils.require_trusted_proxy_request``.

What this module does

- parses the pipe-separated group header and keeps only entries
  starting with the configured prefix (default ``litellm-``);
- maps those filtered groups onto a fixed set of
  ``LitellmUserRoles`` values, with the highest-privilege match
  winning when several prefixed groups are present;
- returns a single frozen ``AuthentikIdentity`` (user id, optional
  email, optional display name, role) for downstream provisioning;
- delegates dashboard-session provisioning to the existing
  ``SSOAuthenticationHandler.get_redirect_response_from_openid``
  pipeline; lookup, upsert, key generation, JWT signing, cookies,
  and redirects are NOT reimplemented here.

What this module deliberately does NOT do

- accept roles, budgets, models, permissions, or keys from
  headers — those are policy grants, never identity;
- read ``X-Forwarded-For`` directly. The trust check uses
  ``request.client.host`` (the ASGI ``scope["client"]``), which the
  ASGI server MAY have rewritten from ``X-Forwarded-For`` if
  ``FORWARDED_ALLOW_IPS`` is set. Operators MUST run LiteLLM with
  ``FORWARDED_ALLOW_IPS`` unset or set to the specific CIDR of the
  reverse proxy in front of LiteLLM, and MUST NOT set it to ``*``.
  When ``FORWARDED_ALLOW_IPS='*'`` is detected at config load time,
  ``enforce_authentik_proxy_startup_guards`` fails closed and disables
  the feature.
- reimplement user lookup, upsert, key generation, JWT signing,
  cookies, or redirects.
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import HTTPException, Request, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import LitellmUserRoles
from litellm.proxy.auth.trusted_proxy_utils import require_trusted_proxy_request
from litellm.proxy.management_endpoints.ui_sso import SSOAuthenticationHandler

_AUTHENTIK_FEATURE_NAME = "Authentik proxy auth"
_AUTHENTIK_FEATURE_SETTING = "enable_authentik_proxy_auth"
_AUTHENTIK_GROUP_PREFIX_SETTING = "authentik_group_prefix"
_AUTHENTIK_GROUP_PREFIX_DEFAULT = "litellm-"
_AUTHENTIK_IDENTITY_HEADERS: tuple[str, ...] = (
    "x-authentik-uid",
    "x-authentik-username",
    "x-authentik-groups",
    "x-authentik-email",
    "x-authentik-name",
)


def enforce_authentik_proxy_startup_guards(general_settings: dict[str, Any]) -> None:
    """
    Fail-closed check on the configuration in which the app-level trust
    decision is forgeable.

    Uvicorn's ``ProxyHeadersMiddleware`` rewrites ``scope["client"]``
    from ``X-Forwarded-For`` for any peer listed in
    ``FORWARDED_ALLOW_IPS``. With ``FORWARDED_ALLOW_IPS='*'`` any
    caller can spoof the client IP, and the direct-peer trust check
    becomes equivalent to trusting ``X-Forwarded-For`` from any
    source. We disable the feature rather than start in that state.

    Call once from the config-load path after ``general_settings`` is
    parsed. The dict is mutated in place so subsequent reads see the
    disabled flag.
    """
    if general_settings.get(_AUTHENTIK_FEATURE_SETTING, False) is not True:
        return
    if os.environ.get("FORWARDED_ALLOW_IPS") == "*":
        verbose_proxy_logger.error(
            "Authentik proxy auth is DISABLED because FORWARDED_ALLOW_IPS='*' "
            "would let any caller spoof X-Forwarded-For and bypass the "
            "direct-peer trust check. Set FORWARDED_ALLOW_IPS to the "
            "reverse proxy's CIDR (e.g. '127.0.0.1/32') or unset it, then "
            "re-enable enable_authentik_proxy_auth."
        )
        general_settings[_AUTHENTIK_FEATURE_SETTING] = False


# Privilege order, highest first. Unknown prefixed groups are ignored.
_ROLE_BY_SUFFIX: dict[str, LitellmUserRoles] = {
    "proxy_admin": LitellmUserRoles.PROXY_ADMIN,
    "proxy_admin_viewer": LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
    "internal_user": LitellmUserRoles.INTERNAL_USER,
    "internal_user_viewer": LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
}


# Highest privilege first; used by ``resolve_authentik_role`` to pick
# the winning match when several prefixed groups are present. Built from
# ``_ROLE_BY_SUFFIX.values()`` so adding a new role can't desync the two.
_ROLE_PRIVILEGE: tuple[LitellmUserRoles, ...] = tuple(_ROLE_BY_SUFFIX.values())


class AuthentikIdentity(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_id: str
    email: str | None = None
    display_name: str | None = None
    role: LitellmUserRoles


def parse_authentik_groups(raw: str | None, prefix: str) -> tuple[str, ...]:
    """
    Extract ``prefix``-bearing entries from a pipe-separated groups
    string and return the suffix (prefix-stripped) form.

    Whitespace around each entry is stripped; entries without the
    prefix are dropped so callers never see unrelated group names.
    ``None`` and empty inputs collapse to an empty tuple. The
    stripped form is what ``resolve_authentik_role`` expects.
    """
    if not raw:
        return ()
    return tuple(
        stripped[len(prefix) :]
        for stripped in (group.strip() for group in raw.split("|"))
        if stripped.startswith(prefix) and len(stripped) > len(prefix)
    )


def resolve_authentik_role(
    app_groups: tuple[str, ...],
) -> LitellmUserRoles | None:
    """
    Map ``app_groups`` to the highest-privilege matching
    ``LitellmUserRoles``.

    ``app_groups`` are the prefix-stripped suffixes produced by
    ``parse_authentik_groups``; unknown or absent entries return
    ``None`` so the caller can deny.
    """
    best: LitellmUserRoles | None = None
    for group in app_groups:
        role = _ROLE_BY_SUFFIX.get(group)
        if role is None:
            continue
        if best is None or _ROLE_PRIVILEGE.index(role) < _ROLE_PRIVILEGE.index(best):
            best = role
    return best


def _read_header(request: Request, name: str) -> str | None:
    value = request.headers.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def authentik_identity_from_request(
    request: Request,
    general_settings: dict[str, Any],
) -> AuthentikIdentity:
    """
    Resolve the trusted Authentik identity on a request.

    Order matters: the trust check runs first and short-circuits
    before any header is read, so an untrusted peer can never
    exercise role resolution or downstream provisioning. The trust
    check raises ``ValueError`` to signal a policy decision; we map
    that to ``HTTPException(401)`` so the route returns a clean 401
    rather than letting ``ValueError`` fall through to the generic
    500 in the app-level exception handler (with a full ERROR-level
    traceback per request).
    """
    try:
        require_trusted_proxy_request(
            request=request,
            general_settings=general_settings,
            feature_name=_AUTHENTIK_FEATURE_NAME,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        ) from exc

    for _header_name in _AUTHENTIK_IDENTITY_HEADERS:
        if len(request.headers.getlist(_header_name)) > 1:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=(
                    f"Duplicate {_header_name} header — the trusted proxy "
                    "must replace, not append, Authentik identity headers."
                ),
            )

    user_id = _read_header(request, "X-authentik-uid") or _read_header(request, "X-authentik-username")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Authentik proxy auth requires X-authentik-uid or "
                "X-authentik-username when the request is from a "
                "trusted proxy."
            ),
        )

    email = _read_header(request, "X-authentik-email")
    display_name = _read_header(request, "X-authentik-name")

    prefix_setting = general_settings.get(
        _AUTHENTIK_GROUP_PREFIX_SETTING,
        _AUTHENTIK_GROUP_PREFIX_DEFAULT,
    )
    if isinstance(prefix_setting, str) and prefix_setting:
        prefix = prefix_setting
    else:
        prefix = _AUTHENTIK_GROUP_PREFIX_DEFAULT

    raw_groups = request.headers.get("X-authentik-groups")
    app_groups = parse_authentik_groups(raw_groups, prefix)
    role = resolve_authentik_role(app_groups)
    if role is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Authentik proxy auth requires at least one group "
                f"starting with the configured prefix {prefix!r} that "
                f"maps to a LitellmUserRoles value; got "
                f"{list(app_groups)!r}."
            ),
        )

    return AuthentikIdentity(
        user_id=user_id,
        email=email,
        display_name=display_name,
        role=role,
    )


async def handle_authentik_ui_login(
    request: Request,
    return_to: str | None,
) -> RedirectResponse:
    """Resolve the trusted Authentik identity and hand it to the existing
    SSO dashboard-session pipeline.

    The pipeline (lookup, upsert, key generation, JWT signing, cookies,
    redirects) is intentionally NOT reimplemented here; this module only
    packages the identity into ``CustomOpenID`` and delegates.
    """
    from litellm.proxy.management_endpoints.types import CustomOpenID
    from litellm.proxy.proxy_server import general_settings

    identity = authentik_identity_from_request(
        request=request,
        general_settings=general_settings,
    )

    openid = CustomOpenID(
        id=identity.user_id,
        email=identity.email,
        display_name=identity.display_name,
        provider="authentik",
        team_ids=[],
        user_role=identity.role,
    )

    return await SSOAuthenticationHandler.get_redirect_response_from_openid(
        result=openid,
        request=request,
        ui_access_mode=general_settings.get("ui_access_mode", None),
        return_to=return_to,
    )
