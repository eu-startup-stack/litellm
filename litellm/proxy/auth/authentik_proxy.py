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
  email, optional display name, role) for downstream provisioning.

What this module deliberately does NOT do

- accept roles, budgets, models, permissions, or keys from
  headers — those are policy grants, never identity;
- trust ``X-Forwarded-For`` — only the direct TCP peer is
  authoritative for the trust decision.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request, status
from pydantic import BaseModel, ConfigDict

from litellm.proxy._types import LitellmUserRoles
from litellm.proxy.auth.trusted_proxy_utils import require_trusted_proxy_request

_AUTHENTIK_FEATURE_NAME = "Authentik proxy auth"
_AUTHENTIK_GROUP_PREFIX_SETTING = "authentik_group_prefix"
_AUTHENTIK_GROUP_PREFIX_DEFAULT = "litellm-"


# Privilege order, highest first. Unknown prefixed groups are ignored.
_ROLE_BY_SUFFIX: dict[str, LitellmUserRoles] = {
    "proxy_admin": LitellmUserRoles.PROXY_ADMIN,
    "proxy_admin_viewer": LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
    "internal_user": LitellmUserRoles.INTERNAL_USER,
    "internal_user_viewer": LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
}


# Highest privilege first; used by ``resolve_authentik_role`` to pick
# the winning match when several prefixed groups are present.
_ROLE_PRIVILEGE: tuple[LitellmUserRoles, ...] = (
    LitellmUserRoles.PROXY_ADMIN,
    LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
    LitellmUserRoles.INTERNAL_USER,
    LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
)


class AuthentikIdentity(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_id: str
    email: str | None = None
    display_name: str | None = None
    role: LitellmUserRoles | None = None


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
    exercise role resolution or downstream provisioning.
    """
    require_trusted_proxy_request(
        request=request,
        general_settings=general_settings,
        feature_name=_AUTHENTIK_FEATURE_NAME,
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
