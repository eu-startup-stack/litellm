"""
Unit tests for the VERIA-55 fixes:

- Key update may not assign a key to an organization the caller is not a
  member of.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth


# ---------------------------------------------------------------------------
# /key/update — _validate_caller_can_assign_key_org
# ---------------------------------------------------------------------------


def _make_prisma_with_user_orgs(user_id: str, org_ids: list):
    prisma = MagicMock()
    user_row = MagicMock()
    user_row.organization_memberships = [
        MagicMock(organization_id=org_id) for org_id in org_ids
    ]
    prisma.db.litellm_usertable.find_unique = AsyncMock(return_value=user_row)
    return prisma


@pytest.mark.asyncio
async def test_assign_key_org_allows_member():
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        _validate_caller_can_assign_key_org,
    )

    prisma = _make_prisma_with_user_orgs("alice", ["org-1", "org-2"])
    caller = UserAPIKeyAuth(
        user_id="alice",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
    )
    # Should not raise.
    await _validate_caller_can_assign_key_org(
        user_api_key_dict=caller,
        organization_id="org-2",
        prisma_client=prisma,
    )


@pytest.mark.asyncio
async def test_assign_key_org_blocks_non_member():
    """The IDOR: caller asks to point a key at an org they don't belong to."""
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        _validate_caller_can_assign_key_org,
    )

    prisma = _make_prisma_with_user_orgs("alice", ["org-1"])
    caller = UserAPIKeyAuth(
        user_id="alice",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
    )
    with pytest.raises(HTTPException) as exc_info:
        await _validate_caller_can_assign_key_org(
            user_api_key_dict=caller,
            organization_id="someone-elses-org",
            prisma_client=prisma,
        )
    assert exc_info.value.status_code == 403
    assert "someone-elses-org" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_assign_key_org_blocks_caller_without_user_id():
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        _validate_caller_can_assign_key_org,
    )

    prisma = MagicMock()
    caller = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER.value,
    )
    with pytest.raises(HTTPException) as exc_info:
        await _validate_caller_can_assign_key_org(
            user_api_key_dict=caller,
            organization_id="org-1",
            prisma_client=prisma,
        )
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_assign_key_org_blocks_caller_with_no_memberships():
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        _validate_caller_can_assign_key_org,
    )

    prisma = MagicMock()
    user_row = MagicMock()
    user_row.organization_memberships = None
    prisma.db.litellm_usertable.find_unique = AsyncMock(return_value=user_row)

    caller = UserAPIKeyAuth(
        user_id="alice",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
    )
    with pytest.raises(HTTPException) as exc_info:
        await _validate_caller_can_assign_key_org(
            user_api_key_dict=caller,
            organization_id="org-1",
            prisma_client=prisma,
        )
    assert exc_info.value.status_code == 403
