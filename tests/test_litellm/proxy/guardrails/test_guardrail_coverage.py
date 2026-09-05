"""
Regression tests for guardrail-coverage gaps.

Each test confirms that a previously-bypassable input shape now triggers
inspection by the relevant guardrail hook:

- VERIA-11: multimodal list-format ``content`` is inspected (no longer
  silently skipped because of an ``isinstance(content, str)`` check).
- fniVO9-F: Responses-API ``data["input"]`` is inspected (no longer
  silently skipped because the hook only looked at ``data["messages"]``).
- yVS0wMDO: Aim's post-call hook inspects every choice when ``n>1``,
  not just ``choices[0]``.
"""

from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import Request, Response

from litellm import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import Choices, Message, ModelResponse


@pytest.fixture
def user_api_key():
    return UserAPIKeyAuth(api_key="hashed", user_id="u", key_alias=None)


# ── Aim ───────────────────────────────────────────────────────────────────────


def _aim_no_action_response() -> Response:
    return Response(
        status_code=200,
        json={"required_action": None},
        request=Request("POST", "https://api.aim.security/fw/v1/analyze"),
    )


@pytest.mark.asyncio
async def test_aim_inspects_multimodal_list_content(user_api_key, monkeypatch):
    monkeypatch.setenv("AIM_API_KEY", "hs-aim-key")
    from litellm.proxy.guardrails.guardrail_hooks.aim.aim import AimGuardrail

    guard = AimGuardrail()
    sent_payload: Dict[str, Any] = {}

    async def capture(url, headers, json):
        sent_payload.update(json)
        return _aim_no_action_response()

    with patch.object(guard.async_handler, "post", side_effect=capture):
        await guard.async_pre_call_hook(
            user_api_key_dict=user_api_key,
            cache=DualCache(),
            data={
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "secret payload"},
                            {"type": "image_url", "image_url": {"url": "..."}},
                        ],
                    }
                ]
            },
            call_type="acompletion",
        )

    # The multimodal text part must be visible to Aim.
    assert sent_payload["messages"] == [{"role": "user", "content": "secret payload"}]


@pytest.mark.asyncio
async def test_aim_inspects_responses_api_input(user_api_key, monkeypatch):
    monkeypatch.setenv("AIM_API_KEY", "hs-aim-key")
    from litellm.proxy.guardrails.guardrail_hooks.aim.aim import AimGuardrail

    guard = AimGuardrail()
    sent_payload: Dict[str, Any] = {}

    async def capture(url, headers, json):
        sent_payload.update(json)
        return _aim_no_action_response()

    with patch.object(guard.async_handler, "post", side_effect=capture):
        await guard.async_pre_call_hook(
            user_api_key_dict=user_api_key,
            cache=DualCache(),
            data={"input": "responses-api content"},
            call_type="acompletion",
        )

    assert sent_payload["messages"] == [
        {"role": "user", "content": "responses-api content"}
    ]


@pytest.mark.asyncio
async def test_aim_post_call_inspects_all_choices(user_api_key, monkeypatch):
    """yVS0wMDO: ``n>1`` no longer bypasses Aim by hiding violations in
    ``choices[1+]``."""
    monkeypatch.setenv("AIM_API_KEY", "hs-aim-key")
    from litellm.proxy.guardrails.guardrail_hooks.aim.aim import AimGuardrail

    guard = AimGuardrail()
    inspected_outputs = []

    async def capture(request_data, output, hook, key_alias):
        inspected_outputs.append(output)
        return {"redacted_output": output}

    response = ModelResponse(
        choices=[
            Choices(index=0, message=Message(role="assistant", content="first")),
            Choices(index=1, message=Message(role="assistant", content="second")),
            Choices(index=2, message=Message(role="assistant", content="third")),
        ]
    )

    with patch.object(guard, "call_aim_guardrail_on_output", side_effect=capture):
        await guard.async_post_call_success_hook(
            data={"messages": [{"role": "user", "content": "hi"}]},
            user_api_key_dict=user_api_key,
            response=response,
        )

    # ``asyncio.gather`` is used for parallelism, so order of inspection is
    # not guaranteed.
    assert sorted(inspected_outputs) == ["first", "second", "third"]


# ── Lakera v2 ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_lakera_v2_inspects_responses_api_input(user_api_key, monkeypatch):
    monkeypatch.setenv("LAKERA_API_KEY", "lk-test")
    from litellm.proxy.guardrails.guardrail_hooks.lakera_ai_v2 import (
        LakeraAIGuardrail,
    )

    guard = LakeraAIGuardrail(api_key="lk-test", on_flagged="monitor")

    seen_messages = []

    async def fake_call_v2_guard(messages, request_data, event_type):
        seen_messages.append(messages)
        return {"flagged": False}, {}

    with patch.object(guard, "call_v2_guard", side_effect=fake_call_v2_guard):
        await guard.async_pre_call_hook(
            user_api_key_dict=user_api_key,
            cache=DualCache(),
            data={"input": "responses-api content"},
            call_type="responses",
        )

    assert seen_messages == [[{"role": "user", "content": "responses-api content"}]]


@pytest.mark.asyncio
async def test_lakera_v2_responses_api_input_redacted_writeback(
    user_api_key, monkeypatch
):
    """Greptile P1: when input arrives via Responses-API ``data["input"]``
    (string) and Lakera flags PII, the redacted content must be written
    back to ``data["input"]`` — the Responses-API backend reads from
    ``input``, so writing only to ``messages`` would let unredacted PII
    reach the LLM."""
    monkeypatch.setenv("LAKERA_API_KEY", "lk-test")
    from litellm.proxy.guardrails.guardrail_hooks.lakera_ai_v2 import (
        LakeraAIGuardrail,
    )

    guard = LakeraAIGuardrail(api_key="lk-test", on_flagged="block")

    async def fake_call_v2_guard(messages, request_data, event_type):
        return ({"flagged": True, "payload": []}, {"EMAIL": 1})

    def fake_mask(messages, lakera_response, masked_entity_count):
        return [{"role": "user", "content": "[REDACTED EMAIL]"}]

    with (
        patch.object(guard, "call_v2_guard", side_effect=fake_call_v2_guard),
        patch.object(guard, "_is_only_pii_violation", return_value=True),
        patch.object(guard, "_mask_pii_in_messages", side_effect=fake_mask),
    ):
        data = {"input": "user@example.com leaked"}
        await guard.async_pre_call_hook(
            user_api_key_dict=user_api_key,
            cache=DualCache(),
            data=data,
            call_type="responses",
        )

    assert data["input"] == "[REDACTED EMAIL]"


@pytest.mark.asyncio
async def test_aim_responses_api_input_anonymize_writeback(user_api_key, monkeypatch):
    """Greptile P1: Aim's anonymize action must redact ``data["input"]``
    for Responses-API requests, not just ``data["messages"]``."""
    monkeypatch.setenv("AIM_API_KEY", "hs-aim-key")
    from litellm.proxy.guardrails.guardrail_hooks.aim.aim import AimGuardrail

    guard = AimGuardrail()

    aim_response_body = {
        "required_action": {"action_type": "anonymize_action"},
        "redacted_chat": {
            "all_redacted_messages": [
                {"role": "user", "content": "[REDACTED] anonymised"}
            ]
        },
    }

    async def capture(url, headers, json):
        return Response(
            status_code=200,
            json=aim_response_body,
            request=Request("POST", "https://api.aim.security/fw/v1/analyze"),
        )

    with patch.object(guard.async_handler, "post", side_effect=capture):
        data = {"input": "user@example.com leaked"}
        await guard.async_pre_call_hook(
            user_api_key_dict=user_api_key,
            cache=DualCache(),
            data=data,
            call_type="responses",
        )

    assert data["input"] == "[REDACTED] anonymised"


@pytest.mark.asyncio
async def test_lakera_v2_multimodal_pii_degrades_to_block(user_api_key, monkeypatch):
    """Mask-in-place uses Lakera offsets and cannot preserve image/audio
    parts of multimodal input. When PII is detected on a multimodal
    request, the hook must raise the block exception instead of silently
    flattening ``data["messages"]`` to text-only."""
    monkeypatch.setenv("LAKERA_API_KEY", "lk-test")
    from fastapi import HTTPException

    from litellm.proxy.guardrails.guardrail_hooks.lakera_ai_v2 import (
        LakeraAIGuardrail,
    )

    guard = LakeraAIGuardrail(api_key="lk-test", on_flagged="block")

    async def fake_call_v2_guard(messages, request_data, event_type):
        return (
            {
                "flagged": True,
                "payload": [{"detector_type": "pii/email", "start": 0, "end": 5}],
            },
            {"EMAIL": 1},
        )

    with (
        patch.object(guard, "call_v2_guard", side_effect=fake_call_v2_guard),
        patch.object(guard, "_is_only_pii_violation", return_value=True),
        patch.object(
            guard,
            "_get_http_exception_for_blocked_guardrail",
            return_value=HTTPException(status_code=400, detail="blocked"),
        ),
    ):
        with pytest.raises(HTTPException):
            await guard.async_pre_call_hook(
                user_api_key_dict=user_api_key,
                cache=DualCache(),
                data={
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "leak"},
                                {"type": "image_url", "image_url": {"url": "..."}},
                            ],
                        }
                    ]
                },
                call_type="acompletion",
            )


@pytest.mark.asyncio
async def test_lakera_v2_inspects_multimodal_list_content(user_api_key, monkeypatch):
    monkeypatch.setenv("LAKERA_API_KEY", "lk-test")
    from litellm.proxy.guardrails.guardrail_hooks.lakera_ai_v2 import (
        LakeraAIGuardrail,
    )

    guard = LakeraAIGuardrail(api_key="lk-test", on_flagged="monitor")
    seen_messages = []

    async def fake_call_v2_guard(messages, request_data, event_type):
        seen_messages.append(messages)
        return {"flagged": False}, {}

    with patch.object(guard, "call_v2_guard", side_effect=fake_call_v2_guard):
        await guard.async_pre_call_hook(
            user_api_key_dict=user_api_key,
            cache=DualCache(),
            data={
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "AKIAEXAMPLE"},
                            {"type": "image_url", "image_url": {"url": "..."}},
                        ],
                    }
                ]
            },
            call_type="acompletion",
        )

    assert seen_messages == [[{"role": "user", "content": "AKIAEXAMPLE"}]]


# ── Lasso ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_lasso_multimodal_falls_back_to_classify(user_api_key, monkeypatch):
    """Lasso's classifix (mask) endpoint returns text that overwrites
    ``data["messages"]``. For multimodal input that would silently strip
    image parts — the hook must use the classify endpoint instead and
    leave the original payload intact."""
    monkeypatch.setenv("LASSO_API_KEY", "ls-test")
    from litellm.proxy.guardrails.guardrail_hooks.lasso.lasso import LassoGuardrail

    guard = LassoGuardrail(lasso_api_key="ls-test", mask=True)

    masking_called = False
    classify_called = False

    async def fake_masking(data, cache, message_type, messages):
        nonlocal masking_called
        masking_called = True
        return data

    async def fake_classification(data, cache, message_type, messages):
        nonlocal classify_called
        classify_called = True
        return data

    with (
        patch.object(guard, "_handle_masking", side_effect=fake_masking),
        patch.object(guard, "_handle_classification", side_effect=fake_classification),
    ):
        await guard._run_lasso_guardrail(
            data={
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "hello"},
                            {"type": "image_url", "image_url": {"url": "..."}},
                        ],
                    }
                ]
            },
            cache=DualCache(),
            message_type="PROMPT",
        )

    assert classify_called is True
    assert masking_called is False


@pytest.mark.asyncio
async def test_lasso_inspects_responses_api_input(user_api_key, monkeypatch):
    monkeypatch.setenv("LASSO_API_KEY", "ls-test")
    from litellm.proxy.guardrails.guardrail_hooks.lasso.lasso import LassoGuardrail

    guard = LassoGuardrail(lasso_api_key="ls-test")

    seen_messages = []

    async def fake_handle_classification(data, cache, message_type, messages):
        seen_messages.append(messages)
        return data

    with patch.object(
        guard, "_handle_classification", side_effect=fake_handle_classification
    ):
        await guard._run_lasso_guardrail(
            data={"input": "responses-api content"},
            cache=DualCache(),
            message_type="PROMPT",
        )

    assert seen_messages == [[{"role": "user", "content": "responses-api content"}]]


@pytest.mark.asyncio
async def test_lasso_masking_writes_back_responses_api_input(user_api_key, monkeypatch):
    """Krrish blocker: Lasso classifix masking must update ``data["input"]``
    for Responses-API requests, not only ``data["messages"]``."""
    monkeypatch.setenv("LASSO_API_KEY", "ls-test")
    from litellm.proxy.guardrails.guardrail_hooks.lasso.lasso import LassoGuardrail

    guard = LassoGuardrail(lasso_api_key="ls-test", mask=True)
    lasso_response = {
        "violations_detected": True,
        "deputies": {"pii": True},
        "findings": {"pii": [{"action": "AUTO_MASKING"}]},
        "messages": [{"role": "user", "content": "[REDACTED]"}],
    }

    async def fake_call_lasso_api(headers, payload, api_url=None):
        return lasso_response

    data = {"input": "user@example.com leaked"}

    with patch.object(guard, "_call_lasso_api", side_effect=fake_call_lasso_api):
        await guard._run_lasso_guardrail(
            data=data,
            cache=DualCache(),
            message_type="PROMPT",
        )

    assert data["input"] == "[REDACTED]"


# ── Azure Content Safety ──────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call_type, data",
    [
        (
            "acompletion",
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "scan me"},
                            {"type": "image_url", "image_url": {"url": "..."}},
                        ],
                    }
                ]
            },
        ),
        ("aresponses", {"input": "scan me"}),
    ],
)
async def test_azure_content_safety_pre_call_fires_on_runtime_call_types(
    user_api_key, call_type, data
):
    """The proxy ingress passes ``route_type`` straight through as
    ``call_type`` — ``acompletion`` for chat completions and
    ``aresponses`` for the Responses API. The hook must inspect text
    fragments under both, not only the literal ``"completion"`` string
    used by some SDK callers."""
    from litellm.proxy.hooks.azure_content_safety import _PROXY_AzureContentSafety

    guard = _PROXY_AzureContentSafety.__new__(_PROXY_AzureContentSafety)
    seen = []

    async def fake_test_violation(content, source=None):
        seen.append((content, source))

    guard.test_violation = fake_test_violation
    await guard.async_pre_call_hook(
        user_api_key_dict=user_api_key,
        cache=DualCache(),
        data=data,
        call_type=call_type,
    )
    assert ("scan me", "input") in seen


@pytest.mark.asyncio
async def test_azure_content_safety_post_call_checks_all_choices(user_api_key):
    """Krrish blocker: ``n>1`` responses must not bypass Azure Content Safety
    by placing the unsafe text in ``choices[1+]``."""
    from fastapi import HTTPException
    from litellm.proxy.hooks.azure_content_safety import _PROXY_AzureContentSafety

    guard = _PROXY_AzureContentSafety.__new__(_PROXY_AzureContentSafety)
    seen_outputs = []

    async def fake_test_violation(content, source=None):
        seen_outputs.append((content, source))
        if "unsafe" in content:
            raise HTTPException(status_code=400, detail={"error": "unsafe"})

    guard.test_violation = fake_test_violation
    response = ModelResponse(
        choices=[
            Choices(index=0, message=Message(role="assistant", content="clean")),
            Choices(index=1, message=Message(role="assistant", content="unsafe")),
            Choices(index=2, message=Message(role="assistant", content="later")),
        ]
    )

    with pytest.raises(HTTPException):
        await guard.async_post_call_success_hook(
            data={},
            user_api_key_dict=user_api_key,
            response=response,
        )

    assert seen_outputs == [("clean", "output"), ("unsafe", "output")]

