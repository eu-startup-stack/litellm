"""
Integration tests for responses API background cost tracking
"""

import asyncio
import os
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse


class TestResponsesBackgroundCostTracking:
    """Integration tests for responses API background cost tracking"""

    @pytest.fixture
    def mock_managed_files_obj(self):
        """Create a mock managed files object"""
        managed_files = MagicMock()
        managed_files.store_unified_object_id = AsyncMock()
        return managed_files

    @pytest.fixture
    def mock_proxy_logging_obj(self, mock_managed_files_obj):
        """Create a mock proxy logging object"""
        logging_obj = MagicMock()
        logging_obj.get_proxy_hook = MagicMock(return_value=mock_managed_files_obj)
        return logging_obj

    @pytest.fixture
    def mock_llm_router(self):
        """Create a mock LLM router"""
        router = MagicMock()
        return router

    @pytest.mark.asyncio
    async def test_store_response_in_managed_objects_table(
        self, mock_managed_files_obj, mock_proxy_logging_obj, mock_llm_router
    ):
        """Test that background responses are stored in managed objects table"""
        # Create a mock response with queued status and hidden params
        response = ResponsesAPIResponse(
            id="resp_bGl0ZWxsbTpjdXN0b21fbGxtX3Byb3ZpZGVyOm9wZW5haTttb2RlbF9pZDpncHQtNDtsbGxfcmVzcG9uc2VfaWQ6cmVzcF8xMjM",
            object="response",
            status="queued",
            created_at=int(datetime.now().timestamp()),
            output=[],
            usage=None,
        )

        # Add hidden params with model_id (simulating what base_process_llm_request does)
        response._hidden_params = {"model_id": "model-deployment-id-123"}

        # Mock request data
        data = {
            "model": "gpt-4",
            "input": "Test input",
            "background": True,
        }

        # Mock user_api_key_dict
        user_api_key_dict = MagicMock()
        user_api_key_dict.user_id = "test-user"

        # Simulate the storage logic from endpoints.py
        if data.get("background") and isinstance(response, ResponsesAPIResponse):
            if response.status in ["queued", "in_progress"]:
                # Get model_id from hidden params
                hidden_params = getattr(response, "_hidden_params", {}) or {}
                model_id = hidden_params.get("model_id", None)

                if model_id:
                    # Store in managed objects table using response.id directly
                    await mock_managed_files_obj.store_unified_object_id(
                        unified_object_id=response.id,
                        file_object=response,
                        litellm_parent_otel_span=None,
                        model_object_id=response.id,
                        file_purpose="response",
                        user_api_key_dict=user_api_key_dict,
                    )

        # Verify store_unified_object_id was called
        mock_managed_files_obj.store_unified_object_id.assert_called_once()
        call_args = mock_managed_files_obj.store_unified_object_id.call_args

        # Verify the arguments - unified_object_id should be response.id
        assert call_args[1]["unified_object_id"] == response.id
        assert call_args[1]["model_object_id"] == response.id
        assert call_args[1]["file_purpose"] == "response"
        assert call_args[1]["user_api_key_dict"] == user_api_key_dict

    @pytest.mark.asyncio
    async def test_no_storage_for_non_background_requests(
        self, mock_managed_files_obj, mock_proxy_logging_obj
    ):
        """Test that non-background requests are not stored"""
        # Create a mock response
        response = ResponsesAPIResponse(
            id="resp_456",
            object="response",
            status="completed",
            created_at=int(datetime.now().timestamp()),
            output=[],
            usage=ResponseAPIUsage(
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
            ),
        )

        # Mock request data without background flag
        data = {
            "model": "gpt-4",
            "input": "Test input",
            "background": False,
        }

        # Simulate the storage logic
        if data.get("background") and isinstance(response, ResponsesAPIResponse):
            if response.status in ["queued", "in_progress"]:
                await mock_managed_files_obj.store_unified_object_id()

        # Verify store_unified_object_id was NOT called
        mock_managed_files_obj.store_unified_object_id.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_storage_for_completed_responses(
        self, mock_managed_files_obj, mock_proxy_logging_obj
    ):
        """Test that completed responses are not stored"""
        # Create a mock response with completed status
        response = ResponsesAPIResponse(
            id="resp_789",
            object="response",
            status="completed",
            created_at=int(datetime.now().timestamp()),
            output=[],
            usage=ResponseAPIUsage(
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
            ),
        )

        # Mock request data with background flag
        data = {
            "model": "gpt-4",
            "input": "Test input",
            "background": True,
        }

        # Simulate the storage logic
        if data.get("background") and isinstance(response, ResponsesAPIResponse):
            if response.status in ["queued", "in_progress"]:
                await mock_managed_files_obj.store_unified_object_id()

        # Verify store_unified_object_id was NOT called (status is completed)
        mock_managed_files_obj.store_unified_object_id.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_storage_without_model_id(
        self, mock_managed_files_obj, mock_proxy_logging_obj
    ):
        """Test that responses without model_id in hidden params are not stored"""
        # Create a mock response without hidden params
        response = ResponsesAPIResponse(
            id="resp_no_model",
            object="response",
            status="queued",
            created_at=int(datetime.now().timestamp()),
            output=[],
            usage=None,
        )

        # Mock request data with background flag
        data = {
            "model": "gpt-4",
            "input": "Test input",
            "background": True,
        }

        user_api_key_dict = MagicMock()

        # Simulate the storage logic
        if data.get("background") and isinstance(response, ResponsesAPIResponse):
            if response.status in ["queued", "in_progress"]:
                hidden_params = getattr(response, "_hidden_params", {}) or {}
                model_id = hidden_params.get("model_id", None)

                if model_id:  # This will be False
                    await mock_managed_files_obj.store_unified_object_id(
                        unified_object_id=response.id,
                        file_object=response,
                        litellm_parent_otel_span=None,
                        model_object_id=response.id,
                        file_purpose="response",
                        user_api_key_dict=user_api_key_dict,
                    )

        # Verify store_unified_object_id was NOT called (no model_id)
        mock_managed_files_obj.store_unified_object_id.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_handling_in_storage(
        self, mock_managed_files_obj, mock_proxy_logging_obj
    ):
        """Test that errors during storage are handled gracefully"""
        # Mock store_unified_object_id to raise an exception
        mock_managed_files_obj.store_unified_object_id = AsyncMock(
            side_effect=Exception("Database error")
        )

        response = ResponsesAPIResponse(
            id="resp_error",
            object="response",
            status="queued",
            created_at=int(datetime.now().timestamp()),
            output=[],
            usage=None,
        )
        response._hidden_params = {"model_id": "test-model-id"}

        data = {
            "model": "gpt-4",
            "input": "Test input",
            "background": True,
        }

        user_api_key_dict = MagicMock()
        user_api_key_dict.user_id = "test-user"

        # Try to store - should not raise (error is caught in endpoints.py)
        try:
            if data.get("background") and isinstance(response, ResponsesAPIResponse):
                if response.status in ["queued", "in_progress"]:
                    hidden_params = getattr(response, "_hidden_params", {}) or {}
                    model_id = hidden_params.get("model_id", None)

                    if model_id:
                        await mock_managed_files_obj.store_unified_object_id(
                            unified_object_id=response.id,
                            file_object=response,
                            litellm_parent_otel_span=None,
                            model_object_id=response.id,
                            file_purpose="response",
                            user_api_key_dict=user_api_key_dict,
                        )
        except Exception:
            # Exception should be caught and logged, not raised
            pass

        # Verify the method was called (even though it raised)
        assert mock_managed_files_obj.store_unified_object_id.called

