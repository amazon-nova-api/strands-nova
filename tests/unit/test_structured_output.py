"""Unit tests for NovaAPIModel structured output functionality."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import BaseModel

from strands.types.exceptions import (
    ContextWindowOverflowException,
    ModelThrottledException,
)
from amazon_nova.nova import NovaAPIModel


class SampleOutputModel(BaseModel):
    """Sample output model for testing."""

    name: str
    age: int
    email: str


class NestedOutputModel(BaseModel):
    """Nested output model for testing."""

    user: SampleOutputModel
    is_active: bool


class TestStructuredOutputFormatting:
    """Test structured output request formatting."""

    def test_structured_output_formats_request_correctly(self):
        """Test that structured output formats request with tool calling."""
        model = NovaAPIModel(model_id="nova-pro-v1", api_key="test-key")
        messages = [{"role": "user", "content": [{"text": "Get user info"}]}]

        # Mock convert_pydantic_to_tool_spec
        with patch("amazon_nova.nova.convert_pydantic_to_tool_spec") as mock_convert:
            mock_convert.return_value = {
                "name": "SampleOutputModel",
                "description": "Sample output model",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "integer"},
                            "email": {"type": "string"},
                        },
                    }
                },
            }

            request = model.format_request(
                messages, [mock_convert.return_value], None, {"any": {}}
            )

            # Override to non-streaming
            request["stream"] = False
            request.pop("stream_options", None)

            assert request["stream"] is False
            assert "stream_options" not in request
            assert request["tool_choice"] == "required"
            assert len(request["tools"]) == 1

    @pytest.mark.asyncio
    async def test_structured_output_success(self):
        """Test successful structured output parsing."""
        model = NovaAPIModel(model_id="nova-pro-v1", api_key="test-key")
        messages = [{"role": "user", "content": [{"text": "Get user info"}]}]

        mock_response_data = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "arguments": json.dumps(
                                        {
                                            "name": "John Doe",
                                            "age": 30,
                                            "email": "john@example.com",
                                        }
                                    )
                                }
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }

        with (
            patch("httpx.AsyncClient") as mock_client,
            patch("amazon_nova.nova.convert_pydantic_to_tool_spec") as mock_convert,
        ):
            mock_convert.return_value = {
                "name": "SampleOutputModel",
                "description": "Sample output model",
                "inputSchema": {"json": {"type": "object", "properties": {}}},
            }

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data

            mock_client_instance = MagicMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock()
            mock_client.return_value = mock_client_instance

            results = []
            async for result in model.structured_output(SampleOutputModel, messages):
                results.append(result)

            assert len(results) == 1
            assert "output" in results[0]
            output = results[0]["output"]
            assert isinstance(output, SampleOutputModel)
            assert output.name == "John Doe"
            assert output.age == 30
            assert output.email == "john@example.com"

    @pytest.mark.asyncio
    async def test_structured_output_no_choices(self):
        """Test structured output with no choices in response."""
        model = NovaAPIModel(model_id="nova-pro-v1", api_key="test-key")
        messages = [{"role": "user", "content": [{"text": "Get user info"}]}]

        mock_response_data = {"choices": []}

        with (
            patch("httpx.AsyncClient") as mock_client,
            patch("amazon_nova.nova.convert_pydantic_to_tool_spec") as mock_convert,
        ):
            mock_convert.return_value = {
                "name": "SampleOutputModel",
                "description": "Sample output model",
                "inputSchema": {"json": {"type": "object", "properties": {}}},
            }

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data

            mock_client_instance = MagicMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value = mock_client_instance

            with pytest.raises(ValueError, match="No choices found"):
                async for _ in model.structured_output(SampleOutputModel, messages):
                    pass

    @pytest.mark.asyncio
    async def test_structured_output_multiple_choices(self):
        """Test structured output with multiple choices (should use first and warn)."""
        model = NovaAPIModel(model_id="nova-pro-v1", api_key="test-key")
        messages = [{"role": "user", "content": [{"text": "Get user info"}]}]

        mock_response_data = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "arguments": json.dumps(
                                        {
                                            "name": "John Doe",
                                            "age": 30,
                                            "email": "john@example.com",
                                        }
                                    )
                                }
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                },
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "arguments": json.dumps(
                                        {
                                            "name": "Jane Doe",
                                            "age": 25,
                                            "email": "jane@example.com",
                                        }
                                    )
                                }
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                },
            ]
        }

        with (
            patch("httpx.AsyncClient") as mock_client,
            patch("amazon_nova.nova.convert_pydantic_to_tool_spec") as mock_convert,
        ):
            mock_convert.return_value = {
                "name": "SampleOutputModel",
                "description": "Sample output model",
                "inputSchema": {"json": {"type": "object", "properties": {}}},
            }

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data

            mock_client_instance = MagicMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock()
            mock_client.return_value = mock_client_instance

            results = []
            async for result in model.structured_output(SampleOutputModel, messages):
                results.append(result)

            # Should use first choice and succeed
            assert len(results) == 1
            assert "output" in results[0]
            output = results[0]["output"]
            # Should be the first choice (John Doe)
            assert output.name == "John Doe"
            assert output.age == 30

    @pytest.mark.asyncio
    async def test_structured_output_wrong_finish_reason(self):
        """Test structured output with wrong finish reason."""
        model = NovaAPIModel(model_id="nova-pro-v1", api_key="test-key")
        messages = [{"role": "user", "content": [{"text": "Get user info"}]}]

        mock_response_data = {
            "choices": [{"message": {"content": "Some text"}, "finish_reason": "stop"}]
        }

        with (
            patch("httpx.AsyncClient") as mock_client,
            patch("amazon_nova.nova.convert_pydantic_to_tool_spec") as mock_convert,
        ):
            mock_convert.return_value = {
                "name": "SampleOutputModel",
                "description": "Sample output model",
                "inputSchema": {"json": {"type": "object", "properties": {}}},
            }

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data

            mock_client_instance = MagicMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value = mock_client_instance

            with pytest.raises(ValueError, match="Expected tool_calls finish reason"):
                async for _ in model.structured_output(SampleOutputModel, messages):
                    pass

    @pytest.mark.asyncio
    async def test_structured_output_no_tool_calls_in_message(self):
        """Test structured output with no tool calls in message."""
        model = NovaAPIModel(model_id="nova-pro-v1", api_key="test-key")
        messages = [{"role": "user", "content": [{"text": "Get user info"}]}]

        mock_response_data = {
            "choices": [
                {"message": {"content": "Some text"}, "finish_reason": "tool_calls"}
            ]
        }

        with (
            patch("httpx.AsyncClient") as mock_client,
            patch("amazon_nova.nova.convert_pydantic_to_tool_spec") as mock_convert,
        ):
            mock_convert.return_value = {
                "name": "SampleOutputModel",
                "description": "Sample output model",
                "inputSchema": {"json": {"type": "object", "properties": {}}},
            }

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data

            mock_client_instance = MagicMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value = mock_client_instance

            with pytest.raises(ValueError, match="No tool_calls found in message"):
                async for _ in model.structured_output(SampleOutputModel, messages):
                    pass

    @pytest.mark.asyncio
    async def test_structured_output_invalid_json(self):
        """Test structured output with invalid JSON in tool call."""
        model = NovaAPIModel(model_id="nova-pro-v1", api_key="test-key")
        messages = [{"role": "user", "content": [{"text": "Get user info"}]}]

        mock_response_data = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [{"function": {"arguments": "invalid json"}}]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }

        with (
            patch("httpx.AsyncClient") as mock_client,
            patch("amazon_nova.nova.convert_pydantic_to_tool_spec") as mock_convert,
        ):
            mock_convert.return_value = {
                "name": "SampleOutputModel",
                "description": "Sample output model",
                "inputSchema": {"json": {"type": "object", "properties": {}}},
            }

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data

            mock_client_instance = MagicMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value = mock_client_instance

            with pytest.raises(ValueError, match="Failed to parse"):
                async for _ in model.structured_output(SampleOutputModel, messages):
                    pass

    @pytest.mark.asyncio
    async def test_structured_output_validation_error(self):
        """Test structured output with pydantic validation error."""
        model = NovaAPIModel(model_id="nova-pro-v1", api_key="test-key")
        messages = [{"role": "user", "content": [{"text": "Get user info"}]}]

        # Missing required field 'age'
        mock_response_data = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "arguments": json.dumps(
                                        {
                                            "name": "John Doe",
                                            "email": "john@example.com",
                                        }
                                    )
                                }
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }

        with (
            patch("httpx.AsyncClient") as mock_client,
            patch("amazon_nova.nova.convert_pydantic_to_tool_spec") as mock_convert,
        ):
            mock_convert.return_value = {
                "name": "SampleOutputModel",
                "description": "Sample output model",
                "inputSchema": {"json": {"type": "object", "properties": {}}},
            }

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data

            mock_client_instance = MagicMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value = mock_client_instance

            with pytest.raises(ValueError, match="Failed to parse"):
                async for _ in model.structured_output(SampleOutputModel, messages):
                    pass

    @pytest.mark.asyncio
    async def test_structured_output_with_system_prompt(self):
        """Test structured output with system prompt."""
        model = NovaAPIModel(model_id="nova-pro-v1", api_key="test-key")
        messages = [{"role": "user", "content": [{"text": "Get user info"}]}]
        system_prompt = "You are a helpful assistant"

        mock_response_data = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "arguments": json.dumps(
                                        {
                                            "name": "Jane Doe",
                                            "age": 25,
                                            "email": "jane@example.com",
                                        }
                                    )
                                }
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }

        with (
            patch("httpx.AsyncClient") as mock_client,
            patch("amazon_nova.nova.convert_pydantic_to_tool_spec") as mock_convert,
        ):
            mock_convert.return_value = {
                "name": "SampleOutputModel",
                "description": "Sample output model",
                "inputSchema": {"json": {"type": "object", "properties": {}}},
            }

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data

            mock_client_instance = MagicMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock()
            mock_client.return_value = mock_client_instance

            results = []
            async for result in model.structured_output(
                SampleOutputModel, messages, system_prompt
            ):
                results.append(result)

            assert len(results) == 1
            output = results[0]["output"]
            assert output.name == "Jane Doe"

    @pytest.mark.asyncio
    async def test_structured_output_timeout_error(self):
        """Test structured output with timeout error."""
        model = NovaAPIModel(model_id="nova-pro-v1", api_key="test-key")
        messages = [{"role": "user", "content": [{"text": "Get user info"}]}]

        with (
            patch("httpx.AsyncClient") as mock_client,
            patch("amazon_nova.nova.convert_pydantic_to_tool_spec") as mock_convert,
        ):
            mock_convert.return_value = {
                "name": "SampleOutputModel",
                "description": "Sample output model",
                "inputSchema": {"json": {"type": "object", "properties": {}}},
            }

            mock_client_instance = MagicMock()
            mock_client_instance.post = AsyncMock(
                side_effect=httpx.TimeoutException("Timeout")
            )
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value = mock_client_instance

            with pytest.raises(httpx.TimeoutException):
                async for _ in model.structured_output(SampleOutputModel, messages):
                    pass

    @pytest.mark.asyncio
    async def test_structured_output_request_error(self):
        """Test structured output with request error."""
        model = NovaAPIModel(model_id="nova-pro-v1", api_key="test-key")
        messages = [{"role": "user", "content": [{"text": "Get user info"}]}]

        with (
            patch("httpx.AsyncClient") as mock_client,
            patch("amazon_nova.nova.convert_pydantic_to_tool_spec") as mock_convert,
        ):
            mock_convert.return_value = {
                "name": "SampleOutputModel",
                "description": "Sample output model",
                "inputSchema": {"json": {"type": "object", "properties": {}}},
            }

            mock_client_instance = MagicMock()
            mock_client_instance.post = AsyncMock(
                side_effect=httpx.RequestError("Network error")
            )
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value = mock_client_instance

            with pytest.raises(httpx.RequestError):
                async for _ in model.structured_output(SampleOutputModel, messages):
                    pass

    @pytest.mark.asyncio
    async def test_structured_output_api_error_context_window(self):
        """Test structured output with context window overflow error."""
        model = NovaAPIModel(model_id="nova-pro-v1", api_key="test-key")
        messages = [{"role": "user", "content": [{"text": "Get user info"}]}]

        with (
            patch("httpx.AsyncClient") as mock_client,
            patch("amazon_nova.nova.convert_pydantic_to_tool_spec") as mock_convert,
        ):
            mock_convert.return_value = {
                "name": "SampleOutputModel",
                "description": "Sample output model",
                "inputSchema": {"json": {"type": "object", "properties": {}}},
            }

            mock_response = MagicMock()
            mock_response.status_code = 400
            mock_response.json.return_value = {
                "error": {"message": "context length exceeded"}
            }
            mock_response.aread = AsyncMock()

            mock_client_instance = MagicMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value = mock_client_instance

            with pytest.raises(ContextWindowOverflowException):
                async for _ in model.structured_output(SampleOutputModel, messages):
                    pass

    @pytest.mark.asyncio
    async def test_structured_output_api_error_throttled(self):
        """Test structured output with throttling error."""
        model = NovaAPIModel(model_id="nova-pro-v1", api_key="test-key")
        messages = [{"role": "user", "content": [{"text": "Get user info"}]}]

        with (
            patch("httpx.AsyncClient") as mock_client,
            patch("amazon_nova.nova.convert_pydantic_to_tool_spec") as mock_convert,
        ):
            mock_convert.return_value = {
                "name": "SampleOutputModel",
                "description": "Sample output model",
                "inputSchema": {"json": {"type": "object", "properties": {}}},
            }

            mock_response = MagicMock()
            mock_response.status_code = 429
            mock_response.json.return_value = {
                "error": {"message": "rate limit exceeded"}
            }
            mock_response.aread = AsyncMock()

            mock_client_instance = MagicMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value = mock_client_instance

            with pytest.raises(ModelThrottledException):
                async for _ in model.structured_output(SampleOutputModel, messages):
                    pass
