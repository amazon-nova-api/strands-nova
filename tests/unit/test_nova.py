"""Unit tests for Nova API model provider."""

from unittest.mock import AsyncMock, patch

import pytest

from strands_nova import NovaModel, NovaModelError


@pytest.fixture
def nova_model():
    """Create a NovaModel instance for testing."""
    return NovaModel(api_key="test-api-key", model="nova-premier-v1")


def test_initialization():
    """Test NovaModel initialization."""
    model = NovaModel(api_key="test-key", model="nova-premier-v1")
    assert model.api_key == "test-key"
    assert model.model == "nova-premier-v1"
    assert model.temperature == 0.7
    assert model.max_tokens == 4096


def test_initialization_with_params():
    """Test NovaModel initialization with custom parameters."""
    model = NovaModel(
        api_key="test-key",
        model="nova-premier-v1",
        temperature=0.5,
        max_tokens=2048,
        top_p=0.9,
    )
    assert model.temperature == 0.5
    assert model.max_tokens == 2048
    assert model.top_p == 0.9


def test_initialization_with_reasoning_effort():
    """Test NovaModel initialization with reasoning_effort parameter."""
    model = NovaModel(
        api_key="test-key",
        model="mumbai-flintflex-reasoning-v3",
        reasoning_effort="medium"
    )
    assert model.reasoning_effort == "medium"


def test_initialization_with_web_search():
    """Test NovaModel initialization with web_search_options parameter."""
    web_search_opts = {"search_context_size": "low"}
    model = NovaModel(
        api_key="test-key",
        model="nova-premier-v1",
        web_search_options=web_search_opts
    )
    assert model.web_search_options == web_search_opts


def test_initialization_without_api_key():
    """Test NovaModel initialization without API key."""
    with pytest.raises(ValueError, match="Nova API key is required"):
        NovaModel(model="nova-premier-v1")


@pytest.mark.asyncio
async def test_stream_with_successful_response(nova_model):
    """Test streaming with successful response."""
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "text/event-stream"}

    # Mock SSE stream - OpenAI format with choices array
    async def mock_aiter():
        yield b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
        yield b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n'
        yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'

    mock_response.aiter_bytes = mock_aiter

    with patch("httpx.AsyncClient") as mock_client:
        mock_instance = AsyncMock()
        mock_stream = AsyncMock()
        mock_stream.__aenter__.return_value = mock_response
        mock_instance.stream = AsyncMock(return_value=mock_stream)
        mock_client.return_value.__aenter__.return_value = mock_instance

        result = ""
        async for event in nova_model.stream("Test prompt"):
            # Check for contentBlockDelta events with text
            if isinstance(event, dict) and "contentBlockDelta" in event:
                delta = event["contentBlockDelta"].get("delta", {})
                if "text" in delta:
                    result += delta["text"]

        assert result == "Hello world"


@pytest.mark.asyncio
async def test_stream_with_error_response(nova_model):
    """Test streaming with error response."""
    mock_response = AsyncMock()
    mock_response.status_code = 401
    mock_response.aread = AsyncMock(return_value=b"Unauthorized")

    with patch("httpx.AsyncClient") as mock_client:
        mock_instance = AsyncMock()
        mock_stream = AsyncMock()
        mock_stream.__aenter__.return_value = mock_response
        mock_instance.stream = AsyncMock(return_value=mock_stream)
        mock_client.return_value.__aenter__.return_value = mock_instance

        with pytest.raises(NovaModelError, match="Nova API request failed"):
            async for _ in nova_model.stream("Test prompt"):
                pass


@pytest.mark.asyncio
async def test_stream_with_system_message(nova_model):
    """Test streaming with system message."""
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "text/event-stream"}

    async def mock_aiter():
        yield b'data: {"choices":[{"delta":{"content":"Response"}}]}\n\n'
        yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'

    mock_response.aiter_bytes = mock_aiter

    with patch("httpx.AsyncClient") as mock_client:
        mock_instance = AsyncMock()
        mock_stream = AsyncMock()
        mock_stream.__aenter__.return_value = mock_response
        mock_instance.stream = AsyncMock(return_value=mock_stream)
        mock_client.return_value.__aenter__.return_value = mock_instance

        result = ""
        async for event in nova_model.stream("Test prompt", system_prompt="You are a helpful assistant"):
            if isinstance(event, dict) and "contentBlockDelta" in event:
                delta = event["contentBlockDelta"].get("delta", {})
                if "text" in delta:
                    result += delta["text"]

        # Verify the system message was included in the request
        call_args = mock_instance.stream.call_args
        json_data = call_args.kwargs["json"]
        assert any(msg["role"] == "system" for msg in json_data["messages"])


@pytest.mark.asyncio
async def test_stream_with_tools(nova_model):
    """Test streaming with tool specifications."""
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "text/event-stream"}

    async def mock_aiter():
        yield b'data: {"choices":[{"delta":{"content":"Using tool"}}]}\n\n'
        yield b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n'

    mock_response.aiter_bytes = mock_aiter

    tool_specs = [
        {
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                }
            }
        }
    ]

    with patch("httpx.AsyncClient") as mock_client:
        mock_instance = AsyncMock()
        mock_stream = AsyncMock()
        mock_stream.__aenter__.return_value = mock_response
        mock_instance.stream = AsyncMock(return_value=mock_stream)
        mock_client.return_value.__aenter__.return_value = mock_instance

        async for event in nova_model.stream("What's the weather?", tool_specs=tool_specs):
            pass

        # Verify tools were included in the request
        call_args = mock_instance.stream.call_args
        json_data = call_args.kwargs["json"]
        assert "tools" in json_data
        assert json_data["tools"][0]["type"] == "function"


@pytest.mark.asyncio
async def test_structured_output_not_implemented(nova_model):
    """Test that structured_output raises NotImplementedError."""
    from pydantic import BaseModel

    class TestModel(BaseModel):
        result: str

    with pytest.raises(NotImplementedError, match="Structured output is not yet supported for Nova models"):
        # structured_output is an async generator, need to call it properly
        async for _ in nova_model.structured_output(TestModel, "Test prompt"):
            pass


def test_model_str_representation(nova_model):
    """Test string representation of NovaModel."""
    str_repr = str(nova_model)
    assert "NovaModel" in str_repr
    assert "nova-premier-v1" in str_repr


def test_update_config(nova_model):
    """Test updating model configuration."""
    nova_model.update_config(temperature=0.5, max_tokens=2048)
    assert nova_model.temperature == 0.5
    assert nova_model.max_tokens == 2048


def test_get_config(nova_model):
    """Test getting model configuration."""
    config = nova_model.get_config()
    assert config["model"] == "nova-premier-v1"
    assert config["temperature"] == 0.7
    assert config["max_tokens"] == 4096


def test_get_config_with_reasoning_effort():
    """Test getting model configuration with reasoning_effort."""
    model = NovaModel(
        api_key="test-key",
        model="mumbai-flintflex-reasoning-v3",
        reasoning_effort="high"
    )
    config = model.get_config()
    assert config["reasoning_effort"] == "high"


def test_get_config_with_web_search():
    """Test getting model configuration with web_search_options."""
    web_search_opts = {"search_context_size": "medium"}
    model = NovaModel(
        api_key="test-key",
        model="nova-premier-v1",
        web_search_options=web_search_opts
    )
    config = model.get_config()
    assert config["web_search_options"] == web_search_opts
