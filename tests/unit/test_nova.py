"""Unit tests for Nova API model provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from strands.types.exceptions import (
    ContextWindowOverflowException,
    ModelThrottledException,
)

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
    model = NovaModel(api_key="test-key", model="mumbai-flintflex-reasoning-v3", reasoning_effort="medium")
    assert model.reasoning_effort == "medium"


def test_initialization_with_web_search():
    """Test NovaModel initialization with web_search_options parameter."""
    web_search_opts = {"search_context_size": "low"}
    model = NovaModel(api_key="test-key", model="nova-premier-v1", web_search_options=web_search_opts)
    assert model.web_search_options == web_search_opts


def test_initialization_without_api_key():
    """Test NovaModel initialization without API key."""
    # Temporarily remove NOVA_API_KEY from environment
    import os

    with patch.dict(os.environ, {"NOVA_API_KEY": ""}, clear=False):
        # Ensure the key is actually removed
        if "NOVA_API_KEY" in os.environ:
            del os.environ["NOVA_API_KEY"]
        with pytest.raises(ValueError, match="Nova API key is required"):
            NovaModel(model="nova-premier-v1")


@pytest.mark.asyncio
async def test_stream_with_successful_response(nova_model):
    """Test streaming with successful response and all event types."""
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "text/event-stream"}

    # Mock SSE stream - OpenAI format with choices array
    async def mock_aiter():
        yield b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
        yield b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
        yield b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n'
        yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        yield b'data: {"usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}}\n\n'
        yield b"data: [DONE]\n\n"

    mock_response.aiter_bytes = mock_aiter

    with patch("httpx.AsyncClient") as mock_client:
        mock_instance = MagicMock()
        mock_stream = AsyncMock()
        mock_stream.__aenter__.return_value = mock_response
        mock_instance.stream.return_value = mock_stream
        mock_client.return_value.__aenter__.return_value = mock_instance

        events = []
        result = ""
        async for event in nova_model.stream("Test prompt"):
            events.append(event)
            # Check for contentBlockDelta events with text
            if isinstance(event, dict) and "contentBlockDelta" in event:
                delta = event["contentBlockDelta"].get("delta", {})
                if "text" in delta:
                    result += delta["text"]

        assert result == "Hello world"

        # Verify all event types are present
        assert any("messageStart" in e for e in events), "Missing messageStart event"
        assert any("contentBlockStart" in e for e in events), "Missing contentBlockStart event"
        assert any("contentBlockDelta" in e for e in events), "Missing contentBlockDelta event"
        assert any("contentBlockStop" in e for e in events), "Missing contentBlockStop event"
        assert any("messageStop" in e for e in events), "Missing messageStop event"
        assert any("metadata" in e for e in events), "Missing metadata event"


@pytest.mark.asyncio
async def test_stream_with_authentication_error(nova_model):
    """Test streaming with authentication error (401)."""
    mock_response = AsyncMock()
    mock_response.status_code = 401
    mock_response.aread = AsyncMock(return_value=b"Unauthorized")

    with patch("httpx.AsyncClient") as mock_client:
        mock_instance = MagicMock()
        mock_stream = AsyncMock()
        mock_stream.__aenter__.return_value = mock_response
        mock_instance.stream.return_value = mock_stream
        mock_client.return_value.__aenter__.return_value = mock_instance

        with pytest.raises(NovaModelError, match="Authentication failed"):
            async for _ in nova_model.stream("Test prompt"):
                pass


@pytest.mark.asyncio
async def test_stream_with_rate_limit_error(nova_model):
    """Test streaming with rate limit error (429)."""
    mock_response = AsyncMock()
    mock_response.status_code = 429
    mock_response.aread = AsyncMock(return_value=b"Rate limit exceeded")

    with patch("httpx.AsyncClient") as mock_client:
        mock_instance = MagicMock()
        mock_stream = AsyncMock()
        mock_stream.__aenter__.return_value = mock_response
        mock_instance.stream.return_value = mock_stream
        mock_client.return_value.__aenter__.return_value = mock_instance

        with pytest.raises(ModelThrottledException, match="rate limit exceeded"):
            async for _ in nova_model.stream("Test prompt"):
                pass


@pytest.mark.asyncio
async def test_stream_with_context_overflow_error(nova_model):
    """Test streaming with context window overflow error (400)."""
    mock_response = AsyncMock()
    mock_response.status_code = 400
    mock_response.aread = AsyncMock(return_value=b"Context window exceeded maximum token limit")

    with patch("httpx.AsyncClient") as mock_client:
        mock_instance = MagicMock()
        mock_stream = AsyncMock()
        mock_stream.__aenter__.return_value = mock_response
        mock_instance.stream.return_value = mock_stream
        mock_client.return_value.__aenter__.return_value = mock_instance

        with pytest.raises(ContextWindowOverflowException, match="Context window exceeded"):
            async for _ in nova_model.stream("Test prompt"):
                pass


@pytest.mark.asyncio
async def test_stream_with_model_not_found_error(nova_model):
    """Test streaming with model not found error (404)."""
    mock_response = AsyncMock()
    mock_response.status_code = 404
    mock_response.aread = AsyncMock(return_value=b"Model not found")

    with patch("httpx.AsyncClient") as mock_client:
        mock_instance = MagicMock()
        mock_stream = AsyncMock()
        mock_stream.__aenter__.return_value = mock_response
        mock_instance.stream.return_value = mock_stream
        mock_client.return_value.__aenter__.return_value = mock_instance

        with pytest.raises(NovaModelError, match="Model.*not found"):
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
        mock_instance = MagicMock()
        mock_stream = AsyncMock()
        mock_stream.__aenter__.return_value = mock_response
        mock_instance.stream.return_value = mock_stream
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
    """Test streaming with tool specifications and correct format conversion."""
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "text/event-stream"}

    async def mock_aiter():
        yield b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
        yield b'data: {"choices":[{"delta":{"content":"Using tool"}}]}\n\n'
        yield b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n'

    mock_response.aiter_bytes = mock_aiter

    # Use Strands SDK format with inputSchema
    tool_specs = [
        {
            "name": "get_weather",
            "description": "Get current weather",
            "inputSchema": {  # Strands SDK uses inputSchema
                "type": "object",
                "properties": {"location": {"type": "string"}},
            },
        }
    ]

    with patch("httpx.AsyncClient") as mock_client:
        mock_instance = MagicMock()
        mock_stream = AsyncMock()
        mock_stream.__aenter__.return_value = mock_response
        mock_instance.stream.return_value = mock_stream
        mock_client.return_value.__aenter__.return_value = mock_instance

        async for event in nova_model.stream("What's the weather?", tool_specs=tool_specs):
            pass

        # Verify tools were included in the request with correct conversion
        call_args = mock_instance.stream.call_args
        json_data = call_args.kwargs["json"]
        assert "tools" in json_data
        assert json_data["tools"][0]["type"] == "function"
        # Verify inputSchema was converted to parameters
        assert "parameters" in json_data["tools"][0]["function"]
        assert json_data["tools"][0]["function"]["parameters"]["type"] == "object"


@pytest.mark.asyncio
async def test_tool_spec_conversion(nova_model):
    """Test that Strands SDK inputSchema gets converted to Nova API parameters."""
    # Strands SDK format
    strands_tool_specs = [
        {
            "name": "calculator",
            "description": "Perform calculations",
            "inputSchema": {
                "type": "object",
                "properties": {"operation": {"type": "string"}, "x": {"type": "number"}, "y": {"type": "number"}},
                "required": ["operation", "x", "y"],
            },
        }
    ]

    # Convert using the model's method
    nova_tools = nova_model._convert_tool_specs_to_nova_format(strands_tool_specs)

    # Verify conversion
    assert nova_tools is not None
    assert len(nova_tools) == 1
    assert nova_tools[0]["type"] == "function"
    assert nova_tools[0]["function"]["name"] == "calculator"
    assert nova_tools[0]["function"]["description"] == "Perform calculations"
    # inputSchema should be converted to parameters
    assert "parameters" in nova_tools[0]["function"]
    assert nova_tools[0]["function"]["parameters"]["type"] == "object"
    assert "operation" in nova_tools[0]["function"]["parameters"]["properties"]


@pytest.mark.asyncio
async def test_tool_choice_conversion():
    """Test tool_choice format conversion."""
    model = NovaModel(api_key="test-key", model="nova-premier-v1")

    # Test auto
    assert model._format_tool_choice({"auto": {}}) == "auto"

    # Test any -> required
    assert model._format_tool_choice({"any": {}}) == "required"

    # Test specific tool
    result = model._format_tool_choice({"tool": {"name": "my_tool"}})
    assert result == {"type": "function", "function": {"name": "my_tool"}}

    # Test None defaults to auto
    assert model._format_tool_choice(None) == "auto"


@pytest.mark.asyncio
async def test_stream_with_tool_calls(nova_model):
    """Test streaming with tool call events."""
    mock_response = AsyncMock()
    mock_response.status_code = 200

    async def mock_aiter():
        yield b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
        # Tool call start
        yield b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_123","type":"function","function":{"name":"get_weather","arguments":""}}]}}]}\n\n'
        # Tool arguments
        yield b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"location\\""}}]}}]}\n\n'
        yield b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":": \\"Seattle\\"}"}}]}}]}\n\n'
        # Finish
        yield b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n'

    mock_response.aiter_bytes = mock_aiter

    with patch("httpx.AsyncClient") as mock_client:
        mock_instance = MagicMock()
        mock_stream = AsyncMock()
        mock_stream.__aenter__.return_value = mock_response
        mock_instance.stream.return_value = mock_stream
        mock_client.return_value.__aenter__.return_value = mock_instance

        events = []
        async for event in nova_model.stream("What's the weather?"):
            events.append(event)

        # Verify tool-related events
        tool_start_events = [
            e
            for e in events
            if "contentBlockStart" in e and "toolUse" in e.get("contentBlockStart", {}).get("start", {})
        ]
        assert len(tool_start_events) > 0, "Should have tool start event"

        # Verify tool use structure
        tool_use = tool_start_events[0]["contentBlockStart"]["start"]["toolUse"]
        assert tool_use["name"] == "get_weather"
        assert "toolUseId" in tool_use


@pytest.mark.asyncio
async def test_stream_with_system_prompt_content(nova_model):
    """Test streaming with system_prompt_content parameter."""
    mock_response = AsyncMock()
    mock_response.status_code = 200

    async def mock_aiter():
        yield b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
        yield b'data: {"choices":[{"delta":{"content":"Response"}}]}\n\n'
        yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'

    mock_response.aiter_bytes = mock_aiter

    with patch("httpx.AsyncClient") as mock_client:
        mock_instance = MagicMock()
        mock_stream = AsyncMock()
        mock_stream.__aenter__.return_value = mock_response
        mock_instance.stream.return_value = mock_stream
        mock_client.return_value.__aenter__.return_value = mock_instance

        # Use system_prompt_content
        system_content = [{"text": "You are a helpful assistant."}, {"text": "Always be concise."}]

        async for event in nova_model.stream("Test prompt", system_prompt_content=system_content):
            pass

        # Verify system messages were included
        call_args = mock_instance.stream.call_args
        json_data = call_args.kwargs["json"]
        system_messages = [msg for msg in json_data["messages"] if msg["role"] == "system"]
        assert len(system_messages) == 2
        assert system_messages[0]["content"] == "You are a helpful assistant."
        assert system_messages[1]["content"] == "Always be concise."


@pytest.mark.asyncio
async def test_structured_output_not_implemented(nova_model):
    """Test that structured_output raises NotImplementedError."""
    from pydantic import BaseModel

    class TestModel(BaseModel):
        result: str

    with pytest.raises(NotImplementedError, match="Structured output is not yet supported"):
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
    model = NovaModel(api_key="test-key", model="mumbai-flintflex-reasoning-v3", reasoning_effort="high")
    config = model.get_config()
    assert config["reasoning_effort"] == "high"


def test_get_config_with_web_search():
    """Test getting model configuration with web_search_options."""
    web_search_opts = {"search_context_size": "medium"}
    model = NovaModel(api_key="test-key", model="nova-premier-v1", web_search_options=web_search_opts)
    config = model.get_config()
    assert config["web_search_options"] == web_search_opts
