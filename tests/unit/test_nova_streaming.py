"""Comprehensive unit tests for NovaModel streaming functionality."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from strands_nova import NovaModel


@pytest.fixture
def nova_model():
    """Create a NovaModel instance for testing."""
    return NovaModel(api_key="test-api-key", model="nova-premier-v1")


class TestBasicStreaming:
    """Test basic streaming functionality."""

    @pytest.mark.asyncio
    async def test_stream_basic_text_response(self, nova_model):
        """Test streaming with basic text response."""
        mock_response = AsyncMock()
        mock_response.status_code = 200

        async def mock_aiter():
            yield b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
            yield b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
            yield b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n'
            yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
            yield b"data: [DONE]\n\n"

        mock_response.aiter_bytes = mock_aiter

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_stream = AsyncMock()
            mock_stream.__aenter__.return_value = mock_response
            mock_instance.stream.return_value = mock_stream
            mock_client.return_value.__aenter__.return_value = mock_instance

            events = []
            async for event in nova_model.stream("Test prompt"):
                events.append(event)

            assert len(events) > 0
            assert any("messageStart" in e for e in events)
            assert any("contentBlockStart" in e for e in events)
            assert any("contentBlockDelta" in e for e in events)
            assert any("contentBlockStop" in e for e in events)
            assert any("messageStop" in e for e in events)

    @pytest.mark.asyncio
    async def test_stream_with_empty_response(self, nova_model):
        """Test streaming with empty response."""
        mock_response = AsyncMock()
        mock_response.status_code = 200

        async def mock_aiter():
            yield b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
            yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
            yield b"data: [DONE]\n\n"

        mock_response.aiter_bytes = mock_aiter

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_stream = AsyncMock()
            mock_stream.__aenter__.return_value = mock_response
            mock_instance.stream.return_value = mock_stream
            mock_client.return_value.__aenter__.return_value = mock_instance

            events = []
            async for event in nova_model.stream("Test prompt"):
                events.append(event)

            # Should still emit messageStart and messageStop
            assert any("messageStart" in e for e in events)
            assert any("messageStop" in e for e in events)

    @pytest.mark.asyncio
    async def test_stream_accumulates_text_correctly(self, nova_model):
        """Test that streaming accumulates text content correctly."""
        mock_response = AsyncMock()
        mock_response.status_code = 200

        async def mock_aiter():
            yield b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
            yield b'data: {"choices":[{"delta":{"content":"The "}}]}\n\n'
            yield b'data: {"choices":[{"delta":{"content":"quick "}}]}\n\n'
            yield b'data: {"choices":[{"delta":{"content":"brown "}}]}\n\n'
            yield b'data: {"choices":[{"delta":{"content":"fox"}}]}\n\n'
            yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'

        mock_response.aiter_bytes = mock_aiter

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_stream = AsyncMock()
            mock_stream.__aenter__.return_value = mock_response
            mock_instance.stream.return_value = mock_stream
            mock_client.return_value.__aenter__.return_value = mock_instance

            accumulated_text = ""
            async for event in nova_model.stream("Test prompt"):
                if "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"].get("delta", {})
                    if "text" in delta:
                        accumulated_text += delta["text"]

            assert accumulated_text == "The quick brown fox"


class TestStreamingWithMetadata:
    """Test streaming with usage metadata."""

    @pytest.mark.asyncio
    async def test_stream_includes_usage_metadata(self, nova_model):
        """Test that streaming includes usage metadata."""
        mock_response = AsyncMock()
        mock_response.status_code = 200

        async def mock_aiter():
            yield b'data: {"choices":[{"delta":{"content":"Test"}}]}\n\n'
            yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
            yield b'data: {"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}\n\n'

        mock_response.aiter_bytes = mock_aiter

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_stream = AsyncMock()
            mock_stream.__aenter__.return_value = mock_response
            mock_instance.stream.return_value = mock_stream
            mock_client.return_value.__aenter__.return_value = mock_instance

            events = []
            async for event in nova_model.stream("Test prompt"):
                events.append(event)

            metadata_events = [e for e in events if "metadata" in e]
            assert len(metadata_events) == 1

            metadata = metadata_events[0]["metadata"]
            assert metadata["usage"]["inputTokens"] == 10
            assert metadata["usage"]["outputTokens"] == 5
            assert metadata["usage"]["totalTokens"] == 15

    @pytest.mark.asyncio
    async def test_stream_metadata_has_metrics(self, nova_model):
        """Test that metadata includes metrics."""
        mock_response = AsyncMock()
        mock_response.status_code = 200

        async def mock_aiter():
            yield b'data: {"choices":[{"delta":{"content":"Test"}}]}\n\n'
            yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
            yield b'data: {"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}\n\n'

        mock_response.aiter_bytes = mock_aiter

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_stream = AsyncMock()
            mock_stream.__aenter__.return_value = mock_response
            mock_instance.stream.return_value = mock_stream
            mock_client.return_value.__aenter__.return_value = mock_instance

            events = []
            async for event in nova_model.stream("Test prompt"):
                events.append(event)

            metadata_events = [e for e in events if "metadata" in e]
            assert "metrics" in metadata_events[0]["metadata"]
            assert "latencyMs" in metadata_events[0]["metadata"]["metrics"]


class TestStreamingWithSystemPrompts:
    """Test streaming with system prompts."""

    @pytest.mark.asyncio
    async def test_stream_with_text_system_prompt(self, nova_model):
        """Test streaming with text system prompt."""
        mock_response = AsyncMock()
        mock_response.status_code = 200

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

            async for _ in nova_model.stream("User prompt", system_prompt="You are helpful"):
                pass

            call_args = mock_instance.stream.call_args
            json_data = call_args.kwargs["json"]
            system_msgs = [m for m in json_data["messages"] if m["role"] == "system"]
            assert len(system_msgs) == 1
            assert system_msgs[0]["content"] == "You are helpful"

    @pytest.mark.asyncio
    async def test_stream_with_system_prompt_content(self, nova_model):
        """Test streaming with system_prompt_content."""
        mock_response = AsyncMock()
        mock_response.status_code = 200

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

            system_content = [{"text": "First instruction"}, {"text": "Second instruction"}]

            async for _ in nova_model.stream("User prompt", system_prompt_content=system_content):
                pass

            call_args = mock_instance.stream.call_args
            json_data = call_args.kwargs["json"]
            system_msgs = [m for m in json_data["messages"] if m["role"] == "system"]
            assert len(system_msgs) == 2

    @pytest.mark.asyncio
    async def test_stream_with_both_system_prompt_types(self, nova_model):
        """Test streaming with both system_prompt and system_prompt_content."""
        mock_response = AsyncMock()
        mock_response.status_code = 200

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

            async for _ in nova_model.stream(
                "User prompt", system_prompt="Text prompt", system_prompt_content=[{"text": "Content prompt"}]
            ):
                pass

            call_args = mock_instance.stream.call_args
            json_data = call_args.kwargs["json"]
            system_msgs = [m for m in json_data["messages"] if m["role"] == "system"]
            assert len(system_msgs) == 2


class TestStreamingWithTools:
    """Test streaming with tool specifications."""

    @pytest.mark.asyncio
    async def test_stream_with_tool_specs(self, nova_model):
        """Test streaming with tool specifications."""
        mock_response = AsyncMock()
        mock_response.status_code = 200

        async def mock_aiter():
            yield b'data: {"choices":[{"delta":{"content":"Using tools"}}]}\n\n'
            yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'

        mock_response.aiter_bytes = mock_aiter

        tool_specs = [
            {"name": "get_weather", "description": "Get weather", "inputSchema": {"type": "object", "properties": {}}}
        ]

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_stream = AsyncMock()
            mock_stream.__aenter__.return_value = mock_response
            mock_instance.stream.return_value = mock_stream
            mock_client.return_value.__aenter__.return_value = mock_instance

            async for _ in nova_model.stream("Test", tool_specs=tool_specs):
                pass

            call_args = mock_instance.stream.call_args
            json_data = call_args.kwargs["json"]
            assert "tools" in json_data
            assert len(json_data["tools"]) == 1

    @pytest.mark.asyncio
    async def test_stream_with_tool_choice(self, nova_model):
        """Test streaming with tool_choice parameter."""
        mock_response = AsyncMock()
        mock_response.status_code = 200

        async def mock_aiter():
            yield b'data: {"choices":[{"delta":{"content":"Response"}}]}\n\n'
            yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'

        mock_response.aiter_bytes = mock_aiter

        tool_specs = [{"name": "my_tool", "description": "Tool", "inputSchema": {}}]

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_stream = AsyncMock()
            mock_stream.__aenter__.return_value = mock_response
            mock_instance.stream.return_value = mock_stream
            mock_client.return_value.__aenter__.return_value = mock_instance

            async for _ in nova_model.stream("Test", tool_specs=tool_specs, tool_choice={"tool": {"name": "my_tool"}}):
                pass

            call_args = mock_instance.stream.call_args
            json_data = call_args.kwargs["json"]
            assert "tool_choice" in json_data
            assert json_data["tool_choice"]["function"]["name"] == "my_tool"

    @pytest.mark.asyncio
    async def test_stream_with_tool_call_response(self, nova_model):
        """Test streaming with tool call in response."""
        mock_response = AsyncMock()
        mock_response.status_code = 200

        async def mock_aiter():
            yield b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
            yield b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_123","type":"function","function":{"name":"get_weather","arguments":""}}]}}]}\n\n'
            yield b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"loc\\""}}]}}]}\n\n'
            yield b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ation\\":\\"NYC\\"}"}}]}}]}\n\n'
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

            tool_start_events = [
                e
                for e in events
                if "contentBlockStart" in e and "toolUse" in e.get("contentBlockStart", {}).get("start", {})
            ]
            assert len(tool_start_events) > 0

    @pytest.mark.asyncio
    async def test_stream_finish_reason_tool_calls(self, nova_model):
        """Test that tool_calls finish_reason is mapped correctly."""
        mock_response = AsyncMock()
        mock_response.status_code = 200

        async def mock_aiter():
            yield b'data: {"choices":[{"delta":{"content":"Text"}}]}\n\n'
            yield b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n'

        mock_response.aiter_bytes = mock_aiter

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_stream = AsyncMock()
            mock_stream.__aenter__.return_value = mock_response
            mock_instance.stream.return_value = mock_stream
            mock_client.return_value.__aenter__.return_value = mock_instance

            events = []
            async for event in nova_model.stream("Test"):
                events.append(event)

            stop_events = [e for e in events if "messageStop" in e]
            assert len(stop_events) == 1
            assert stop_events[0]["messageStop"]["stopReason"] == "tool_use"


class TestStreamingFinishReasons:
    """Test different finish reasons in streaming."""

    @pytest.mark.asyncio
    async def test_stream_finish_reason_stop(self, nova_model):
        """Test finish_reason 'stop' maps to 'end_turn'."""
        mock_response = AsyncMock()
        mock_response.status_code = 200

        async def mock_aiter():
            yield b'data: {"choices":[{"delta":{"content":"Text"}}]}\n\n'
            yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'

        mock_response.aiter_bytes = mock_aiter

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_stream = AsyncMock()
            mock_stream.__aenter__.return_value = mock_response
            mock_instance.stream.return_value = mock_stream
            mock_client.return_value.__aenter__.return_value = mock_instance

            events = []
            async for event in nova_model.stream("Test"):
                events.append(event)

            stop_events = [e for e in events if "messageStop" in e]
            assert stop_events[0]["messageStop"]["stopReason"] == "end_turn"

    @pytest.mark.asyncio
    async def test_stream_finish_reason_length(self, nova_model):
        """Test finish_reason 'length' maps to 'max_tokens'."""
        mock_response = AsyncMock()
        mock_response.status_code = 200

        async def mock_aiter():
            yield b'data: {"choices":[{"delta":{"content":"Text"}}]}\n\n'
            yield b'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\n'

        mock_response.aiter_bytes = mock_aiter

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_stream = AsyncMock()
            mock_stream.__aenter__.return_value = mock_response
            mock_instance.stream.return_value = mock_stream
            mock_client.return_value.__aenter__.return_value = mock_instance

            events = []
            async for event in nova_model.stream("Test"):
                events.append(event)

            stop_events = [e for e in events if "messageStop" in e]
            assert stop_events[0]["messageStop"]["stopReason"] == "max_tokens"


class TestStreamingRequestConfiguration:
    """Test request configuration in streaming."""

    @pytest.mark.asyncio
    async def test_stream_includes_temperature_in_request(self, nova_model):
        """Test that streaming includes temperature in request."""
        mock_response = AsyncMock()
        mock_response.status_code = 200

        async def mock_aiter():
            yield b'data: {"choices":[{"delta":{"content":"Test"}}]}\n\n'
            yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'

        mock_response.aiter_bytes = mock_aiter

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_stream = AsyncMock()
            mock_stream.__aenter__.return_value = mock_response
            mock_instance.stream.return_value = mock_stream
            mock_client.return_value.__aenter__.return_value = mock_instance

            async for _ in nova_model.stream("Test"):
                pass

            call_args = mock_instance.stream.call_args
            json_data = call_args.kwargs["json"]
            assert "temperature" in json_data
            assert json_data["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_stream_overrides_temperature_with_kwargs(self, nova_model):
        """Test that kwargs override default temperature."""
        mock_response = AsyncMock()
        mock_response.status_code = 200

        async def mock_aiter():
            yield b'data: {"choices":[{"delta":{"content":"Test"}}]}\n\n'
            yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'

        mock_response.aiter_bytes = mock_aiter

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_stream = AsyncMock()
            mock_stream.__aenter__.return_value = mock_response
            mock_instance.stream.return_value = mock_stream
            mock_client.return_value.__aenter__.return_value = mock_instance

            async for _ in nova_model.stream("Test", temperature=0.3):
                pass

            call_args = mock_instance.stream.call_args
            json_data = call_args.kwargs["json"]
            assert json_data["temperature"] == 0.3

    @pytest.mark.asyncio
    async def test_stream_includes_stop_sequences(self, nova_model):
        """Test that stop sequences are included in request."""
        nova_model.stop = ["END", "STOP"]

        mock_response = AsyncMock()
        mock_response.status_code = 200

        async def mock_aiter():
            yield b'data: {"choices":[{"delta":{"content":"Test"}}]}\n\n'
            yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'

        mock_response.aiter_bytes = mock_aiter

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_stream = AsyncMock()
            mock_stream.__aenter__.return_value = mock_response
            mock_instance.stream.return_value = mock_stream
            mock_client.return_value.__aenter__.return_value = mock_instance

            async for _ in nova_model.stream("Test"):
                pass

            call_args = mock_instance.stream.call_args
            json_data = call_args.kwargs["json"]
            assert "stop" in json_data
            assert json_data["stop"] == ["END", "STOP"]

    @pytest.mark.asyncio
    async def test_stream_includes_stream_options(self, nova_model):
        """Test that stream_options are included in request."""
        mock_response = AsyncMock()
        mock_response.status_code = 200

        async def mock_aiter():
            yield b'data: {"choices":[{"delta":{"content":"Test"}}]}\n\n'
            yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'

        mock_response.aiter_bytes = mock_aiter

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_stream = AsyncMock()
            mock_stream.__aenter__.return_value = mock_response
            mock_instance.stream.return_value = mock_stream
            mock_client.return_value.__aenter__.return_value = mock_instance

            async for _ in nova_model.stream("Test"):
                pass

            call_args = mock_instance.stream.call_args
            json_data = call_args.kwargs["json"]
            assert "stream_options" in json_data
            assert json_data["stream_options"]["include_usage"] is True

    @pytest.mark.asyncio
    async def test_stream_with_reasoning_effort(self):
        """Test streaming with reasoning_effort parameter."""
        model = NovaModel(api_key="test-key", reasoning_effort="high")

        mock_response = AsyncMock()
        mock_response.status_code = 200

        async def mock_aiter():
            yield b'data: {"choices":[{"delta":{"content":"Test"}}]}\n\n'
            yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'

        mock_response.aiter_bytes = mock_aiter

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_stream = AsyncMock()
            mock_stream.__aenter__.return_value = mock_response
            mock_instance.stream.return_value = mock_stream
            mock_client.return_value.__aenter__.return_value = mock_instance

            async for _ in model.stream("Test"):
                pass

            call_args = mock_instance.stream.call_args
            json_data = call_args.kwargs["json"]
            assert "reasoning_effort" in json_data
            assert json_data["reasoning_effort"] == "high"

    @pytest.mark.asyncio
    async def test_stream_with_web_search_options(self):
        """Test streaming with web_search_options parameter."""
        web_opts = {"search_context_size": "medium"}
        model = NovaModel(api_key="test-key", web_search_options=web_opts)

        mock_response = AsyncMock()
        mock_response.status_code = 200

        async def mock_aiter():
            yield b'data: {"choices":[{"delta":{"content":"Test"}}]}\n\n'
            yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'

        mock_response.aiter_bytes = mock_aiter

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_stream = AsyncMock()
            mock_stream.__aenter__.return_value = mock_response
            mock_instance.stream.return_value = mock_stream
            mock_client.return_value.__aenter__.return_value = mock_instance

            async for _ in model.stream("Test"):
                pass

            call_args = mock_instance.stream.call_args
            json_data = call_args.kwargs["json"]
            assert "web_search_options" in json_data
            assert json_data["web_search_options"] == web_opts


class TestStreamingEdgeCases:
    """Test edge cases in streaming."""

    @pytest.mark.asyncio
    async def test_stream_handles_malformed_json(self, nova_model):
        """Test streaming handles malformed JSON gracefully."""
        mock_response = AsyncMock()
        mock_response.status_code = 200

        async def mock_aiter():
            yield b'data: {"choices":[{"delta":{"content":"Valid"}}]}\n\n'
            yield b"data: {invalid json}\n\n"
            yield b'data: {"choices":[{"delta":{"content":" text"}}]}\n\n'
            yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'

        mock_response.aiter_bytes = mock_aiter

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_stream = AsyncMock()
            mock_stream.__aenter__.return_value = mock_response
            mock_instance.stream.return_value = mock_stream
            mock_client.return_value.__aenter__.return_value = mock_instance

            events = []
            async for event in nova_model.stream("Test"):
                events.append(event)

            # Should continue processing valid events
            assert len(events) > 0

    @pytest.mark.asyncio
    async def test_stream_handles_empty_chunks(self, nova_model):
        """Test streaming handles empty chunks."""
        mock_response = AsyncMock()
        mock_response.status_code = 200

        async def mock_aiter():
            yield b""
            yield b'data: {"choices":[{"delta":{"content":"Test"}}]}\n\n'
            yield b"\n\n"
            yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'

        mock_response.aiter_bytes = mock_aiter

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_stream = AsyncMock()
            mock_stream.__aenter__.return_value = mock_response
            mock_instance.stream.return_value = mock_stream
            mock_client.return_value.__aenter__.return_value = mock_instance

            events = []
            async for event in nova_model.stream("Test"):
                events.append(event)

            assert len(events) > 0

    @pytest.mark.asyncio
    async def test_stream_handles_unicode_content(self, nova_model):
        """Test streaming handles unicode content correctly."""
        mock_response = AsyncMock()
        mock_response.status_code = 200

        async def mock_aiter():
            yield b'data: {"choices":[{"delta":{"content":"Hello \\u4e16\\u754c"}}]}\n\n'
            yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'

        mock_response.aiter_bytes = mock_aiter

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_stream = AsyncMock()
            mock_stream.__aenter__.return_value = mock_response
            mock_instance.stream.return_value = mock_stream
            mock_client.return_value.__aenter__.return_value = mock_instance

            events = []
            async for event in nova_model.stream("Test"):
                events.append(event)

            assert len(events) > 0
