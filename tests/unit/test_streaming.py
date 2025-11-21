"""Unit tests for NovaModel streaming functionality."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from strands.types.exceptions import ContextWindowOverflowException, ModelThrottledException
from strands_nova.nova import NovaModel


class TestChunkFormatting:
    """Test formatting of response chunks."""

    def test_format_chunk_message_start(self):
        """Test formatting message_start chunk."""
        model = NovaModel(model_id="nova-pro-v1", api_key="test-key")
        event = {"chunk_type": "message_start"}
        
        result = model.format_chunk(event)
        
        assert result == {"messageStart": {"role": "assistant"}}

    def test_format_chunk_content_start_text(self):
        """Test formatting content_start chunk for text."""
        model = NovaModel(model_id="nova-pro-v1", api_key="test-key")
        event = {"chunk_type": "content_start", "data_type": "text"}
        
        result = model.format_chunk(event)
        
        assert result == {"contentBlockStart": {"start": {}}}

    def test_format_chunk_content_start_tool(self):
        """Test formatting content_start chunk for tool."""
        model = NovaModel(model_id="nova-pro-v1", api_key="test-key")
        event = {
            "chunk_type": "content_start",
            "data_type": "tool",
            "data": {"name": "get_weather", "id": "tool_123"}
        }
        
        result = model.format_chunk(event)
        
        assert result == {
            "contentBlockStart": {
                "start": {
                    "toolUse": {
                        "name": "get_weather",
                        "toolUseId": "tool_123"
                    }
                }
            }
        }

    def test_format_chunk_content_delta_text(self):
        """Test formatting content_delta chunk for text."""
        model = NovaModel(model_id="nova-pro-v1", api_key="test-key")
        event = {"chunk_type": "content_delta", "data_type": "text", "data": "Hello"}
        
        result = model.format_chunk(event)
        
        assert result == {"contentBlockDelta": {"delta": {"text": "Hello"}}}

    def test_format_chunk_content_delta_tool(self):
        """Test formatting content_delta chunk for tool."""
        model = NovaModel(model_id="nova-pro-v1", api_key="test-key")
        event = {
            "chunk_type": "content_delta",
            "data_type": "tool",
            "data": {"arguments": '{"city": "NYC"}'}
        }
        
        result = model.format_chunk(event)
        
        assert result == {
            "contentBlockDelta": {
                "delta": {
                    "toolUse": {"input": '{"city": "NYC"}'}
                }
            }
        }

    def test_format_chunk_content_delta_reasoning(self):
        """Test formatting content_delta chunk for reasoning content."""
        model = NovaModel(model_id="nova-pro-v1", api_key="test-key")
        event = {
            "chunk_type": "content_delta",
            "data_type": "reasoning_content",
            "data": "Thinking..."
        }
        
        result = model.format_chunk(event)
        
        assert result == {
            "contentBlockDelta": {
                "delta": {
                    "reasoningContent": {"text": "Thinking..."}
                }
            }
        }

    def test_format_chunk_content_stop(self):
        """Test formatting content_stop chunk."""
        model = NovaModel(model_id="nova-pro-v1", api_key="test-key")
        event = {"chunk_type": "content_stop"}
        
        result = model.format_chunk(event)
        
        assert result == {"contentBlockStop": {}}

    def test_format_chunk_message_stop_tool_calls(self):
        """Test formatting message_stop chunk with tool_calls finish reason."""
        model = NovaModel(model_id="nova-pro-v1", api_key="test-key")
        event = {"chunk_type": "message_stop", "data": "tool_calls"}
        
        result = model.format_chunk(event)
        
        assert result == {"messageStop": {"stopReason": "tool_use"}}

    def test_format_chunk_message_stop_length(self):
        """Test formatting message_stop chunk with length finish reason."""
        model = NovaModel(model_id="nova-pro-v1", api_key="test-key")
        event = {"chunk_type": "message_stop", "data": "length"}
        
        result = model.format_chunk(event)
        
        assert result == {"messageStop": {"stopReason": "max_tokens"}}

    def test_format_chunk_message_stop_default(self):
        """Test formatting message_stop chunk with default finish reason."""
        model = NovaModel(model_id="nova-pro-v1", api_key="test-key")
        event = {"chunk_type": "message_stop", "data": "stop"}
        
        result = model.format_chunk(event)
        
        assert result == {"messageStop": {"stopReason": "end_turn"}}

    def test_format_chunk_metadata(self):
        """Test formatting metadata chunk."""
        model = NovaModel(model_id="nova-pro-v1", api_key="test-key")
        event = {
            "chunk_type": "metadata",
            "data": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            }
        }
        
        result = model.format_chunk(event)
        
        assert result["metadata"]["usage"]["inputTokens"] == 100
        assert result["metadata"]["usage"]["outputTokens"] == 50
        assert result["metadata"]["usage"]["totalTokens"] == 150

    def test_format_chunk_unknown_type_raises_error(self):
        """Test formatting unknown chunk type raises RuntimeError."""
        model = NovaModel(model_id="nova-pro-v1", api_key="test-key")
        event = {"chunk_type": "unknown_type"}
        
        with pytest.raises(RuntimeError, match="unknown type"):
            model.format_chunk(event)


class TestStreamSwitchContent:
    """Test content stream switching logic."""

    def test_stream_switch_content_same_type(self):
        """Test switching to same content type."""
        model = NovaModel(model_id="nova-pro-v1", api_key="test-key")
        
        chunks, data_type = model._stream_switch_content("text", "text")
        
        assert chunks == []
        assert data_type == "text"

    def test_stream_switch_content_different_type(self):
        """Test switching to different content type."""
        model = NovaModel(model_id="nova-pro-v1", api_key="test-key")
        
        chunks, data_type = model._stream_switch_content("text", "tool")
        
        assert len(chunks) == 2
        assert chunks[0] == {"contentBlockStop": {}}
        assert chunks[1] == {"contentBlockStart": {"start": {}}}
        assert data_type == "text"

    def test_stream_switch_content_from_none(self):
        """Test switching from None to content type."""
        model = NovaModel(model_id="nova-pro-v1", api_key="test-key")
        
        chunks, data_type = model._stream_switch_content("text", None)
        
        assert len(chunks) == 1
        assert chunks[0] == {"contentBlockStart": {"start": {}}}
        assert data_type == "text"


class TestSSEStreamParsing:
    """Test Server-Sent Events stream parsing."""

    @pytest.mark.asyncio
    async def test_parse_sse_stream_valid_data(self):
        """Test parsing valid SSE stream."""
        model = NovaModel(model_id="nova-pro-v1", api_key="test-key")
        
        # Mock response with SSE data
        async def mock_aiter_lines():
            lines = [
                "data: " + json.dumps({"event": "test1"}),
                "",
                "data: " + json.dumps({"event": "test2"}),
                "data: [DONE]"
            ]
            for line in lines:
                yield line
        
        mock_response = MagicMock()
        mock_response.aiter_lines = mock_aiter_lines
        
        events = []
        async for event in model._parse_sse_stream(mock_response):
            events.append(event)
        
        assert len(events) == 2
        assert events[0] == {"event": "test1"}
        assert events[1] == {"event": "test2"}

    @pytest.mark.asyncio
    async def test_parse_sse_stream_invalid_json(self):
        """Test parsing SSE stream with invalid JSON."""
        model = NovaModel(model_id="nova-pro-v1", api_key="test-key")
        
        async def mock_aiter_lines():
            lines = [
                "data: invalid json",
                "data: " + json.dumps({"event": "valid"}),
            ]
            for line in lines:
                yield line
        
        mock_response = MagicMock()
        mock_response.aiter_lines = mock_aiter_lines
        
        events = []
        async for event in model._parse_sse_stream(mock_response):
            events.append(event)
        
        # Invalid JSON should be skipped
        assert len(events) == 1
        assert events[0] == {"event": "valid"}

    @pytest.mark.asyncio
    async def test_parse_sse_stream_empty_lines(self):
        """Test parsing SSE stream with empty lines."""
        model = NovaModel(model_id="nova-pro-v1", api_key="test-key")
        
        async def mock_aiter_lines():
            lines = [
                "",
                "data: " + json.dumps({"event": "test"}),
                "",
                "data: [DONE]"
            ]
            for line in lines:
                yield line
        
        mock_response = MagicMock()
        mock_response.aiter_lines = mock_aiter_lines
        
        events = []
        async for event in model._parse_sse_stream(mock_response):
            events.append(event)
        
        assert len(events) == 1


class TestErrorHandling:
    """Test error handling in streaming."""

    def test_handle_api_error_400_context_window(self):
        """Test handling 400 error for context window overflow."""
        model = NovaModel(model_id="nova-pro-v1", api_key="test-key")
        
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "error": {"message": "context length exceeded"}
        }
        
        with pytest.raises(ContextWindowOverflowException):
            model._handle_api_error(mock_response)

    def test_handle_api_error_400_validation(self):
        """Test handling 400 validation error."""
        model = NovaModel(model_id="nova-pro-v1", api_key="test-key")
        
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "error": {"message": "invalid parameter"}
        }
        
        with pytest.raises(Exception, match="validation error"):
            model._handle_api_error(mock_response)

    def test_handle_api_error_404(self):
        """Test handling 404 model not found error."""
        model = NovaModel(model_id="nova-pro-v1", api_key="test-key")
        
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.json.return_value = {
            "error": {"message": "model not found"}
        }
        
        with pytest.raises(Exception, match="model not found"):
            model._handle_api_error(mock_response)

    def test_handle_api_error_429(self):
        """Test handling 429 throttling error."""
        model = NovaModel(model_id="nova-pro-v1", api_key="test-key")
        
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.json.return_value = {
            "error": {"message": "rate limit exceeded"}
        }
        
        with pytest.raises(ModelThrottledException):
            model._handle_api_error(mock_response)

    def test_handle_api_error_500(self):
        """Test handling 500 model error."""
        model = NovaModel(model_id="nova-pro-v1", api_key="test-key")
        
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.return_value = {
            "error": {"message": "internal server error"}
        }
        
        with pytest.raises(Exception, match="model error"):
            model._handle_api_error(mock_response)

    def test_handle_api_error_json_parse_failure(self):
        """Test handling error when JSON parsing fails."""
        model = NovaModel(model_id="nova-pro-v1", api_key="test-key")
        
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.side_effect = Exception("JSON parse error")
        mock_response.text = "Raw error text"
        
        with pytest.raises(Exception, match="Raw error text"):
            model._handle_api_error(mock_response)


class TestStreamingIntegration:
    """Integration tests for streaming functionality."""

    @pytest.mark.asyncio
    async def test_stream_non_streaming_mode(self):
        """Test streaming with non-streaming mode."""
        model = NovaModel(model_id="nova-pro-v1", api_key="test-key", stream=False)
        messages = [{"role": "user", "content": [{"text": "Hello"}]}]
        
        mock_response_data = {
            "choices": [{
                "message": {"content": "Hi there"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_stream_context = MagicMock()
            mock_stream_context.__aenter__ = AsyncMock()
            mock_stream_context.__aexit__ = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.aread = AsyncMock(return_value=json.dumps(mock_response_data).encode())
            mock_stream_context.__aenter__.return_value = mock_response
            
            mock_client_instance = MagicMock()
            mock_client_instance.stream.return_value = mock_stream_context
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock()
            mock_client.return_value = mock_client_instance
            
            chunks = []
            async for chunk in model.stream(messages):
                chunks.append(chunk)
            
            # Verify chunks
            assert any("messageStart" in chunk for chunk in chunks)
            assert any("contentBlockDelta" in chunk and chunk["contentBlockDelta"]["delta"].get("text") == "Hi there" for chunk in chunks)
            assert any("messageStop" in chunk for chunk in chunks)

    @pytest.mark.asyncio
    async def test_stream_timeout_error(self):
        """Test handling timeout error during streaming."""
        model = NovaModel(model_id="nova-pro-v1", api_key="test-key")
        messages = [{"role": "user", "content": [{"text": "Hello"}]}]
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_client_instance = MagicMock()
            mock_client_instance.stream = MagicMock(side_effect=httpx.TimeoutException("Timeout"))
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value = mock_client_instance
            
            with pytest.raises(httpx.TimeoutException):
                async for _ in model.stream(messages):
                    pass

    @pytest.mark.asyncio
    async def test_stream_request_error(self):
        """Test handling request error during streaming."""
        model = NovaModel(model_id="nova-pro-v1", api_key="test-key")
        messages = [{"role": "user", "content": [{"text": "Hello"}]}]
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_client_instance = MagicMock()
            mock_client_instance.stream = MagicMock(side_effect=httpx.RequestError("Network error"))
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client.return_value = mock_client_instance
            
            with pytest.raises(httpx.RequestError):
                async for _ in model.stream(messages):
                    pass
