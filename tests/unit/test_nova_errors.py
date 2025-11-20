"""Comprehensive unit tests for NovaModel error handling."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from strands.types.exceptions import ContextWindowOverflowException, ModelThrottledException

from strands_nova import NovaModel, NovaModelError


@pytest.fixture
def nova_model():
    """Create a NovaModel instance for testing."""
    return NovaModel(api_key="test-api-key", model="nova-premier-v1")


class TestAuthenticationErrors:
    """Test authentication error handling."""

    @pytest.mark.asyncio
    async def test_stream_with_401_error(self, nova_model):
        """Test streaming handles 401 authentication error."""
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
    async def test_stream_with_500_auth_error(self, nova_model):
        """Test streaming handles 500 error with auth failure."""
        mock_response = AsyncMock()
        mock_response.status_code = 500
        mock_response.aread = AsyncMock(return_value=b"<html><body>Login Provider Error</body></html>")

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
    async def test_stream_with_500_non_auth_error(self, nova_model):
        """Test streaming handles 500 error without auth failure."""
        mock_response = AsyncMock()
        mock_response.status_code = 500
        mock_response.aread = AsyncMock(return_value=b"Internal Server Error")

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_stream = AsyncMock()
            mock_stream.__aenter__.return_value = mock_response
            mock_instance.stream.return_value = mock_stream
            mock_client.return_value.__aenter__.return_value = mock_instance

            with pytest.raises(NovaModelError, match="Nova API error"):
                async for _ in nova_model.stream("Test prompt"):
                    pass


class TestModelNotFoundErrors:
    """Test model not found error handling."""

    @pytest.mark.asyncio
    async def test_stream_with_404_error(self, nova_model):
        """Test streaming handles 404 model not found error."""
        mock_response = AsyncMock()
        mock_response.status_code = 404
        mock_response.aread = AsyncMock(return_value=b"Model not found")

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_stream = AsyncMock()
            mock_stream.__aenter__.return_value = mock_response
            mock_instance.stream.return_value = mock_stream
            mock_client.return_value.__aenter__.return_value = mock_instance

            with pytest.raises(NovaModelError, match="not found"):
                async for _ in nova_model.stream("Test prompt"):
                    pass

    @pytest.mark.asyncio
    async def test_stream_404_includes_model_name(self, nova_model):
        """Test that 404 error includes model name in message."""
        mock_response = AsyncMock()
        mock_response.status_code = 404
        mock_response.aread = AsyncMock(return_value=b"Not found")

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_stream = AsyncMock()
            mock_stream.__aenter__.return_value = mock_response
            mock_instance.stream.return_value = mock_stream
            mock_client.return_value.__aenter__.return_value = mock_instance

            with pytest.raises(NovaModelError, match="nova-premier-v1"):
                async for _ in nova_model.stream("Test prompt"):
                    pass


class TestRateLimitErrors:
    """Test rate limiting error handling."""

    @pytest.mark.asyncio
    async def test_stream_with_429_error(self, nova_model):
        """Test streaming handles 429 rate limit error."""
        mock_response = AsyncMock()
        mock_response.status_code = 429
        mock_response.aread = AsyncMock(return_value=b"Rate limit exceeded")

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_stream = AsyncMock()
            mock_stream.__aenter__.return_value = mock_response
            mock_instance.stream.return_value = mock_stream
            mock_client.return_value.__aenter__.return_value = mock_instance

            with pytest.raises(ModelThrottledException, match="rate limit"):
                async for _ in nova_model.stream("Test prompt"):
                    pass

    @pytest.mark.asyncio
    async def test_stream_429_preserves_error_message(self, nova_model):
        """Test that 429 error preserves API error message."""
        error_msg = "Rate limit: 100 requests per minute exceeded"
        mock_response = AsyncMock()
        mock_response.status_code = 429
        mock_response.aread = AsyncMock(return_value=error_msg.encode())

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_stream = AsyncMock()
            mock_stream.__aenter__.return_value = mock_response
            mock_instance.stream.return_value = mock_stream
            mock_client.return_value.__aenter__.return_value = mock_instance

            with pytest.raises(ModelThrottledException, match="100 requests"):
                async for _ in nova_model.stream("Test prompt"):
                    pass


class TestContextWindowErrors:
    """Test context window overflow error handling."""

    @pytest.mark.asyncio
    async def test_stream_with_context_overflow(self, nova_model):
        """Test streaming handles context window overflow."""
        mock_response = AsyncMock()
        mock_response.status_code = 400
        mock_response.aread = AsyncMock(return_value=b"Error: Request exceeds maximum context window")

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_stream = AsyncMock()
            mock_stream.__aenter__.return_value = mock_response
            mock_instance.stream.return_value = mock_stream
            mock_client.return_value.__aenter__.return_value = mock_instance

            with pytest.raises(ContextWindowOverflowException, match="Context window"):
                async for _ in nova_model.stream("Test prompt"):
                    pass

    @pytest.mark.asyncio
    async def test_stream_detects_context_keyword(self, nova_model):
        """Test that context overflow is detected by keyword."""
        mock_response = AsyncMock()
        mock_response.status_code = 400
        mock_response.aread = AsyncMock(return_value=b"context length exceeded")

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_stream = AsyncMock()
            mock_stream.__aenter__.return_value = mock_response
            mock_instance.stream.return_value = mock_stream
            mock_client.return_value.__aenter__.return_value = mock_instance

            with pytest.raises(ContextWindowOverflowException):
                async for _ in nova_model.stream("Test prompt"):
                    pass

    @pytest.mark.asyncio
    async def test_stream_detects_maximum_keyword(self, nova_model):
        """Test detection using 'maximum' keyword."""
        mock_response = AsyncMock()
        mock_response.status_code = 400
        mock_response.aread = AsyncMock(return_value=b"Exceeded maximum tokens")

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_stream = AsyncMock()
            mock_stream.__aenter__.return_value = mock_response
            mock_instance.stream.return_value = mock_stream
            mock_client.return_value.__aenter__.return_value = mock_instance

            with pytest.raises(ContextWindowOverflowException):
                async for _ in nova_model.stream("Test prompt"):
                    pass

    @pytest.mark.asyncio
    async def test_stream_detects_token_limit_keyword(self, nova_model):
        """Test detection using 'token limit' keyword."""
        mock_response = AsyncMock()
        mock_response.status_code = 400
        mock_response.aread = AsyncMock(return_value=b"Request exceeds token limit")

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_stream = AsyncMock()
            mock_stream.__aenter__.return_value = mock_response
            mock_instance.stream.return_value = mock_stream
            mock_client.return_value.__aenter__.return_value = mock_instance

            with pytest.raises(ContextWindowOverflowException):
                async for _ in nova_model.stream("Test prompt"):
                    pass


class TestBadRequestErrors:
    """Test bad request error handling."""

    @pytest.mark.asyncio
    async def test_stream_with_generic_400_error(self, nova_model):
        """Test streaming handles generic 400 bad request."""
        mock_response = AsyncMock()
        mock_response.status_code = 400
        mock_response.aread = AsyncMock(return_value=b"Invalid request parameters")

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_stream = AsyncMock()
            mock_stream.__aenter__.return_value = mock_response
            mock_instance.stream.return_value = mock_stream
            mock_client.return_value.__aenter__.return_value = mock_instance

            with pytest.raises(NovaModelError, match="Bad request"):
                async for _ in nova_model.stream("Test prompt"):
                    pass

    @pytest.mark.asyncio
    async def test_stream_400_preserves_error_details(self, nova_model):
        """Test that 400 error preserves API error details."""
        error_msg = "Invalid parameter: temperature must be between 0 and 1"
        mock_response = AsyncMock()
        mock_response.status_code = 400
        mock_response.aread = AsyncMock(return_value=error_msg.encode())

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_stream = AsyncMock()
            mock_stream.__aenter__.return_value = mock_response
            mock_instance.stream.return_value = mock_stream
            mock_client.return_value.__aenter__.return_value = mock_instance

            with pytest.raises(NovaModelError, match="temperature must be"):
                async for _ in nova_model.stream("Test prompt"):
                    pass


class TestNetworkErrors:
    """Test network error handling."""

    @pytest.mark.asyncio
    async def test_stream_with_timeout(self, nova_model):
        """Test streaming handles timeout errors."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_stream = AsyncMock()
            mock_stream.__aenter__.side_effect = httpx.TimeoutException("Timeout")
            mock_instance.stream.return_value = mock_stream
            mock_client.return_value.__aenter__.return_value = mock_instance

            with pytest.raises(ModelThrottledException, match="timeout"):
                async for _ in nova_model.stream("Test prompt"):
                    pass

    @pytest.mark.asyncio
    async def test_stream_with_network_error(self, nova_model):
        """Test streaming handles network errors."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_stream = AsyncMock()
            mock_stream.__aenter__.side_effect = httpx.NetworkError("Network error")
            mock_instance.stream.return_value = mock_stream
            mock_client.return_value.__aenter__.return_value = mock_instance

            with pytest.raises(NovaModelError, match="Network error"):
                async for _ in nova_model.stream("Test prompt"):
                    pass

    @pytest.mark.asyncio
    async def test_stream_with_request_error(self, nova_model):
        """Test streaming handles generic request errors."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_stream = AsyncMock()
            mock_stream.__aenter__.side_effect = httpx.RequestError("Request failed")
            mock_instance.stream.return_value = mock_stream
            mock_client.return_value.__aenter__.return_value = mock_instance

            with pytest.raises(NovaModelError, match="Request error"):
                async for _ in nova_model.stream("Test prompt"):
                    pass


class TestOtherHTTPErrors:
    """Test handling of other HTTP error codes."""

    @pytest.mark.asyncio
    async def test_stream_with_502_error(self, nova_model):
        """Test streaming handles 502 bad gateway error."""
        mock_response = AsyncMock()
        mock_response.status_code = 502
        mock_response.aread = AsyncMock(return_value=b"Bad Gateway")

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_stream = AsyncMock()
            mock_stream.__aenter__.return_value = mock_response
            mock_instance.stream.return_value = mock_stream
            mock_client.return_value.__aenter__.return_value = mock_instance

            with pytest.raises(NovaModelError, match="502"):
                async for _ in nova_model.stream("Test prompt"):
                    pass

    @pytest.mark.asyncio
    async def test_stream_with_503_error(self, nova_model):
        """Test streaming handles 503 service unavailable error."""
        mock_response = AsyncMock()
        mock_response.status_code = 503
        mock_response.aread = AsyncMock(return_value=b"Service Unavailable")

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_stream = AsyncMock()
            mock_stream.__aenter__.return_value = mock_response
            mock_instance.stream.return_value = mock_stream
            mock_client.return_value.__aenter__.return_value = mock_instance

            with pytest.raises(NovaModelError, match="503"):
                async for _ in nova_model.stream("Test prompt"):
                    pass

    @pytest.mark.asyncio
    async def test_stream_with_unknown_status_code(self, nova_model):
        """Test streaming handles unknown status codes."""
        mock_response = AsyncMock()
        mock_response.status_code = 418  # I'm a teapot
        mock_response.aread = AsyncMock(return_value=b"I'm a teapot")

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_stream = AsyncMock()
            mock_stream.__aenter__.return_value = mock_response
            mock_instance.stream.return_value = mock_stream
            mock_client.return_value.__aenter__.return_value = mock_instance

            with pytest.raises(NovaModelError, match="418"):
                async for _ in nova_model.stream("Test prompt"):
                    pass


class TestStructuredOutputError:
    """Test structured output not implemented error."""

    @pytest.mark.asyncio
    async def test_structured_output_raises_not_implemented(self, nova_model):
        """Test that structured_output raises NotImplementedError."""
        from pydantic import BaseModel

        class OutputModel(BaseModel):
            result: str

        with pytest.raises(NotImplementedError, match="not yet supported"):
            async for _ in nova_model.structured_output(OutputModel, "Test prompt"):
                pass

    @pytest.mark.asyncio
    async def test_structured_output_error_message_mentions_nova(self, nova_model):
        """Test that error message mentions Nova models."""
        from pydantic import BaseModel

        class OutputModel(BaseModel):
            result: str

        with pytest.raises(NotImplementedError, match="Nova models"):
            async for _ in nova_model.structured_output(OutputModel, "Test prompt"):
                pass


class TestErrorMessageQuality:
    """Test quality and clarity of error messages."""

    @pytest.mark.asyncio
    async def test_auth_error_mentions_env_var(self, nova_model):
        """Test that auth error mentions NOVA_API_KEY."""
        mock_response = AsyncMock()
        mock_response.status_code = 401
        mock_response.aread = AsyncMock(return_value=b"Unauthorized")

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_stream = AsyncMock()
            mock_stream.__aenter__.return_value = mock_response
            mock_instance.stream.return_value = mock_stream
            mock_client.return_value.__aenter__.return_value = mock_instance

            with pytest.raises(NovaModelError, match="NOVA_API_KEY"):
                async for _ in nova_model.stream("Test prompt"):
                    pass

    @pytest.mark.asyncio
    async def test_rate_limit_error_is_informative(self, nova_model):
        """Test that rate limit error is clear and actionable."""
        mock_response = AsyncMock()
        mock_response.status_code = 429
        mock_response.aread = AsyncMock(return_value=b"Too many requests")

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_stream = AsyncMock()
            mock_stream.__aenter__.return_value = mock_response
            mock_instance.stream.return_value = mock_stream
            mock_client.return_value.__aenter__.return_value = mock_instance

            try:
                async for _ in nova_model.stream("Test prompt"):
                    pass
            except ModelThrottledException as e:
                assert "Nova API" in str(e)

    @pytest.mark.asyncio
    async def test_context_error_includes_context_info(self, nova_model):
        """Test that context overflow error is informative."""
        mock_response = AsyncMock()
        mock_response.status_code = 400
        mock_response.aread = AsyncMock(return_value=b"Context window exceeded")

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_stream = AsyncMock()
            mock_stream.__aenter__.return_value = mock_response
            mock_instance.stream.return_value = mock_stream
            mock_client.return_value.__aenter__.return_value = mock_instance

            try:
                async for _ in nova_model.stream("Test prompt"):
                    pass
            except ContextWindowOverflowException as e:
                assert "Context window" in str(e)


class TestErrorRecovery:
    """Test error recovery and graceful degradation."""

    @pytest.mark.asyncio
    async def test_partial_stream_failure(self, nova_model):
        """Test handling of stream that fails mid-way."""
        mock_response = AsyncMock()
        mock_response.status_code = 200

        async def mock_aiter():
            yield b'data: {"choices":[{"delta":{"content":"Start"}}]}\n\n'
            # Simulate connection drop or error
            raise httpx.NetworkError("Connection lost")

        mock_response.aiter_bytes = mock_aiter

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_stream = AsyncMock()
            mock_stream.__aenter__.return_value = mock_response
            mock_instance.stream.return_value = mock_stream
            mock_client.return_value.__aenter__.return_value = mock_instance

            with pytest.raises(NovaModelError, match="Network error"):
                async for _ in nova_model.stream("Test prompt"):
                    pass
