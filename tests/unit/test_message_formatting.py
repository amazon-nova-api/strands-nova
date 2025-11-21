"""Unit tests for NovaModel message formatting."""

import base64

import pytest

from strands_nova.nova import NovaModel


class TestContentBlockFormatting:
    """Test formatting of individual content blocks."""

    def test_format_text_content(self):
        """Test formatting text content block."""
        content = {"text": "Hello, world!"}
        
        result = NovaModel.format_request_message_content(content)
        
        assert result == {"text": "Hello, world!", "type": "text"}

    def test_format_image_content(self):
        """Test formatting image content block."""
        image_bytes = b"fake_image_data"
        content = {
            "image": {
                "format": "png",
                "source": {"bytes": image_bytes}
            }
        }
        
        result = NovaModel.format_request_message_content(content)
        
        expected_b64 = base64.b64encode(image_bytes).decode("utf-8")
        assert result["type"] == "image_url"
        assert result["image_url"]["url"] == f"data:image/png;base64,{expected_b64}"

    def test_format_document_content(self):
        """Test formatting document content block."""
        doc_bytes = b"fake_pdf_data"
        content = {
            "document": {
                "format": "pdf",
                "name": "test.pdf",
                "source": {"bytes": doc_bytes}
            }
        }
        
        result = NovaModel.format_request_message_content(content)
        
        expected_b64 = base64.b64encode(doc_bytes).decode("utf-8")
        assert result["type"] == "file"
        assert result["file"]["filename"] == "test.pdf"
        assert result["file"]["file_data"] == f"data:application/pdf;base64,{expected_b64}"

    def test_format_audio_content(self):
        """Test formatting audio content block."""
        audio_bytes = b"fake_audio_data"
        content = {
            "audio": {
                "format": "mp3",
                "source": {"bytes": audio_bytes}
            }
        }
        
        result = NovaModel.format_request_message_content(content)
        
        expected_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        assert result["type"] == "input_audio"
        assert result["input_audio"]["data"] == expected_b64
        assert result["input_audio"]["format"] == "mp3"

    def test_format_unsupported_content_raises_error(self):
        """Test formatting unsupported content type raises TypeError."""
        content = {"unknown_type": "data"}
        
        with pytest.raises(TypeError, match="unsupported type"):
            NovaModel.format_request_message_content(content)


class TestToolFormatting:
    """Test formatting of tool-related messages."""

    def test_format_tool_call(self):
        """Test formatting tool use as tool call."""
        tool_use = {
            "toolUseId": "tool_123",
            "name": "get_weather",
            "input": {"city": "San Francisco"}
        }
        
        result = NovaModel.format_request_message_tool_call(tool_use)
        
        assert result["type"] == "function"
        assert result["id"] == "tool_123"
        assert result["function"]["name"] == "get_weather"
        assert result["function"]["arguments"] == '{"city": "San Francisco"}'

    def test_format_tool_result_with_text(self):
        """Test formatting tool result with text content."""
        tool_result = {
            "toolUseId": "tool_123",
            "content": [
                {"text": "Weather is sunny"}
            ]
        }
        
        result = NovaModel.format_request_tool_message(tool_result)
        
        assert result["role"] == "tool"
        assert result["tool_call_id"] == "tool_123"
        assert len(result["content"]) == 1
        assert result["content"][0]["text"] == "Weather is sunny"

    def test_format_tool_result_with_json(self):
        """Test formatting tool result with JSON content."""
        tool_result = {
            "toolUseId": "tool_456",
            "content": [
                {"json": {"temperature": 72, "conditions": "sunny"}}
            ]
        }
        
        result = NovaModel.format_request_tool_message(tool_result)
        
        assert result["role"] == "tool"
        assert result["tool_call_id"] == "tool_456"
        assert len(result["content"]) == 1
        # JSON should be converted to text
        assert "temperature" in result["content"][0]["text"]


class TestToolChoiceFormatting:
    """Test formatting of tool choice configurations."""

    def test_format_tool_choice_auto(self):
        """Test formatting auto tool choice."""
        result = NovaModel._format_request_tool_choice({"auto": {}})
        
        assert result == {"tool_choice": "auto"}

    def test_format_tool_choice_any(self):
        """Test formatting any/required tool choice."""
        result = NovaModel._format_request_tool_choice({"any": {}})
        
        assert result == {"tool_choice": "required"}

    def test_format_tool_choice_specific_tool(self):
        """Test formatting specific tool choice."""
        result = NovaModel._format_request_tool_choice({"tool": {"name": "get_weather"}})
        
        assert result == {
            "tool_choice": {
                "type": "function",
                "function": {"name": "get_weather"}
            }
        }

    def test_format_tool_choice_none(self):
        """Test formatting None tool choice."""
        result = NovaModel._format_request_tool_choice(None)
        
        assert result == {}

    def test_format_tool_choice_unknown(self):
        """Test formatting unknown tool choice defaults to auto."""
        result = NovaModel._format_request_tool_choice({"unknown": {}})
        
        assert result == {"tool_choice": "auto"}


class TestSystemMessageFormatting:
    """Test formatting of system messages."""

    def test_format_system_messages_with_prompt(self):
        """Test formatting system messages from prompt string."""
        result = NovaModel._format_system_messages(system_prompt="You are a helpful assistant")
        
        assert len(result) == 1
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are a helpful assistant"

    def test_format_system_messages_with_content_blocks(self):
        """Test formatting system messages from content blocks."""
        system_content = [
            {"text": "You are a helpful assistant"},
            {"text": "Always be polite"}
        ]
        
        result = NovaModel._format_system_messages(system_prompt_content=system_content)
        
        assert len(result) == 2
        assert result[0]["content"] == "You are a helpful assistant"
        assert result[1]["content"] == "Always be polite"

    def test_format_system_messages_empty(self):
        """Test formatting with no system messages."""
        result = NovaModel._format_system_messages()
        
        assert result == []

    def test_format_system_messages_non_text_ignored(self):
        """Test that non-text system content is ignored."""
        system_content = [
            {"text": "Hello"},
            {"image": {"format": "png", "source": {"bytes": b"data"}}}
        ]
        
        result = NovaModel._format_system_messages(system_prompt_content=system_content)
        
        assert len(result) == 1
        assert result[0]["content"] == "Hello"


class TestRegularMessageFormatting:
    """Test formatting of regular conversation messages."""

    def test_format_regular_messages_simple_text(self):
        """Test formatting simple text messages."""
        messages = [
            {"role": "user", "content": [{"text": "Hello"}]},
            {"role": "assistant", "content": [{"text": "Hi there"}]}
        ]
        
        result = NovaModel._format_regular_messages(messages)
        
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[0]["content"][0]["text"] == "Hello"
        assert result[1]["role"] == "assistant"
        assert result[1]["content"][0]["text"] == "Hi there"

    def test_format_regular_messages_with_tool_use(self):
        """Test formatting messages with tool use."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"text": "Let me check the weather"},
                    {
                        "toolUse": {
                            "toolUseId": "tool_123",
                            "name": "get_weather",
                            "input": {"city": "NYC"}
                        }
                    }
                ]
            }
        ]
        
        result = NovaModel._format_regular_messages(messages)
        
        assert len(result) == 1
        assert "tool_calls" in result[0]
        assert len(result[0]["tool_calls"]) == 1
        assert result[0]["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_format_regular_messages_with_tool_result(self):
        """Test formatting messages with tool results."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "tool_123",
                            "content": [{"text": "It's sunny"}]
                        }
                    }
                ]
            }
        ]
        
        result = NovaModel._format_regular_messages(messages)
        
        # Tool results should be formatted as separate tool messages
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "tool"
        assert result[1]["tool_call_id"] == "tool_123"

    def test_format_regular_messages_filters_reasoning_content(self):
        """Test that reasoning content is filtered out."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"text": "Answer"},
                    {"reasoningContent": {"text": "Thinking..."}}
                ]
            }
        ]
        
        result = NovaModel._format_regular_messages(messages)
        
        # reasoningContent should be filtered
        assert len(result[0]["content"]) == 1
        assert result[0]["content"][0]["text"] == "Answer"


class TestFullRequestFormatting:
    """Test complete request formatting."""

    def test_format_request_messages_complete(self):
        """Test formatting complete message array."""
        messages = [
            {"role": "user", "content": [{"text": "Hello"}]}
        ]
        system_prompt = "You are helpful"
        
        result = NovaModel.format_request_messages(messages, system_prompt)
        
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"

    def test_format_request_complete(self):
        """Test formatting complete request."""
        model = NovaModel(model_id="nova-pro-v1", api_key="test-key")
        messages = [{"role": "user", "content": [{"text": "Hello"}]}]
        tool_specs = [{
            "name": "get_weather",
            "description": "Get weather",
            "inputSchema": {"json": {"type": "object", "properties": {}}}
        }]
        
        result = model.format_request(messages, tool_specs, "You are helpful")
        
        assert result["model"] == "nova-pro-v1"
        assert result["stream"] is True
        assert "messages" in result
        assert len(result["tools"]) == 1
        assert result["tools"][0]["function"]["name"] == "get_weather"
        assert result["stream_options"] == {"include_usage": True}

    def test_format_request_with_tool_choice(self):
        """Test formatting request with tool choice."""
        model = NovaModel(model_id="nova-pro-v1", api_key="test-key")
        messages = [{"role": "user", "content": [{"text": "Hello"}]}]
        tool_choice = {"any": {}}
        
        result = model.format_request(messages, tool_choice=tool_choice)
        
        assert result["tool_choice"] == "required"

    def test_format_request_with_params(self):
        """Test formatting request with model parameters."""
        params = {"temperature": 0.7, "max_tokens": 1000}
        model = NovaModel(model_id="nova-pro-v1", api_key="test-key", params=params)
        messages = [{"role": "user", "content": [{"text": "Hello"}]}]
        
        result = model.format_request(messages)
        
        assert result["temperature"] == 0.7
        assert result["max_tokens"] == 1000

    def test_format_request_filters_empty_messages(self):
        """Test that empty messages are filtered out."""
        model = NovaModel(model_id="nova-pro-v1", api_key="test-key")
        messages = [
            {"role": "user", "content": [{"text": "Hello"}]},
            {"role": "assistant", "content": []}  # Empty content
        ]
        
        result = model.format_request(messages)
        
        # Empty message should be filtered
        assert len([m for m in result["messages"] if m["role"] == "assistant"]) == 0
