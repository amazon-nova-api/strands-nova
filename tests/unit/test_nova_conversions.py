"""Comprehensive unit tests for NovaModel message and tool conversions."""

from strands_nova import NovaModel


class TestMessageConversion:
    """Test _convert_messages_to_nova_format method."""

    def test_convert_simple_string_message(self):
        """Test conversion of simple string message."""
        model = NovaModel(api_key="test-key")
        result = model._convert_messages_to_nova_format("Hello, world!")

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello, world!"

    def test_convert_string_with_system_prompt(self):
        """Test conversion with system prompt."""
        model = NovaModel(api_key="test-key")
        result = model._convert_messages_to_nova_format(
            "User message", system_prompt="You are a helpful assistant"
        )

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are a helpful assistant"
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "User message"

    def test_convert_messages_list_with_dict_format(self):
        """Test conversion of messages list in dict format."""
        model = NovaModel(api_key="test-key")
        messages = [
            {"role": "user", "content": [{"text": "Hello"}]},
            {"role": "assistant", "content": [{"text": "Hi there"}]},
            {"role": "user", "content": [{"text": "How are you?"}]},
        ]
        result = model._convert_messages_to_nova_format(messages)

        assert len(result) == 3
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == "Hi there"
        assert result[2]["role"] == "user"
        assert result[2]["content"] == "How are you?"

    def test_convert_messages_with_multiple_content_blocks(self):
        """Test conversion of messages with multiple content blocks."""
        model = NovaModel(api_key="test-key")
        messages = [
            {
                "role": "user",
                "content": [
                    {"text": "First part"},
                    {"text": "Second part"},
                    {"text": "Third part"},
                ],
            }
        ]
        result = model._convert_messages_to_nova_format(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "First part\nSecond part\nThird part"

    def test_convert_messages_with_string_content_blocks(self):
        """Test conversion with string content blocks instead of dicts."""
        model = NovaModel(api_key="test-key")
        messages = [{"role": "user", "content": ["Hello", "World"]}]
        result = model._convert_messages_to_nova_format(messages)

        assert len(result) == 1
        assert result[0]["content"] == "Hello\nWorld"

    def test_convert_messages_with_simple_string_content(self):
        """Test conversion with simple string content."""
        model = NovaModel(api_key="test-key")
        messages = [{"role": "user", "content": "Simple string"}]
        result = model._convert_messages_to_nova_format(messages)

        assert len(result) == 1
        assert result[0]["content"] == "Simple string"

    def test_convert_empty_content_blocks(self):
        """Test conversion with empty content blocks list."""
        model = NovaModel(api_key="test-key")
        messages = [{"role": "user", "content": []}]
        result = model._convert_messages_to_nova_format(messages)

        assert len(result) == 1
        assert result[0]["content"] == ""

    def test_convert_mixed_roles(self):
        """Test conversion with various role types."""
        model = NovaModel(api_key="test-key")
        messages = [
            {"role": "system", "content": [{"text": "System message"}]},
            {"role": "user", "content": [{"text": "User message"}]},
            {"role": "assistant", "content": [{"text": "Assistant message"}]},
        ]
        result = model._convert_messages_to_nova_format(messages)

        assert len(result) == 3
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"

    def test_convert_with_fallback_for_other_formats(self):
        """Test fallback for messages that don't match expected format."""
        model = NovaModel(api_key="test-key")
        messages = [{"role": "user", "text": "Non-standard format"}]
        result = model._convert_messages_to_nova_format(messages)

        assert len(result) == 1
        assert result[0] == {"role": "user", "text": "Non-standard format"}

    def test_convert_empty_messages_list(self):
        """Test conversion with empty messages list."""
        model = NovaModel(api_key="test-key")
        result = model._convert_messages_to_nova_format([])
        assert len(result) == 0

    def test_convert_with_unicode_characters(self):
        """Test conversion with unicode characters."""
        model = NovaModel(api_key="test-key")
        messages = [{"role": "user", "content": [{"text": "Hello 世界 🌍"}]}]
        result = model._convert_messages_to_nova_format(messages)

        assert result[0]["content"] == "Hello 世界 🌍"

    def test_convert_with_newlines_in_content(self):
        """Test conversion preserving newlines in content."""
        model = NovaModel(api_key="test-key")
        messages = [{"role": "user", "content": [{"text": "Line 1\nLine 2\nLine 3"}]}]
        result = model._convert_messages_to_nova_format(messages)

        assert result[0]["content"] == "Line 1\nLine 2\nLine 3"

    def test_convert_with_special_characters(self):
        """Test conversion with special characters."""
        model = NovaModel(api_key="test-key")
        messages = [{"role": "user", "content": [{"text": "Special: !@#$%^&*()"}]}]
        result = model._convert_messages_to_nova_format(messages)

        assert result[0]["content"] == "Special: !@#$%^&*()"


class TestToolSpecConversion:
    """Test _convert_tool_specs_to_nova_format method."""

    def test_convert_none_tool_specs(self):
        """Test conversion of None tool specs."""
        model = NovaModel(api_key="test-key")
        result = model._convert_tool_specs_to_nova_format(None)
        assert result is None

    def test_convert_empty_tool_specs_list(self):
        """Test conversion of empty tool specs list."""
        model = NovaModel(api_key="test-key")
        result = model._convert_tool_specs_to_nova_format([])
        assert result is None

    def test_convert_single_tool_spec(self):
        """Test conversion of single tool spec."""
        model = NovaModel(api_key="test-key")
        tool_specs = [
            {
                "name": "get_weather",
                "description": "Get current weather",
                "inputSchema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            }
        ]
        result = model._convert_tool_specs_to_nova_format(tool_specs)

        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"
        assert result[0]["function"]["description"] == "Get current weather"
        assert "parameters" in result[0]["function"]
        assert result[0]["function"]["parameters"]["type"] == "object"

    def test_convert_multiple_tool_specs(self):
        """Test conversion of multiple tool specs."""
        model = NovaModel(api_key="test-key")
        tool_specs = [
            {
                "name": "tool1",
                "description": "First tool",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "tool2",
                "description": "Second tool",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "tool3",
                "description": "Third tool",
                "inputSchema": {"type": "object", "properties": {}},
            },
        ]
        result = model._convert_tool_specs_to_nova_format(tool_specs)

        assert len(result) == 3
        assert result[0]["function"]["name"] == "tool1"
        assert result[1]["function"]["name"] == "tool2"
        assert result[2]["function"]["name"] == "tool3"

    def test_convert_tool_with_required_parameters(self):
        """Test conversion of tool with required parameters."""
        model = NovaModel(api_key="test-key")
        tool_specs = [
            {
                "name": "calculator",
                "description": "Perform calculations",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "operation": {"type": "string"},
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                    },
                    "required": ["operation", "x", "y"],
                },
            }
        ]
        result = model._convert_tool_specs_to_nova_format(tool_specs)

        assert result[0]["function"]["parameters"]["required"] == [
            "operation",
            "x",
            "y",
        ]

    def test_convert_tool_with_complex_schema(self):
        """Test conversion of tool with complex input schema."""
        model = NovaModel(api_key="test-key")
        tool_specs = [
            {
                "name": "complex_tool",
                "description": "Complex tool",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "nested": {
                            "type": "object",
                            "properties": {
                                "field1": {"type": "string"},
                                "field2": {"type": "integer"},
                            },
                        },
                        "array_field": {"type": "array", "items": {"type": "string"}},
                    },
                },
            }
        ]
        result = model._convert_tool_specs_to_nova_format(tool_specs)

        parameters = result[0]["function"]["parameters"]
        assert parameters["properties"]["nested"]["type"] == "object"
        assert parameters["properties"]["array_field"]["type"] == "array"

    def test_convert_tool_preserves_all_schema_details(self):
        """Test that conversion preserves all schema details."""
        model = NovaModel(api_key="test-key")
        tool_specs = [
            {
                "name": "detailed_tool",
                "description": "Detailed tool with all properties",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "string_field": {
                            "type": "string",
                            "description": "A string field",
                            "minLength": 1,
                            "maxLength": 100,
                        },
                        "number_field": {
                            "type": "number",
                            "description": "A number field",
                            "minimum": 0,
                            "maximum": 100,
                        },
                    },
                    "required": ["string_field"],
                    "additionalProperties": False,
                },
            }
        ]
        result = model._convert_tool_specs_to_nova_format(tool_specs)

        parameters = result[0]["function"]["parameters"]
        assert parameters["properties"]["string_field"]["minLength"] == 1
        assert parameters["properties"]["number_field"]["minimum"] == 0
        assert parameters["additionalProperties"] is False


class TestToolChoiceFormatting:
    """Test _format_tool_choice method."""

    def test_format_none_tool_choice(self):
        """Test formatting None tool choice."""
        model = NovaModel(api_key="test-key")
        result = model._format_tool_choice(None)
        assert result == "auto"

    def test_format_auto_tool_choice(self):
        """Test formatting auto tool choice."""
        model = NovaModel(api_key="test-key")
        result = model._format_tool_choice({"auto": {}})
        assert result == "auto"

    def test_format_any_tool_choice(self):
        """Test formatting any tool choice (converts to 'required')."""
        model = NovaModel(api_key="test-key")
        result = model._format_tool_choice({"any": {}})
        assert result == "required"

    def test_format_specific_tool_choice(self):
        """Test formatting specific tool choice."""
        model = NovaModel(api_key="test-key")
        result = model._format_tool_choice({"tool": {"name": "my_tool"}})

        assert isinstance(result, dict)
        assert result["type"] == "function"
        assert result["function"]["name"] == "my_tool"

    def test_format_tool_choice_with_different_tool_names(self):
        """Test formatting specific tool choice with various tool names."""
        model = NovaModel(api_key="test-key")

        for tool_name in ["get_weather", "calculator", "search_web", "tool123"]:
            result = model._format_tool_choice({"tool": {"name": tool_name}})
            assert result["function"]["name"] == tool_name

    def test_format_unknown_tool_choice_defaults_to_auto(self):
        """Test that unknown tool choice format defaults to auto."""
        model = NovaModel(api_key="test-key")
        result = model._format_tool_choice({"unknown": {}})
        assert result == "auto"

    def test_format_malformed_tool_choice(self):
        """Test handling of malformed tool choice."""
        model = NovaModel(api_key="test-key")
        result = model._format_tool_choice({"tool": {}})
        assert result == "auto"

    def test_format_empty_dict_tool_choice(self):
        """Test formatting empty dict tool choice."""
        model = NovaModel(api_key="test-key")
        result = model._format_tool_choice({})
        assert result == "auto"


class TestConversionEdgeCases:
    """Test edge cases in conversions."""

    def test_message_conversion_with_none_in_content_blocks(self):
        """Test handling of None values in content blocks."""
        model = NovaModel(api_key="test-key")
        messages = [
            {
                "role": "user",
                "content": [{"text": "Valid"}, None, {"text": "Also valid"}],
            }
        ]
        # Should handle gracefully without crashing
        result = model._convert_messages_to_nova_format(messages)
        assert len(result) > 0

    def test_tool_spec_conversion_with_minimal_schema(self):
        """Test tool conversion with minimal schema."""
        model = NovaModel(api_key="test-key")
        tool_specs = [
            {"name": "simple_tool", "description": "Simple", "inputSchema": {}}
        ]
        result = model._convert_tool_specs_to_nova_format(tool_specs)

        assert len(result) == 1
        assert result[0]["function"]["parameters"] == {}

    def test_message_conversion_preserves_message_order(self):
        """Test that message conversion preserves order."""
        model = NovaModel(api_key="test-key")
        messages = [
            {"role": "user", "content": [{"text": "Message 1"}]},
            {"role": "assistant", "content": [{"text": "Message 2"}]},
            {"role": "user", "content": [{"text": "Message 3"}]},
            {"role": "assistant", "content": [{"text": "Message 4"}]},
        ]
        result = model._convert_messages_to_nova_format(messages)

        assert len(result) == 4
        assert result[0]["content"] == "Message 1"
        assert result[1]["content"] == "Message 2"
        assert result[2]["content"] == "Message 3"
        assert result[3]["content"] == "Message 4"

    def test_tool_spec_conversion_preserves_order(self):
        """Test that tool spec conversion preserves order."""
        model = NovaModel(api_key="test-key")
        tool_specs = [
            {"name": "tool_a", "description": "A", "inputSchema": {}},
            {"name": "tool_b", "description": "B", "inputSchema": {}},
            {"name": "tool_c", "description": "C", "inputSchema": {}},
        ]
        result = model._convert_tool_specs_to_nova_format(tool_specs)

        assert result[0]["function"]["name"] == "tool_a"
        assert result[1]["function"]["name"] == "tool_b"
        assert result[2]["function"]["name"] == "tool_c"

    def test_message_with_very_long_content(self):
        """Test conversion with very long message content."""
        model = NovaModel(api_key="test-key")
        long_text = "A" * 10000
        messages = [{"role": "user", "content": [{"text": long_text}]}]
        result = model._convert_messages_to_nova_format(messages)

        assert len(result[0]["content"]) == 10000

    def test_tool_with_very_long_description(self):
        """Test tool conversion with very long description."""
        model = NovaModel(api_key="test-key")
        long_description = "This is a very detailed description. " * 100
        tool_specs = [
            {
                "name": "detailed_tool",
                "description": long_description,
                "inputSchema": {},
            }
        ]
        result = model._convert_tool_specs_to_nova_format(tool_specs)

        assert result[0]["function"]["description"] == long_description

    def test_message_conversion_with_numeric_content(self):
        """Test handling of numeric content in messages."""
        model = NovaModel(api_key="test-key")
        messages = [{"role": "user", "content": 12345}]
        result = model._convert_messages_to_nova_format(messages)

        assert result[0]["content"] == "12345"

    def test_message_conversion_with_boolean_content(self):
        """Test handling of boolean content in messages."""
        model = NovaModel(api_key="test-key")
        messages = [{"role": "user", "content": True}]
        result = model._convert_messages_to_nova_format(messages)

        assert result[0]["content"] == "True"
