"""Pytest configuration and fixtures for unit tests."""

import pytest


@pytest.fixture
def mock_api_key():
    """Provide a mock API key for testing."""
    return "test-api-key-12345"


@pytest.fixture
def sample_messages():
    """Provide sample messages for testing."""
    return [
        {"role": "user", "content": [{"text": "Hello, how are you?"}]},
        {"role": "assistant", "content": [{"text": "I'm doing well, thank you!"}]},
    ]


@pytest.fixture
def sample_tool_spec():
    """Provide a sample tool specification for testing."""
    return {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city name"}
                },
                "required": ["location"],
            }
        },
    }


@pytest.fixture
def sample_system_prompt():
    """Provide a sample system prompt for testing."""
    return "You are a helpful AI assistant."
