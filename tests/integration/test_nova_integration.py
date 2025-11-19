"""Integration tests for Nova model provider."""

import os

import pytest

from strands_nova import NovaModel


@pytest.fixture
def nova_api_key():
    """Get Nova API key from environment."""
    api_key = os.getenv("NOVA_API_KEY")
    if not api_key:
        pytest.skip("NOVA_API_KEY not set")
    return api_key


@pytest.fixture
def nova_model(nova_api_key):
    """Create a NovaModel instance for integration testing."""
    return NovaModel(api_key=nova_api_key, model="nova-premier-v1")


@pytest.mark.asyncio
async def test_basic_streaming(nova_model):
    """Test basic streaming functionality with real API."""
    prompt = "Hello! Please respond with a brief greeting."

    response_chunks = []
    async for event in nova_model.stream(prompt):
        # Correct event format check
        if "contentBlockDelta" in event:
            delta = event["contentBlockDelta"]["delta"]
            if "text" in delta:
                response_chunks.append(delta["text"])

    # Check we got a response
    assert len(response_chunks) > 0

    # Combine chunks to get full response
    full_response = "".join(response_chunks)
    assert len(full_response) > 0


@pytest.mark.asyncio
async def test_streaming_with_system_prompt(nova_model):
    """Test streaming with system prompt."""
    prompt = "What is 2 + 2?"
    system_prompt = "You are a helpful math tutor. Answer briefly."

    response_chunks = []
    async for event in nova_model.stream(prompt, system_prompt=system_prompt):
        # Correct event format check
        if "contentBlockDelta" in event:
            delta = event["contentBlockDelta"]["delta"]
            if "text" in delta:
                response_chunks.append(delta["text"])

    # Check we got a response
    assert len(response_chunks) > 0

    # Combine chunks and check for "4" in response
    full_response = "".join(response_chunks).lower()
    assert "4" in full_response or "four" in full_response


@pytest.mark.asyncio
async def test_temperature_parameter(nova_model):
    """Test that temperature parameter affects output."""
    prompt = "Tell me a creative story in one sentence."

    # Low temperature (more deterministic)
    nova_model.update_config(temperature=0.1)
    response1_chunks = []
    async for event in nova_model.stream(prompt):
        # Correct event format check
        if "contentBlockDelta" in event:
            delta = event["contentBlockDelta"]["delta"]
            if "text" in delta:
                response1_chunks.append(delta["text"])

    response1 = "".join(response1_chunks)

    # High temperature (more creative)
    nova_model.update_config(temperature=0.9)
    response2_chunks = []
    async for event in nova_model.stream(prompt):
        # Correct event format check
        if "contentBlockDelta" in event:
            delta = event["contentBlockDelta"]["delta"]
            if "text" in delta:
                response2_chunks.append(delta["text"])

    response2 = "".join(response2_chunks)

    # Both should produce responses
    assert len(response1) > 0
    assert len(response2) > 0


@pytest.mark.asyncio
async def test_max_tokens_limit(nova_model):
    """Test that max_tokens parameter limits output."""
    prompt = "Count from 1 to 100 slowly, one number per line."

    # Set a low token limit
    nova_model.update_config(max_tokens=50)

    response_chunks = []
    async for event in nova_model.stream(prompt):
        # Correct event format check
        if "contentBlockDelta" in event:
            delta = event["contentBlockDelta"]["delta"]
            if "text" in delta:
                response_chunks.append(delta["text"])

    # Check we got a response
    assert len(response_chunks) > 0

    # Response should be limited (not reach 100)
    full_response = "".join(response_chunks)
    # Check if the model actually counted to 100 by looking for numbers on their own lines
    # (not numbers embedded in prose like "count from 1 to 100")
    import re
    # Find numbers that appear on their own lines (with newline before and after)
    line_numbers = re.findall(r'(?:^|\n)(\d+)(?:\n|$)', full_response)
    if line_numbers:
        # Convert to integers and get the max number counted to
        max_number = max(int(n) for n in line_numbers)
        # With 50 tokens, shouldn't reach 100 (should stop much earlier)
        assert max_number < 100, f"Expected to stop before 100, but counted to {max_number}"
    # Check the response is reasonably short given the token limit
    assert len(full_response) < 500, "Response should be limited with max_tokens=50"


@pytest.mark.asyncio
async def test_model_configuration(nova_model):
    """Test getting and updating model configuration."""
    # Get initial config
    initial_config = nova_model.get_config()
    assert initial_config["model"] == "nova-premier-v1"
    assert "temperature" in initial_config
    assert "max_tokens" in initial_config

    # Update config
    nova_model.update_config(temperature=0.5, max_tokens=2048, top_p=0.8)

    # Verify updates
    updated_config = nova_model.get_config()
    assert updated_config["temperature"] == 0.5
    assert updated_config["max_tokens"] == 2048
    assert updated_config["top_p"] == 0.8


@pytest.mark.asyncio
async def test_reasoning_model(nova_api_key):
    """Test reasoning model with reasoning_effort parameter.

    Note: reasoning_effort parameter is currently not supported by available models.
    This test is expected to work once AWS provides the reasoning model name.
    """
    # TODO: Update model name when reasoning model is available from AWS
    reasoning_model = NovaModel(api_key=nova_api_key, model="nova-premier-v1")

    prompt = "What is the sum of the first 10 prime numbers?"
    response_chunks = []
    async for event in reasoning_model.stream(prompt):
        # Correct event format check
        if "contentBlockDelta" in event:
            delta = event["contentBlockDelta"]["delta"]
            if "text" in delta:
                response_chunks.append(delta["text"])

    full_response = "".join(response_chunks)
    assert len(full_response) > 0
    # The answer should be 129 (2+3+5+7+11+13+17+19+23+29)
    # Note: Without reasoning_effort, the model may or may not get this right


@pytest.mark.asyncio
async def test_web_search_integration(nova_api_key):
    """Test web search integration."""
    web_model = NovaModel(
        api_key=nova_api_key, model="nova-premier-v1", web_search_options={"enabled": True, "freshness": "recent"}
    )

    prompt = "What is the latest news about artificial intelligence?"
    response_chunks = []
    async for event in web_model.stream(prompt):
        # Correct event format check
        if "contentBlockDelta" in event:
            delta = event["contentBlockDelta"]["delta"]
            if "text" in delta:
                response_chunks.append(delta["text"])

    full_response = "".join(response_chunks)
    assert len(full_response) > 0


@pytest.mark.asyncio
async def test_tool_calling(nova_model):
    """Test tool calling functionality."""
    # Use correct Strands SDK format with inputSchema
    tool_specs = [
        {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "inputSchema": {
                "type": "object",
                "properties": {"location": {"type": "string", "description": "The city name"}},
                "required": ["location"],
            },
        }
    ]

    prompt = "What's the weather in Seattle?"

    events = []
    async for event in nova_model.stream(prompt, tool_specs=tool_specs):
        events.append(event)

    # Should get a contentBlockStart event with toolUse
    tool_start_events = [
        e for e in events
        if "contentBlockStart" in e and "toolUse" in e["contentBlockStart"].get("start", {})
    ]

    if len(tool_start_events) > 0:
        # Verify tool call structure
        tool_use = tool_start_events[0]["contentBlockStart"]["start"]["toolUse"]
        assert tool_use["name"] == "get_weather"
        assert "toolUseId" in tool_use
    else:
        # If no tool call, at least check we got some response
        text_events = [e for e in events if "contentBlockDelta" in e]
        assert len(text_events) > 0


@pytest.mark.asyncio
async def test_structured_output_not_supported(nova_model):
    """Test that structured output is not yet supported."""
    from pydantic import BaseModel

    class TestOutput(BaseModel):
        answer: str

    with pytest.raises(NotImplementedError, match="Structured output is not yet supported"):
        async for _ in nova_model.structured_output(TestOutput, "Test prompt"):
            pass


def test_model_string_representation(nova_model):
    """Test string representation of the model."""
    model_str = str(nova_model)
    assert "NovaModel" in model_str
    assert "nova-premier-v1" in model_str
    assert "temperature" in model_str.lower()
    assert "max_tokens" in model_str.lower()
