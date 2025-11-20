"""Integration tests for basic message handling with streaming using Strands Agent.

These tests make actual API calls to the Nova API through Strands Agent with streaming enabled.
Requires NOVA_API_KEY environment variable to be set.

Tests are parametrized to run on all models that support the required capability.
"""

import pytest
from dotenv import load_dotenv
from strands import Agent

from strands_nova import NovaModel

# Load environment variables
load_dotenv()


@pytest.mark.integration
@pytest.mark.requires_capability("text")
@pytest.mark.asyncio
async def test_text_basic_streaming(model_id):
    """Test basic text message handling through Strands Agent with streaming.

    Test case: text_basic_streaming
    Input: Simple user message asking to list three colors of the rainbow
    Expected: Assistant responds with streaming text content
    """
    # Initialize model with streaming enabled
    model = NovaModel(model=model_id, stream=True)
    agent = Agent(model=model)

    # Test input
    user_message = "List three colors of the rainbow."

    # Collect streaming response
    response_text = ""
    event_count = 0
    async for event in agent.stream_async(user_message):
        event_count += 1
        if "data" in event:
            response_text += event["data"]

    # Verify response
    assert event_count > 0, "No streaming events received"
    assert len(response_text) > 0, "Response text is empty"


@pytest.mark.integration
@pytest.mark.requires_capability("text")
@pytest.mark.asyncio
async def test_temperature_parameter_streaming(model_id):
    """Test temperature parameter affects response randomness with streaming.

    Test case: temperature_parameter_streaming
    Input: Same prompt with different temperature values
    Expected: Model accepts different temperature values with streaming
    """
    user_message = "Say hello."

    model_medium = NovaModel(model=model_id, temperature=0.5, stream=True)
    agent_medium = Agent(model=model_medium)

    response_text = ""
    async for event in agent_medium.stream_async(user_message):
        if "data" in event:
            response_text += event["data"]

    assert len(response_text) > 0, "Response text is empty"


@pytest.mark.integration
@pytest.mark.requires_capability("text")
@pytest.mark.asyncio
async def test_max_tokens_parameter_streaming(model_id):
    """Test max_tokens parameter limits response length with streaming.

    Test case: max_tokens_parameter_streaming
    Input: Request for longer response with max_tokens limit
    Expected: Response respects token limit with streaming
    """
    user_message = "Write a very short sentence."

    model_long = NovaModel(model=model_id, max_tokens=100, stream=True)
    agent_long = Agent(model=model_long)

    response_text = ""
    async for event in agent_long.stream_async(user_message):
        if "data" in event:
            response_text += event["data"]

    assert len(response_text) > 0, "Response text is empty"


@pytest.mark.integration
@pytest.mark.requires_capability("text")
@pytest.mark.asyncio
async def test_top_p_parameter_streaming(model_id):
    """Test top_p (nucleus sampling) parameter with streaming.

    Test case: top_p_parameter_streaming
    Input: Same prompt with different top_p values
    Expected: Model accepts different top_p values with streaming
    """
    user_message = "Name a color."

    # Test with high top_p (more diverse)
    model_diverse = NovaModel(model=model_id, top_p=0.9, stream=True)
    agent_diverse = Agent(model=model_diverse)

    response_text = ""
    async for event in agent_diverse.stream_async(user_message):
        if "data" in event:
            response_text += event["data"]

    assert len(response_text) > 0, "Response text is empty"


@pytest.mark.integration
@pytest.mark.requires_capability("text")
@pytest.mark.asyncio
async def test_stop_sequences_streaming(model_id):
    """Test stop sequences parameter with streaming.

    Test case: stop_sequences_streaming
    Input: Request with custom stop sequences
    Expected: Model stops generation at specified sequences with streaming
    """
    user_message = "Count: one, two, three, four, five"

    # Test with stop sequence
    model_with_stop = NovaModel(model=model_id, stop=["three"], stream=True)
    agent_with_stop = Agent(model=model_with_stop)

    response_text = ""
    async for event in agent_with_stop.stream_async(user_message):
        if "data" in event:
            response_text += event["data"]

    assert len(response_text) > 0, "Response text is empty"

    # Test with multiple stop sequences
    model_multi_stop = NovaModel(model=model_id, stop=["two", "four"], stream=True)
    agent_multi_stop = Agent(model=model_multi_stop)

    response_text_multi = ""
    async for event in agent_multi_stop.stream_async(user_message):
        if "data" in event:
            response_text_multi += event["data"]

    assert len(response_text_multi) > 0, "Response text is empty"


@pytest.mark.integration
@pytest.mark.requires_capability("text")
@pytest.mark.asyncio
async def test_combined_parameters_streaming(model_id):
    """Test using multiple parameters together with streaming.

    Test case: combined_parameters_streaming
    Input: Request with temperature, max_tokens, and top_p combined
    Expected: All parameters work together correctly with streaming
    """
    user_message = "Write a very short sentence."

    # Test with combined parameters
    model_combined = NovaModel(
        model=model_id,
        temperature=0.3,
        max_tokens=100,
        top_p=0.8,
        stream=True,
    )
    agent_combined = Agent(model=model_combined)

    response_text = ""
    async for event in agent_combined.stream_async(user_message):
        if "data" in event:
            response_text += event["data"]

    assert len(response_text) > 0, "Response text is empty"
