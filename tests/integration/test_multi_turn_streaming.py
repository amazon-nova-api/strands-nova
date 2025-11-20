"""Integration tests for multi-turn conversations with streaming using Strands Agent.

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


# Define test tools
def get_weather(location: str, units: str = "celsius") -> str:
    """Retrieve current weather information for a specified location.

    Args:
        location: The city and state/country, e.g., 'Seattle, WA' or 'London, UK'
        units: Temperature unit preference (celsius or fahrenheit)

    Returns:
        Weather information as a string
    """
    # Mock weather data for testing
    weather_data = {
        "New York": "22 degrees and cloudy",
        "Tokyo": "32 degrees and sunny",
        "London": "18 degrees and rainy",
        "Boston": "20 degrees and partly cloudy",
        "Paris": "15 degrees and windy",
        "San Francisco": "18 degrees and foggy",
    }

    # Simple matching - look for city name in location string
    for city, weather in weather_data.items():
        if city.lower() in location.lower():
            return weather

    return "Weather information not available"


@pytest.mark.integration
@pytest.mark.requires_capability("text")
@pytest.mark.asyncio
async def test_multi_turn_triple_tool_call_streaming(model_id):
    """Test multi-turn conversation with triple tool call using streaming.

    Test case: multi_turn_triple_tool_call_streaming
    Input: Multi-turn conversation asking about weather in multiple cities, then follow-up
    Expected: Agent handles conversation history and responds appropriately with streaming
    """
    # Initialize model with streaming enabled
    model = NovaModel(model=model_id, stream=True)
    agent = Agent(model=model, tools=[get_weather])

    # First turn: Ask about weather in multiple cities
    response1_text = ""
    async for event in agent.stream_async("weather in new york, japan, london?"):
        if "data" in event:
            response1_text += event["data"]

    # Verify first response
    assert len(response1_text) > 0, "First response text is empty"

    # Second turn: Follow-up question (agent maintains conversation history)
    response2_text = ""
    async for event in agent.stream_async("What about boston?"):
        if "data" in event:
            response2_text += event["data"]

    # Verify second response
    assert len(response2_text) > 0, "Second response text is empty"


@pytest.mark.integration
@pytest.mark.requires_capability("text")
@pytest.mark.asyncio
async def test_multi_turn_conversation_streaming(model_id):
    """Test basic multi-turn conversation without tools using streaming.

    Test case: multi_turn_conversation_streaming
    Input: Series of related questions
    Expected: Agent maintains context across turns with streaming
    """
    # Initialize model with streaming enabled
    model = NovaModel(model=model_id, stream=True)
    agent = Agent(model=model)

    # First turn
    response1_text = ""
    async for event in agent.stream_async("What are the three primary colors?"):
        if "data" in event:
            response1_text += event["data"]

    # Verify first response
    assert len(response1_text) > 0, "First response text is empty"

    # Second turn - references first turn
    response2_text = ""
    async for event in agent.stream_async(
        "Can you explain why those are considered primary?"
    ):
        if "data" in event:
            response2_text += event["data"]

    # Verify second response
    assert len(response2_text) > 0, "Second response text is empty"

    # Third turn - continues conversation
    response3_text = ""
    async for event in agent.stream_async("What colors can you make by mixing them?"):
        if "data" in event:
            response3_text += event["data"]

    # Verify third response
    assert len(response3_text) > 0, "Third response text is empty"


@pytest.mark.integration
@pytest.mark.requires_capability("text")
@pytest.mark.asyncio
async def test_multi_turn_with_context_shift_streaming(model_id):
    """Test multi-turn conversation with context shifts using streaming.

    Test case: multi_turn_with_context_shift_streaming
    Input: Questions on different topics
    Expected: Agent handles context shifts appropriately with streaming
    """
    # Initialize model with streaming enabled
    model = NovaModel(model=model_id, stream=True)
    agent = Agent(model=model, tools=[get_weather])

    # First turn: Weather question
    response1_text = ""
    async for event in agent.stream_async("What's the weather in Paris?"):
        if "data" in event:
            response1_text += event["data"]

    # Verify first response
    assert len(response1_text) > 0, "First response text is empty"

    # Second turn: Different topic
    response2_text = ""
    async for event in agent.stream_async("Now tell me about the Eiffel Tower."):
        if "data" in event:
            response2_text += event["data"]

    # Verify second response
    assert len(response2_text) > 0, "Second response text is empty"

    # Third turn: Back to weather
    response3_text = ""
    async for event in agent.stream_async(
        "And what about the weather in San Francisco?"
    ):
        if "data" in event:
            response3_text += event["data"]

    # Verify third response
    assert len(response3_text) > 0, "Third response text is empty"
