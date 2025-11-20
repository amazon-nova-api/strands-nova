"""Integration tests for multi-turn conversations using Strands Agent.

These tests make actual API calls to the Nova API through Strands Agent.
Requires NOVA_API_KEY environment variable to be set.
"""

import os

import pytest
from dotenv import load_dotenv
from strands import Agent

from strands_nova import NovaModel

# Load environment variables
load_dotenv()


def get_model_id() -> str:
    """Get the model ID from environment or use default."""
    return os.getenv("NOVA_MODEL_ID", "nova-premier-v1")


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
def test_multi_turn_triple_tool_call():
    """Test multi-turn conversation with triple tool call.

    Test case: multi_turn_triple_tool_call
    Input: Multi-turn conversation asking about weather in multiple cities, then follow-up
    Expected: Agent handles conversation history and responds appropriately
    """
    # Initialize model and agent with tool
    model = NovaModel(model=get_model_id())
    agent = Agent(model=model, tools=[get_weather])

    # First turn: Ask about weather in multiple cities
    response1 = agent("weather in new york, japan, london?")

    # Verify first response
    assert response1 is not None, "No response received for first turn"
    assert hasattr(response1, "message"), "First response has no message attribute"

    # Second turn: Follow-up question (agent maintains conversation history)
    response2 = agent("What about boston?")

    # Verify second response
    assert response2 is not None, "No response received for second turn"
    assert hasattr(response2, "message"), "Second response has no message attribute"
    assert response2.message is not None, "Second response message is None"


@pytest.mark.integration
def test_multi_turn_conversation():
    """Test basic multi-turn conversation without tools.

    Test case: General multi-turn conversation
    Input: Series of related questions
    Expected: Agent maintains context across turns
    """
    # Initialize model and agent
    model = NovaModel(model=get_model_id())
    agent = Agent(model=model)

    # First turn
    response1 = agent("What are the three primary colors?")

    # Verify first response
    assert response1 is not None, "No response received for first turn"
    assert hasattr(response1, "message"), "First response has no message attribute"

    # Second turn - references first turn
    response2 = agent("Can you explain why those are considered primary?")

    # Verify second response
    assert response2 is not None, "No response received for second turn"
    assert hasattr(response2, "message"), "Second response has no message attribute"

    # Third turn - continues conversation
    response3 = agent("What colors can you make by mixing them?")

    # Verify third response
    assert response3 is not None, "No response received for third turn"
    assert hasattr(response3, "message"), "Third response has no message attribute"


@pytest.mark.integration
def test_multi_turn_with_context_shift():
    """Test multi-turn conversation with context shifts.

    Test case: Multi-turn with topic changes
    Input: Questions on different topics
    Expected: Agent handles context shifts appropriately
    """
    # Initialize model and agent with tool
    model = NovaModel(model=get_model_id())
    agent = Agent(model=model, tools=[get_weather])

    # First turn: Weather question
    response1 = agent("What's the weather in Paris?")

    # Verify first response
    assert response1 is not None, "No response received for first turn"

    # Second turn: Different topic
    response2 = agent("Now tell me about the Eiffel Tower.")

    # Verify second response
    assert response2 is not None, "No response received for second turn"
    assert hasattr(response2, "message"), "Second response has no message attribute"

    # Third turn: Back to weather
    response3 = agent("And what about the weather in San Francisco?")

    # Verify third response
    assert response3 is not None, "No response received for third turn"
    assert hasattr(response3, "message"), "Third response has no message attribute"
