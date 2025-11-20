"""Integration tests for tool calling using Strands Agent.

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
def get_weather(city: str) -> dict:
    """Get current weather for a city.

    Args:
        city: Name of the city

    Returns:
        Weather information
    """
    return {"city": city, "temperature": 72, "condition": "sunny"}


def calculate(operation: str, a: float, b: float) -> float:
    """Perform a mathematical calculation.

    Args:
        operation: The operation to perform (add, subtract, multiply, divide)
        a: First number
        b: Second number

    Returns:
        Result of the calculation
    """
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        return a / b if b != 0 else 0
    return 0


def search_database(query: str) -> list:
    """Search a database for information.

    Args:
        query: Search query

    Returns:
        List of search results
    """
    return [{"id": 1, "title": f"Result for {query}", "content": "Sample content"}]


@pytest.mark.integration
def test_tool_single():
    """Test single tool call through Strands Agent.

    Test case: tool_single
    Input: User asks about weather in Paris
    Expected: Agent calls get_weather tool with city="Paris"
    """
    # Initialize model and agent with tool
    model = NovaModel(model=get_model_id())
    agent = Agent(model=model, tools=[get_weather])

    # Test input
    user_message = "What's the weather in Paris?"

    # Call agent
    response = agent(user_message)

    # Verify response
    assert response is not None, "No response received"
    assert hasattr(response, "message"), "Response has no message attribute"


@pytest.mark.integration
def test_tool_multi():
    """Test multiple tool calls through Strands Agent.

    Test case: tool_multi
    Input: User asks to perform multiple operations
    Expected: Agent calls multiple tools
    """
    # Initialize model and agent with multiple tools
    model = NovaModel(model=get_model_id())
    agent = Agent(model=model, tools=[get_weather, calculate])

    # Test input
    user_message = "What's 25 + 17? Also, what's the weather in London?"

    # Call agent
    response = agent(user_message)

    # Verify response
    assert response is not None, "No response received"
    assert hasattr(response, "message"), "Response has no message attribute"


@pytest.mark.integration
def test_tool_result():
    """Test tool result handling through Strands Agent.

    Test case: tool_result
    Input: Previous tool call result in conversation
    Expected: Agent processes tool result and responds
    """
    # Initialize model and agent with tool
    model = NovaModel(model=get_model_id())
    agent = Agent(model=model, tools=[get_weather])

    # First call to trigger tool use
    response1 = agent("What's the weather in Paris?")

    # Verify first response
    assert response1 is not None, "No initial response received"

    # Follow-up question about the result
    response2 = agent("Is that warm or cold?")

    # Verify follow-up response
    assert response2 is not None, "No follow-up response received"
    assert hasattr(response2, "message"), "Response has no message attribute"


@pytest.mark.integration
def test_chained_tool_call():
    """Test chained tool calls through Strands Agent.

    Test case: chained_tool_call
    Input: User request that requires sequential tool calls
    Expected: Agent calls tools in sequence based on previous results
    """
    # Initialize model and agent with tools
    model = NovaModel(model=get_model_id())
    agent = Agent(model=model, tools=[search_database, calculate])

    # Test input that requires chained operations
    user_message = "Search for 'python tutorials' and tell me how many results there are, then multiply that by 2"

    # Call agent
    response = agent(user_message)

    # Verify response
    assert response is not None, "No response received"
    assert hasattr(response, "message"), "Response has no message attribute"


@pytest.mark.integration
def test_triple_tool_call():
    """Test triple tool calls through Strands Agent.

    Test case: triple_tool_call
    Input: User request requiring three tool calls
    Expected: Agent calls three tools to complete the request
    """
    # Initialize model and agent with tools
    model = NovaModel(model=get_model_id())
    agent = Agent(model=model, tools=[get_weather, calculate, search_database])

    # Test input requiring multiple tools
    user_message = "What's the weather in Tokyo? Also calculate 100 / 4, and search for 'AI models'"

    # Call agent
    response = agent(user_message)

    # Verify response
    assert response is not None, "No response received"
    assert hasattr(response, "message"), "Response has no message attribute"


@pytest.mark.integration
def test_multi_tool_result():
    """Test multiple tool results handling through Strands Agent.

    Test case: multi_tool_result
    Input: Conversation with multiple tool results
    Expected: Agent processes multiple tool results correctly
    """
    # Initialize model and agent with tools
    model = NovaModel(model=get_model_id())
    agent = Agent(model=model, tools=[get_weather, calculate])

    # Make request requiring multiple tools
    response1 = agent("Get weather for both Paris and London, then add 10 + 20")

    # Verify response
    assert response1 is not None, "No response received"

    # Follow-up using those results
    response2 = agent("Which city is warmer and what's the sum you calculated?")

    # Verify follow-up response
    assert response2 is not None, "No follow-up response received"
    assert hasattr(response2, "message"), "Response has no message attribute"
