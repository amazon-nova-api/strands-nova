"""Integration tests for tool calling with streaming using Strands Agent.

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
@pytest.mark.requires_capability("text")
@pytest.mark.asyncio
async def test_tool_single_streaming(model_id):
    """Test single tool call through Strands Agent with streaming.

    Test case: tool_single_streaming
    Input: User asks about weather in Paris
    Expected: Agent calls get_weather tool with city="Paris" and streams response
    """
    # Initialize model with streaming enabled
    model = NovaModel(model=model_id, stream=True)
    agent = Agent(model=model, tools=[get_weather])

    # Test input
    user_message = "What's the weather in Paris?"

    # Collect streaming response
    response_text = ""
    async for event in agent.stream_async(user_message):
        if "data" in event:
            response_text += event["data"]

    # Verify response
    assert len(response_text) > 0, "Response text is empty"


@pytest.mark.integration
@pytest.mark.requires_capability("text")
@pytest.mark.asyncio
async def test_tool_multi_streaming(model_id):
    """Test multiple tool calls through Strands Agent with streaming.

    Test case: tool_multi_streaming
    Input: User asks to perform multiple operations
    Expected: Agent calls multiple tools and streams response
    """
    # Initialize model with streaming enabled
    model = NovaModel(model=model_id, stream=True)
    agent = Agent(model=model, tools=[get_weather, calculate])

    # Test input
    user_message = "What's 25 + 17? Also, what's the weather in London?"

    # Collect streaming response
    response_text = ""
    async for event in agent.stream_async(user_message):
        if "data" in event:
            response_text += event["data"]

    # Verify response
    assert len(response_text) > 0, "Response text is empty"


@pytest.mark.integration
@pytest.mark.requires_capability("text")
@pytest.mark.asyncio
async def test_tool_result_streaming(model_id):
    """Test tool result handling through Strands Agent with streaming.

    Test case: tool_result_streaming
    Input: Previous tool call result in conversation
    Expected: Agent processes tool result and streams response
    """
    # Initialize model with streaming enabled
    model = NovaModel(model=model_id, stream=True)
    agent = Agent(model=model, tools=[get_weather])

    # First call to trigger tool use
    response1_text = ""
    async for event in agent.stream_async("What's the weather in Paris?"):
        if "data" in event:
            response1_text += event["data"]

    # Verify first response
    assert len(response1_text) > 0, "First response text is empty"

    # Follow-up question about the result
    response2_text = ""
    async for event in agent.stream_async("Is that warm or cold?"):
        if "data" in event:
            response2_text += event["data"]

    # Verify follow-up response
    assert len(response2_text) > 0, "Follow-up response text is empty"


@pytest.mark.integration
@pytest.mark.requires_capability("text")
@pytest.mark.asyncio
async def test_chained_tool_call_streaming(model_id):
    """Test chained tool calls through Strands Agent with streaming.

    Test case: chained_tool_call_streaming
    Input: User request that requires sequential tool calls
    Expected: Agent calls tools in sequence based on previous results with streaming
    """
    # Initialize model with streaming enabled
    model = NovaModel(model=model_id, stream=True)
    agent = Agent(model=model, tools=[search_database, calculate])

    # Test input that requires chained operations
    user_message = "Search for 'python tutorials' and tell me how many results there are, then multiply that by 2"

    # Collect streaming response
    response_text = ""
    async for event in agent.stream_async(user_message):
        if "data" in event:
            response_text += event["data"]

    # Verify response
    assert len(response_text) > 0, "Response text is empty"


@pytest.mark.integration
@pytest.mark.requires_capability("text")
@pytest.mark.asyncio
async def test_triple_tool_call_streaming(model_id):
    """Test triple tool calls through Strands Agent with streaming.

    Test case: triple_tool_call_streaming
    Input: User request requiring three tool calls
    Expected: Agent calls three tools to complete the request with streaming
    """
    # Initialize model with streaming enabled
    model = NovaModel(model=model_id, stream=True)
    agent = Agent(model=model, tools=[get_weather, calculate, search_database])

    # Test input requiring multiple tools
    user_message = "What's the weather in Tokyo? Also calculate 100 / 4, and search for 'AI models'"

    # Collect streaming response
    response_text = ""
    async for event in agent.stream_async(user_message):
        if "data" in event:
            response_text += event["data"]

    # Verify response
    assert len(response_text) > 0, "Response text is empty"


@pytest.mark.integration
@pytest.mark.requires_capability("text")
@pytest.mark.asyncio
async def test_multi_tool_result_streaming(model_id):
    """Test multiple tool results handling through Strands Agent with streaming.

    Test case: multi_tool_result_streaming
    Input: Conversation with multiple tool results
    Expected: Agent processes multiple tool results correctly with streaming
    """
    # Initialize model with streaming enabled
    model = NovaModel(model=model_id, stream=True)
    agent = Agent(model=model, tools=[get_weather, calculate])

    # Make request requiring multiple tools
    response1_text = ""
    async for event in agent.stream_async(
        "Get weather for both Paris and London, then add 10 + 20"
    ):
        if "data" in event:
            response1_text += event["data"]

    # Verify response
    assert len(response1_text) > 0, "First response text is empty"

    # Follow-up using those results
    response2_text = ""
    async for event in agent.stream_async(
        "Which city is warmer and what's the sum you calculated?"
    ):
        if "data" in event:
            response2_text += event["data"]

    # Verify follow-up response
    assert len(response2_text) > 0, "Follow-up response text is empty"
