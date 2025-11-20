"""Integration tests for basic message handling using Strands Agent.

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


@pytest.mark.integration
def test_text_basic():
    """Test basic text message handling through Strands Agent.

    Test case: text_basic
    Input: Simple user message asking to list three colors of the rainbow
    Expected: Assistant responds with text content
    """
    # Initialize model and agent
    model = NovaModel(model=get_model_id())
    agent = Agent(model=model)

    # Test input
    user_message = "List three colors of the rainbow."

    # Call agent
    response = agent(user_message)

    # Verify response structure
    assert response is not None, "No response received"
    assert hasattr(response, "message"), "Response has no message attribute"
    assert response.message is not None, "Response message is None"
    assert len(response.message) > 0, "Response message is empty"


@pytest.mark.integration
def test_temperature_parameter():
    """Test temperature parameter affects response randomness.

    Test case: temperature_parameter
    Input: Same prompt with different temperature values
    Expected: Model accepts different temperature values
    """
    user_message = "Say hello."

    model_medium = NovaModel(model=get_model_id(), temperature=0.5)
    agent_medium = Agent(model=model_medium)
    response_medium = agent_medium(user_message)

    assert response_medium is not None
    assert hasattr(response_medium, "message")
    assert len(response_medium.message) > 0


@pytest.mark.integration
def test_max_tokens_parameter():
    """Test max_tokens parameter limits response length.

    Test case: max_tokens_parameter
    Input: Request for longer response with max_tokens limit
    Expected: Response respects token limit
    """
    user_message = "Write a very short sentence."

    model_long = NovaModel(model=get_model_id(), max_tokens=100)
    agent_long = Agent(model=model_long)
    response_long = agent_long(user_message)

    assert response_long is not None
    assert hasattr(response_long, "message")
    assert len(response_long.message) > 0


@pytest.mark.integration
def test_top_p_parameter():
    """Test top_p (nucleus sampling) parameter.

    Test case: top_p_parameter
    Input: Same prompt with different top_p values
    Expected: Model accepts different top_p values
    """
    user_message = "Name a color."

    # Test with high top_p (more diverse)
    model_diverse = NovaModel(model=get_model_id(), top_p=0.9)
    agent_diverse = Agent(model=model_diverse)
    response_diverse = agent_diverse(user_message)

    assert response_diverse is not None
    assert hasattr(response_diverse, "message")
    assert len(response_diverse.message) > 0


@pytest.mark.integration
def test_stop_sequences():
    """Test stop sequences parameter.

    Test case: stop_sequences
    Input: Request with custom stop sequences
    Expected: Model stops generation at specified sequences
    """
    user_message = "Count: one, two, three, four, five"

    # Test with stop sequence
    model_with_stop = NovaModel(model=get_model_id(), stop=["three"])
    agent_with_stop = Agent(model=model_with_stop)
    response_with_stop = agent_with_stop(user_message)

    assert response_with_stop is not None
    assert hasattr(response_with_stop, "message")
    assert len(response_with_stop.message) > 0

    # Test with multiple stop sequences
    model_multi_stop = NovaModel(model=get_model_id(), stop=["two", "four"])
    agent_multi_stop = Agent(model=model_multi_stop)
    response_multi_stop = agent_multi_stop(user_message)

    assert response_multi_stop is not None
    assert hasattr(response_multi_stop, "message")
    assert len(response_multi_stop.message) > 0


@pytest.mark.integration
def test_combined_parameters():
    """Test using multiple parameters together.

    Test case: combined_parameters
    Input: Request with temperature, max_tokens, and top_p combined
    Expected: All parameters work together correctly
    """
    user_message = "Write a very short sentence."

    # Test with combined parameters
    model_combined = NovaModel(
        model=get_model_id(),
        temperature=0.3,
        max_tokens=100,
        top_p=0.8,
    )
    agent_combined = Agent(model=model_combined)
    response_combined = agent_combined(user_message)

    assert response_combined is not None
    assert hasattr(response_combined, "message")
    assert len(response_combined.message) > 0
