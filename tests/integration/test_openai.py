import os
import unittest.mock
import pytest
from strands import Agent, tool
from strands.types.exceptions import (
    ContextWindowOverflowException,
    ModelThrottledException,
)
from amazon_nova import NovaAPIModel
from provider_info import nova

pytestmark = nova.mark


def test_agent_invoke(agent):
    result = agent("What is the time and weather in New York?")
    text = result.message["content"][0]["text"].lower()

    assert all(string in text for string in ["12:00", "sunny"])


@pytest.mark.asyncio
async def test_agent_invoke_async(agent):
    result = await agent.invoke_async("What is the time and weather in New York?")
    text = result.message["content"][0]["text"].lower()

    assert all(string in text for string in ["12:00", "sunny"])


@pytest.mark.asyncio
async def test_agent_stream_async(agent):
    stream = agent.stream_async("What is the time and weather in New York?")
    async for event in stream:
        _ = event

    result = event["result"]
    text = result.message["content"][0]["text"].lower()

    assert all(string in text for string in ["12:00", "sunny"])


def test_agent_structured_output(agent, weather):
    tru_weather = agent.structured_output(
        type(weather), "The time is 12:00 and the weather is sunny"
    )
    exp_weather = weather
    assert tru_weather == exp_weather


@pytest.mark.asyncio
async def test_agent_structured_output_async(agent, weather):
    tru_weather = await agent.structured_output_async(
        type(weather), "The time is 12:00 and the weather is sunny"
    )
    exp_weather = weather
    assert tru_weather == exp_weather


def test_agent_structured_output_model(agent, weather):
    tru_weather = agent(
        "The time is 12:00 and the weather is sunny",
        structured_output_model=type(weather),
    ).structured_output
    exp_weather = weather
    assert tru_weather == exp_weather


def test_invoke_multi_modal_input(agent, yellow_img):
    content = [
        {"text": "Is this image red, blue, or yellow?"},
        {
            "image": {
                "format": "png",
                "source": {
                    "bytes": yellow_img,
                },
            },
        },
    ]
    result = agent(content)
    text = result.message["content"][0]["text"].lower()

    assert "yellow" in text


def test_structured_output_multi_modal_input(agent, yellow_img, yellow_color):
    content = [
        {"text": "Is this image red, blue, or yellow?"},
        {
            "image": {
                "format": "png",
                "source": {
                    "bytes": yellow_img,
                },
            },
        },
    ]
    tru_color = agent.structured_output(type(yellow_color), content)
    exp_color = yellow_color
    assert tru_color == exp_color


def test_structured_output_model_multi_modal_input(agent, yellow_img, yellow_color):
    content = [
        {"text": "Is this image red, blue, or yellow?"},
        {
            "image": {
                "format": "png",
                "source": {
                    "bytes": yellow_img,
                },
            },
        },
    ]
    tru_color = agent(
        content, structured_output_model=type(yellow_color)
    ).structured_output
    exp_color = yellow_color
    assert tru_color == exp_color


@pytest.mark.skip("https://github.com/strands-agents/sdk-python/issues/320")
def test_tool_returning_images(model, yellow_img):
    @tool
    def tool_with_image_return():
        return {
            "status": "success",
            "content": [
                {
                    "image": {
                        "format": "png",
                        "source": {"bytes": yellow_img},
                    }
                },
            ],
        }

    agent = Agent(model, tools=[tool_with_image_return])
    # NOTE - this currently fails with: "Invalid 'messages[3]'. Image URLs are only allowed for messages with role
    # 'user', but this message with role 'tool' contains an image URL."
    # See https://github.com/strands-agents/sdk-python/issues/320 for additional details
    agent("Run the the tool and analyze the image")


@pytest.mark.skip("Need information on model limits")
def test_context_window_overflow_integration():
    """Integration test for context window overflow with Nova."""
    mini_model = NovaAPIModel(
        model_id="nova-micro-v1",
        api_key=os.getenv("NOVA_API_KEY"),
    )

    agent = Agent(model=mini_model)

    long_text = (
        "This text is longer than context window, but short enough to not get caught in token rate limit. "
        * 10000
    )

    with pytest.raises(ContextWindowOverflowException):
        agent(long_text)


@pytest.mark.skip("Need information on API limits")
def test_rate_limit_throttling_integration_no_retries():
    """Integration test for rate limit handling with retries disabled."""
    # Patch the event loop constants to disable retries for this test
    with unittest.mock.patch("strands.event_loop.event_loop.MAX_ATTEMPTS", 1):
        mini_model = NovaAPIModel(
            model_id="nova-micro-v1",
            api_key=os.getenv("NOVA_API_KEY"),
        )
        agent = Agent(model=mini_model)

        # Create a message that's very long to trigger token-per-minute rate limits
        # This should be large enough to exceed TPM limits immediately
        very_long_text = "Really long text " * 20000000

        # This should raise ModelThrottledException without retries
        with pytest.raises(ModelThrottledException) as exc_info:
            agent(very_long_text)

        # Verify it's a rate limit error
        error_message = str(exc_info.value).lower()
        assert "rate limit" in error_message or "tokens per min" in error_message


def test_content_blocks_handling(model):
    """Test that content blocks are handled properly without failures."""
    content = [{"text": "What is 2+2?"}, {"text": "Please be brief."}]

    agent = Agent(model=model, load_tools_from_directory=False)
    result = agent(content)

    assert "4" in result.message["content"][0]["text"]


def test_system_prompt_content_integration(model):
    """Integration test for system_prompt_content parameter."""
    from strands.types.content import SystemContentBlock

    system_prompt_content: list[SystemContentBlock] = [
        {
            "text": "You are a helpful assistant that always responds with 'SYSTEM_TEST_RESPONSE'."
        }
    ]

    agent = Agent(model=model, system_prompt=system_prompt_content)
    result = agent("Hello")

    # The response should contain our specific system prompt instruction
    assert "SYSTEM_TEST_RESPONSE" in result.message["content"][0]["text"]


def test_system_prompt_backward_compatibility_integration(model):
    """Integration test for backward compatibility with system_prompt parameter."""
    system_prompt = (
        "You are a helpful assistant that always responds with 'BACKWARD_COMPAT_TEST'."
    )

    agent = Agent(model=model, system_prompt=system_prompt)
    result = agent("Hello")

    # The response should contain our specific system prompt instruction
    assert "BACKWARD_COMPAT_TEST" in result.message["content"][0]["text"]
