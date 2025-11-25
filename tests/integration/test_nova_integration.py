"""Integration tests for Nova model provider.

Tests follow the Agent-based pattern and are categorized by model capabilities:
- TEXT: nova-micro-v1
- MULTIMODAL: nova-lite-v1, nova-premier-v1, nova-pro-v1
- RESEARCH: nova-deep-research-v1, ragentic-cloaked-v1
- WEB SEARCH: nova-premier-v1, nova-pro-v1
- IMAGE GENERATION: nova-orchestrator-v1 (not tested here)
"""

import os
import pytest
from strands import Agent
from strands_nova import NovaModel
from provider_info import nova

pytestmark = nova.mark


@pytest.mark.requires_capability("multimodal")
def test_text_agent_invoke(model_id):
    """Test basic agent invocation with text models."""
    model = NovaModel(model_id=model_id, api_key=os.getenv("NOVA_API_KEY"))
    agent = Agent(model=model, load_tools_from_directory=False)

    result = agent("What is 2+2? Reply with just the number.")
    text = result.message["content"][0]["text"]

    assert "4" in text


@pytest.mark.requires_capability("multimodal")
@pytest.mark.asyncio
async def test_text_agent_invoke_async(model_id):
    """Test async agent invocation with text models."""
    model = NovaModel(model_id=model_id, api_key=os.getenv("NOVA_API_KEY"))
    agent = Agent(model=model, load_tools_from_directory=False)

    result = await agent.invoke_async("What is 2+2? Reply with just the number.")
    text = result.message["content"][0]["text"]

    assert "4" in text


@pytest.mark.requires_capability("multimodal")
@pytest.mark.asyncio
async def test_text_agent_stream_async(model_id):
    """Test async streaming with text models."""
    model = NovaModel(model_id=model_id, api_key=os.getenv("NOVA_API_KEY"))
    agent = Agent(model=model, load_tools_from_directory=False)

    stream = agent.stream_async("What is 2+2? Reply with just the number.")
    async for event in stream:
        _ = event

    result = event["result"]
    text = result.message["content"][0]["text"]

    assert "4" in text


@pytest.mark.requires_capability("multimodal")
def test_model_with_temperature_parameter(model_id):
    """Test models with temperature parameter."""
    model = NovaModel(
        model_id=model_id,
        api_key=os.getenv("NOVA_API_KEY"),
        params={"temperature": 0.1},
    )
    agent = Agent(model=model, load_tools_from_directory=False)

    result = agent("Count from 1 to 5, separated by commas.")
    text = result.message["content"][0]["text"]

    assert "1" in text and "5" in text


@pytest.mark.requires_capability("multimodal")
def test_model_with_max_tokens_parameter(model_id):
    """Test models with max_tokens parameter."""
    model = NovaModel(
        model_id=model_id,
        api_key=os.getenv("NOVA_API_KEY"),
        params={"max_completion_tokens": 100},
    )
    agent = Agent(model=model, load_tools_from_directory=False)

    result = agent("Write a short essay about artificial intelligence in 20 words.")
    text = result.message["content"][0]["text"]

    # Response should be short due to token limit
    assert len(text.split()) < 50


@pytest.mark.requires_capability("multimodal")
def test_text_agent_with_tools(model_id, tools):
    """Test agent with tools using text models."""
    model = NovaModel(model_id=model_id, api_key=os.getenv("NOVA_API_KEY"))
    agent = Agent(model=model, tools=tools)

    result = agent("What is the time and weather in New York?")
    text = result.message["content"][0]["text"].lower()

    assert all(string in text for string in ["12:00", "sunny"])


@pytest.mark.requires_capability("multimodal")
def test_text_agent_structured_output(model_id, weather):
    """Test structured output with text models."""
    model = NovaModel(model_id=model_id, api_key=os.getenv("NOVA_API_KEY"))
    agent = Agent(model=model, load_tools_from_directory=False)

    tru_weather = agent.structured_output(
        type(weather), "The time is 12:00 and the weather is sunny"
    )
    exp_weather = weather
    assert tru_weather == exp_weather


@pytest.mark.requires_capability("multimodal")
@pytest.mark.asyncio
async def test_text_agent_structured_output_async(model_id, weather):
    """Test async structured output with text models."""
    model = NovaModel(model_id=model_id, api_key=os.getenv("NOVA_API_KEY"))
    agent = Agent(model=model, load_tools_from_directory=False)

    tru_weather = await agent.structured_output_async(
        type(weather), "The time is 12:00 and the weather is sunny"
    )
    exp_weather = weather
    assert tru_weather == exp_weather


@pytest.mark.requires_capability("multimodal")
def test_text_agent_system_prompt(model_id):
    """Test system prompt with text models."""
    model = NovaModel(model_id=model_id, api_key=os.getenv("NOVA_API_KEY"))
    system_prompt = (
        "You are a helpful assistant that always responds with 'SYSTEM_TEST_RESPONSE'."
    )
    agent = Agent(
        model=model, system_prompt=system_prompt, load_tools_from_directory=False
    )

    result = agent("Hello")

    assert "SYSTEM_TEST_RESPONSE" in result.message["content"][0]["text"]


@pytest.mark.requires_capability("multimodal")
def test_text_content_blocks_handling(model_id):
    """Test that content blocks are handled properly with text models."""
    model = NovaModel(model_id=model_id, api_key=os.getenv("NOVA_API_KEY"))
    content = [{"text": "What is 2+2?"}, {"text": "Please be brief."}]

    agent = Agent(model=model, load_tools_from_directory=False)
    result = agent(content)

    assert "4" in result.message["content"][0]["text"]


@pytest.mark.requires_capability("multimodal")
def test_multimodal_image_input(model_id, yellow_img):
    """Test image input with multimodal models."""
    model = NovaModel(model_id=model_id, api_key=os.getenv("NOVA_API_KEY"))
    agent = Agent(model=model, load_tools_from_directory=False)

    content = [
        {"text": "Is this image red, blue, or yellow?"},
        {
            "image": {
                "format": "png",
                "source": {"bytes": yellow_img},
            },
        },
    ]
    result = agent(content)
    text = result.message["content"][0]["text"].lower()

    assert "yellow" in text


@pytest.mark.requires_capability("multimodal")
def test_multimodal_structured_output_with_image(model_id, yellow_img, yellow_color):
    """Test structured output with image input for multimodal models."""
    model = NovaModel(model_id=model_id, api_key=os.getenv("NOVA_API_KEY"))
    agent = Agent(model=model, load_tools_from_directory=False)

    content = [
        {"text": "Is this image red, blue, or yellow?"},
        {
            "image": {
                "format": "png",
                "source": {"bytes": yellow_img},
            },
        },
    ]
    tru_color = agent.structured_output(type(yellow_color), content)
    exp_color = yellow_color
    assert tru_color == exp_color


@pytest.mark.requires_capability("multimodal")
def test_multimodal_structured_output_model_with_image(
    model_id, yellow_img, yellow_color
):
    """Test structured_output_model parameter with image input."""
    model = NovaModel(model_id=model_id, api_key=os.getenv("NOVA_API_KEY"))
    agent = Agent(model=model, load_tools_from_directory=False)

    content = [
        {"text": "Is this image red, blue, or yellow?"},
        {
            "image": {
                "format": "png",
                "source": {"bytes": yellow_img},
            },
        },
    ]
    tru_color = agent(
        content, structured_output_model=type(yellow_color)
    ).structured_output
    exp_color = yellow_color
    assert tru_color == exp_color


# @pytest.mark.requires_capability("audio")
# def test_multimodal_audio_input(model_id, audio_data_content):
#     """Test audio input with multimodal models."""
#     model = NovaModel(model_id=model_id, api_key=os.getenv("NOVA_API_KEY"))
#     agent = Agent(model=model, load_tools_from_directory=False)

#     audio_file = audio_data_content['messages'][0]['content'][1]['input_audio']
#     content = [
#         {"text": "Describe what's in this audio."},
#         {
#             "audio": {
#                 "format": audio_file['format'],
#                 "source": {"bytes": audio_file['data']}
#             }
#         }
#     ]

#     try:
#         result = agent(content)
#         text = result.message["content"][0]["text"]
#         assert len(text) > 0, "Model should respond to audio input"
#     except Exception as e:
#         if "audio" in str(e).lower() or "format" in str(e).lower():
#             pytest.skip(f"Model {model_id} does not support audio input: {e}")
#         raise


@pytest.mark.requires_capability("multimodal")
def test_multimodal_file_input(model_id, test_document):
    """Test file/document input with multimodal models."""
    model = NovaModel(model_id=model_id, api_key=os.getenv("NOVA_API_KEY"))
    agent = Agent(model=model, load_tools_from_directory=False)

    content = [
        {
            "text": "What number is mentioned in this document? Reply with just the number."
        },
        {
            "document": {
                "format": "txt",
                "source": {"bytes": test_document},
                "name": "test.txt",
            }
        },
    ]

    try:
        result = agent(content)
        text = result.message["content"][0]["text"]
        assert "42" in text, "Model should correctly identify the number from document"
    except Exception as e:
        if "document" in str(e).lower() or "format" in str(e).lower():
            pytest.skip(f"Model {model_id} does not support document input: {e}")
        raise


# =============================================================================
# TODO: RESEARCH MODEL TESTS
# =============================================================================


# =============================================================================
# TODO: WEB SEARCH MODEL TESTS
# =============================================================================
