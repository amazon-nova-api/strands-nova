"""Integration tests for multimodal inputs with streaming using Strands Agent.

These tests make actual API calls to the Nova API with images, videos, and other media with streaming enabled.
Requires NOVA_API_KEY environment variable to be set.

Tests are parametrized to run on models that support multimodal capabilities.
"""

import pytest
from dotenv import load_dotenv
from strands import Agent

from strands_nova import NovaModel

# Load environment variables
load_dotenv()

# Placeholder URLs - Replace with actual URLs for real testing
SAMPLE_IMAGE_URL = "https://example.com/test-image.jpg"
SAMPLE_VIDEO_URL = "https://example.com/test-video.mp4"
SAMPLE_AUDIO_URL = "https://example.com/test-audio.mp3"

# Placeholder base64 - Replace with actual base64 encoded media
# This is a minimal valid base64 representation (not a real image/video)
SAMPLE_IMAGE_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
SAMPLE_VIDEO_BASE64 = (
    "AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAAIZnJlZQAAABBtZGF0VGhpcyBpcyB0ZXN0"
)


@pytest.mark.integration
@pytest.mark.requires_capability("multimodal")
@pytest.mark.asyncio
async def test_image_url_streaming(model_id):
    """Test image understanding using URL with streaming.

    Test case: image_url_streaming
    Input: User message with image URL
    Expected: Agent analyzes the image and streams response
    """
    # Initialize model with streaming enabled
    model = NovaModel(model=model_id, stream=True)
    agent = Agent(model=model)

    # Create message with image URL
    # OpenAI-compatible format for multimodal messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What do you see in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": SAMPLE_IMAGE_URL},
                },
            ],
        }
    ]

    # Collect streaming response
    response_text = ""
    async for event in agent.stream_async(messages):
        if "data" in event:
            response_text += event["data"]

    # Verify response
    assert len(response_text) > 0, "Response text is empty"


@pytest.mark.integration
@pytest.mark.requires_capability("multimodal")
@pytest.mark.asyncio
async def test_image_base64_streaming(model_id):
    """Test image understanding using base64 encoding with streaming.

    Test case: image_base64_streaming
    Input: User message with base64 encoded image
    Expected: Agent analyzes the image and streams response
    """
    # Initialize model with streaming enabled
    model = NovaModel(model=model_id, stream=True)
    agent = Agent(model=model)

    # Create message with base64 encoded image
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in detail."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{SAMPLE_IMAGE_BASE64}"
                    },
                },
            ],
        }
    ]

    # Collect streaming response
    response_text = ""
    async for event in agent.stream_async(messages):
        if "data" in event:
            response_text += event["data"]

    # Verify response
    assert len(response_text) > 0, "Response text is empty"


@pytest.mark.integration
@pytest.mark.requires_capability("multimodal")
@pytest.mark.asyncio
async def test_multiple_images_streaming(model_id):
    """Test multiple images in a single message with streaming.

    Test case: multiple_images_streaming
    Input: User message with multiple images
    Expected: Agent analyzes all images and streams response
    """
    # Initialize model with streaming enabled
    model = NovaModel(model=model_id, stream=True)
    agent = Agent(model=model)

    # Create message with multiple images
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Compare these two images. What are the differences?",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": SAMPLE_IMAGE_URL},
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{SAMPLE_IMAGE_BASE64}"
                    },
                },
            ],
        }
    ]

    # Collect streaming response
    response_text = ""
    async for event in agent.stream_async(messages):
        if "data" in event:
            response_text += event["data"]

    # Verify response
    assert len(response_text) > 0, "Response text is empty"


@pytest.mark.integration
@pytest.mark.requires_capability("multimodal")
@pytest.mark.asyncio
async def test_video_url_streaming(model_id):
    """Test video understanding using URL with streaming.

    Test case: video_url_streaming
    Input: User message with video URL
    Expected: Agent analyzes the video and streams response
    """
    # Initialize model with streaming enabled
    model = NovaModel(model=model_id, stream=True)
    agent = Agent(model=model)

    # Create message with video URL
    # Note: Video support format may vary by model
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What happens in this video?"},
                {
                    "type": "video_url",
                    "video_url": {"url": SAMPLE_VIDEO_URL},
                },
            ],
        }
    ]

    # Collect streaming response
    response_text = ""
    async for event in agent.stream_async(messages):
        if "data" in event:
            response_text += event["data"]

    # Verify response
    assert len(response_text) > 0, "Response text is empty"


@pytest.mark.integration
@pytest.mark.requires_capability("multimodal")
@pytest.mark.asyncio
async def test_video_base64_streaming(model_id):
    """Test video understanding using base64 encoding with streaming.

    Test case: video_base64_streaming
    Input: User message with base64 encoded video
    Expected: Agent analyzes the video and streams response
    """
    # Initialize model with streaming enabled
    model = NovaModel(model=model_id, stream=True)
    agent = Agent(model=model)

    # Create message with base64 encoded video
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Summarize the content of this video."},
                {
                    "type": "video_url",
                    "video_url": {
                        "url": f"data:video/mp4;base64,{SAMPLE_VIDEO_BASE64}"
                    },
                },
            ],
        }
    ]

    # Collect streaming response
    response_text = ""
    async for event in agent.stream_async(messages):
        if "data" in event:
            response_text += event["data"]

    # Verify response
    assert len(response_text) > 0, "Response text is empty"


@pytest.mark.integration
@pytest.mark.requires_capability("multimodal")
@pytest.mark.asyncio
async def test_mixed_media_streaming(model_id):
    """Test multiple media types in conversation with streaming.

    Test case: mixed_media_streaming
    Input: Conversation with both images and videos
    Expected: Agent handles mixed media types correctly with streaming
    """
    # Initialize model with streaming enabled
    model = NovaModel(model=model_id, stream=True)
    agent = Agent(model=model)

    # First message with image
    messages1 = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this image."},
                {
                    "type": "image_url",
                    "image_url": {"url": SAMPLE_IMAGE_URL},
                },
            ],
        }
    ]

    response1_text = ""
    async for event in agent.stream_async(messages1):
        if "data" in event:
            response1_text += event["data"]

    assert len(response1_text) > 0, "First response text is empty"

    # Follow-up with video
    messages2 = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Now analyze this video."},
                {
                    "type": "video_url",
                    "video_url": {"url": SAMPLE_VIDEO_URL},
                },
            ],
        }
    ]

    response2_text = ""
    async for event in agent.stream_async(messages2):
        if "data" in event:
            response2_text += event["data"]

    assert len(response2_text) > 0, "Second response text is empty"


@pytest.mark.integration
@pytest.mark.requires_capability("multimodal")
@pytest.mark.asyncio
async def test_image_with_detailed_question_streaming(model_id):
    """Test detailed image analysis with specific questions using streaming.

    Test case: image_detailed_question_streaming
    Input: Image with specific analytical questions
    Expected: Agent provides detailed analysis based on questions with streaming
    """
    # Initialize model with streaming enabled
    model = NovaModel(model=model_id, stream=True)
    agent = Agent(model=model)

    # Create message with specific questions
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Looking at this image, please answer:\n1. What objects are present?\n2. What colors dominate?\n3. What is the overall mood or atmosphere?",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": SAMPLE_IMAGE_URL},
                },
            ],
        }
    ]

    # Collect streaming response
    response_text = ""
    async for event in agent.stream_async(messages):
        if "data" in event:
            response_text += event["data"]

    # Verify response
    assert len(response_text) > 0, "Response text is empty"


@pytest.mark.integration
@pytest.mark.requires_capability("multimodal")
@pytest.mark.asyncio
async def test_image_text_interleaved_streaming(model_id):
    """Test interleaved text and images in conversation with streaming.

    Test case: image_text_interleaved_streaming
    Input: Multiple turns with text and images
    Expected: Agent maintains context across multimodal turns with streaming
    """
    # Initialize model with streaming enabled
    model = NovaModel(model=model_id, stream=True)
    agent = Agent(model=model)

    # First turn: text only
    response1_text = ""
    async for event in agent.stream_async(
        "I'm going to show you an image and ask questions about it."
    ):
        if "data" in event:
            response1_text += event["data"]

    assert len(response1_text) > 0, "First response text is empty"

    # Second turn: image with question
    messages2 = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Here's the image. What do you see?"},
                {
                    "type": "image_url",
                    "image_url": {"url": SAMPLE_IMAGE_URL},
                },
            ],
        }
    ]

    response2_text = ""
    async for event in agent.stream_async(messages2):
        if "data" in event:
            response2_text += event["data"]

    assert len(response2_text) > 0, "Second response text is empty"

    # Third turn: follow-up text question
    response3_text = ""
    async for event in agent.stream_async(
        "Based on what you saw, what would you recommend?"
    ):
        if "data" in event:
            response3_text += event["data"]

    assert len(response3_text) > 0, "Third response text is empty"


@pytest.mark.integration
@pytest.mark.requires_capability("multimodal")
@pytest.mark.asyncio
async def test_image_format_variations_streaming(model_id):
    """Test different image format specifications with streaming.

    Test case: image_format_variations_streaming
    Input: Same image in different format specifications
    Expected: Agent handles various image format specifications with streaming
    """
    # Initialize model with streaming enabled
    model = NovaModel(model=model_id, stream=True)
    agent = Agent(model=model)

    # Test with JPEG
    messages_jpeg = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this JPEG image."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{SAMPLE_IMAGE_BASE64}"
                    },
                },
            ],
        }
    ]

    response_jpeg_text = ""
    async for event in agent.stream_async(messages_jpeg):
        if "data" in event:
            response_jpeg_text += event["data"]

    assert len(response_jpeg_text) > 0, "JPEG response text is empty"

    # Test with PNG
    messages_png = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Now describe this PNG image."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{SAMPLE_IMAGE_BASE64}"
                    },
                },
            ],
        }
    ]

    response_png_text = ""
    async for event in agent.stream_async(messages_png):
        if "data" in event:
            response_png_text += event["data"]

    assert len(response_png_text) > 0, "PNG response text is empty"
