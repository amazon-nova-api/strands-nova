"""Integration tests for multimodal inputs using Strands Agent.

These tests make actual API calls to the Nova API with images, videos, and other media.
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
def test_image_url(model_id):
    """Test image understanding using URL.

    Test case: image_url
    Input: User message with image URL
    Expected: Agent analyzes the image and responds
    """
    # Initialize model and agent
    model = NovaModel(model=model_id)
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

    # Call agent with multimodal message
    response = agent(messages)

    # Verify response
    assert response is not None, "No response received"
    assert hasattr(response, "message"), "Response has no message attribute"
    assert len(response.message) > 0, "Response message is empty"


@pytest.mark.integration
@pytest.mark.requires_capability("multimodal")
def test_image_base64(model_id):
    """Test image understanding using base64 encoding.

    Test case: image_base64
    Input: User message with base64 encoded image
    Expected: Agent analyzes the image and responds
    """
    # Initialize model and agent
    model = NovaModel(model=model_id)
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

    # Call agent with multimodal message
    response = agent(messages)

    # Verify response
    assert response is not None, "No response received"
    assert hasattr(response, "message"), "Response has no message attribute"
    assert len(response.message) > 0, "Response message is empty"


@pytest.mark.integration
@pytest.mark.requires_capability("multimodal")
def test_multiple_images(model_id):
    """Test multiple images in a single message.

    Test case: multiple_images
    Input: User message with multiple images
    Expected: Agent analyzes all images and responds
    """
    # Initialize model and agent
    model = NovaModel(model=model_id)
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

    # Call agent with multimodal message
    response = agent(messages)

    # Verify response
    assert response is not None, "No response received"
    assert hasattr(response, "message"), "Response has no message attribute"
    assert len(response.message) > 0, "Response message is empty"


@pytest.mark.integration
@pytest.mark.requires_capability("multimodal")
def test_video_url(model_id):
    """Test video understanding using URL.

    Test case: video_url
    Input: User message with video URL
    Expected: Agent analyzes the video and responds
    """
    # Initialize model and agent
    model = NovaModel(model=model_id)
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

    # Call agent with multimodal message
    response = agent(messages)

    # Verify response
    assert response is not None, "No response received"
    assert hasattr(response, "message"), "Response has no message attribute"
    assert len(response.message) > 0, "Response message is empty"


@pytest.mark.integration
@pytest.mark.requires_capability("multimodal")
def test_video_base64(model_id):
    """Test video understanding using base64 encoding.

    Test case: video_base64
    Input: User message with base64 encoded video
    Expected: Agent analyzes the video and responds
    """
    # Initialize model and agent
    model = NovaModel(model=model_id)
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

    # Call agent with multimodal message
    response = agent(messages)

    # Verify response
    assert response is not None, "No response received"
    assert hasattr(response, "message"), "Response has no message attribute"
    assert len(response.message) > 0, "Response message is empty"


@pytest.mark.integration
@pytest.mark.requires_capability("multimodal")
def test_mixed_media(model_id):
    """Test multiple media types in conversation.

    Test case: mixed_media
    Input: Conversation with both images and videos
    Expected: Agent handles mixed media types correctly
    """
    # Initialize model and agent
    model = NovaModel(model=model_id)
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

    response1 = agent(messages1)
    assert response1 is not None, "No response for first message"

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

    response2 = agent(messages2)
    assert response2 is not None, "No response for second message"
    assert hasattr(response2, "message"), "Response has no message attribute"


@pytest.mark.integration
@pytest.mark.requires_capability("multimodal")
def test_image_with_detailed_question(model_id):
    """Test detailed image analysis with specific questions.

    Test case: image_detailed_question
    Input: Image with specific analytical questions
    Expected: Agent provides detailed analysis based on questions
    """
    # Initialize model and agent
    model = NovaModel(model=model_id)
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

    # Call agent with detailed query
    response = agent(messages)

    # Verify response
    assert response is not None, "No response received"
    assert hasattr(response, "message"), "Response has no message attribute"
    assert len(response.message) > 0, "Response message is empty"


@pytest.mark.integration
@pytest.mark.requires_capability("multimodal")
def test_image_text_interleaved(model_id):
    """Test interleaved text and images in conversation.

    Test case: image_text_interleaved
    Input: Multiple turns with text and images
    Expected: Agent maintains context across multimodal turns
    """
    # Initialize model and agent
    model = NovaModel(model=model_id)
    agent = Agent(model=model)

    # First turn: text only
    response1 = agent("I'm going to show you an image and ask questions about it.")
    assert response1 is not None, "No response for first turn"

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

    response2 = agent(messages2)
    assert response2 is not None, "No response for image turn"

    # Third turn: follow-up text question
    response3 = agent("Based on what you saw, what would you recommend?")
    assert response3 is not None, "No response for follow-up"
    assert hasattr(response3, "message"), "Response has no message attribute"


@pytest.mark.integration
@pytest.mark.requires_capability("multimodal")
def test_image_format_variations(model_id):
    """Test different image format specifications.

    Test case: image_format_variations
    Input: Same image in different format specifications
    Expected: Agent handles various image format specifications
    """
    # Initialize model and agent
    model = NovaModel(model=model_id)
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

    response_jpeg = agent(messages_jpeg)
    assert response_jpeg is not None, "No response for JPEG image"

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

    response_png = agent(messages_png)
    assert response_png is not None, "No response for PNG image"
    assert hasattr(response_png, "message"), "Response has no message attribute"
