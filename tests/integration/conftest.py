"""Configuration and fixtures for integration tests.

This module dynamically fetches available models from the Nova API and
parametrizes tests based on model capabilities.
"""

import os
import json
from typing import Any, Dict, List, Literal
import pydantic
import httpx
import pytest
import strands
from dotenv import load_dotenv
from strands import Agent

from strands_nova import NovaModel

# Load environment variables
load_dotenv()


@pytest.fixture
def yellow_img(pytestconfig):
    path = pytestconfig.rootdir / "tests/integration/yellow.png"
    with open(path, "rb") as fp:
        return fp.read()

@pytest.fixture
def audio_data_content(pytestconfig):
    path = pytestconfig.rootdir / "tests/integration/audio_input_input.json"
    with open(path, 'r') as f:
        data = json.load(f)
        return data


@pytest.fixture
def model():
    return NovaModel(
        model_id="nova-pro-v1",
        api_key=os.getenv("NOVA_API_KEY"),
    )


@pytest.fixture
def weather():
    class Weather(pydantic.BaseModel):
        """Extracts the time and weather from the user's message with the exact strings."""

        time: str
        weather: str

    return Weather(time="12:00", weather="sunny")


@pytest.fixture
def yellow_color():
    class Color(pydantic.BaseModel):
        """Describes a color."""

        name: Literal["red", "blue", "yellow"]

        @pydantic.field_validator("name", mode="after")
        @classmethod
        def lower(_, value):
            return value.lower()

    return Color(name="yellow")


@pytest.fixture
def tools():
    @strands.tool
    def tool_time() -> str:
        return "12:00"

    @strands.tool
    def tool_weather() -> str:
        return "sunny"

    return [tool_time, tool_weather]


@pytest.fixture
def agent(model, tools):
    return Agent(model=model, tools=tools)


def get_available_models() -> List[Dict[str, Any]]:
    """Fetch available models from Nova API.

    Returns:
        list: List of available model dictionaries with their capabilities
    """
    api_key = os.getenv("NOVA_API_KEY")

    if not api_key:
        pytest.skip("NOVA_API_KEY not found in environment variables")

    models_url = "https://api.nova.amazon.com/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = httpx.get(models_url, headers=headers, timeout=30.0)
        response.raise_for_status()
        data = response.json()
        return data.get("data", [])
    except Exception as e:
        pytest.skip(f"Failed to fetch models: {e}")


def categorize_model(model: Dict[str, Any]) -> List[str]:
    """Categorize a model based on its description and capabilities.

    Args:
        model: Model dictionary from API

    Returns:
        list: List of categories this model belongs to
    """
    model_id = model.get("id", "")
    description = model.get("description", "").lower()
    categories = []

    # Image generation models (must check first to exclude from other categories)
    if model_id == "nova-orchestrator-v1" or (
        "generate" in description and "image" in description
    ):
        categories.append("image_generation")
        return (
            categories  # Return early - image generation models only get this category
        )

    # Text-only models
    if "text - text" in description or model_id == "nova-micro-v1":
        categories.append("text")

    # Multimodal models (support images, video, etc.)
    if any(keyword in description for keyword in ["multimodal", "image", "video"]):
        categories.append("multimodal")

    # Research models
    if "research" in description:
        categories.append("research")

    # Web search capable models (based on known capabilities)
    # Note: May need to update this list as more models support web search
    if model_id in ["nova-premier-v1", "nova-pro-v1"]:
        categories.append("web_search")

    return categories


def filter_models_by_capability(
    models: List[Dict[str, Any]], capability: str
) -> List[str]:
    """Filter models based on their capabilities.

    This function balances test coverage with speed by:
    - Testing specialized models on tasks they're designed for
    - Using representative models for basic functionality tests
    - Avoiding redundant testing on similar models

    Args:
        models: List of model dictionaries from API
        capability: Capability to filter by
            - 'text': Text-only tests (nova-micro-v1 + nova-premier-v1 as multimodal representative)
            - 'multimodal': Multimodal tests (all multimodal models)
            - 'image_generation': Image generation tests (nova-orchestrator-v1)
            - 'research': Research-focused tests (nova-deep-research-v1, ragentic-cloaked-v1)
            - 'web_search': Web search tests (nova-premier-v1, nova-pro-v1)
            - 'production': All production models owned by Amazon
            - 'all': All models

    Returns:
        list: List of model IDs that support the capability
    """
    filtered = []

    for model in models:
        model_id = model.get("id")
        owned_by = model.get("owned_by")

        # Only consider production models (owned by Amazon)
        if owned_by != "Amazon":
            continue

        # Get model categories
        model_categories = categorize_model(model)

        # Production models (all Amazon models)
        if capability == "production":
            filtered.append(model_id)

        # Text tests: text-only and multimodal models (excluding image generation only)
        elif capability == "text":
            if (
                "text" in model_categories or "multimodal" in model_categories
            ) and model_id != "nova-orchestrator-v1":
                filtered.append(model_id)

        # Filter by specific capability (multimodal, research, etc.)
        elif capability in model_categories:
            filtered.append(model_id)

        # All models
        elif capability == "all":
            filtered.append(model_id)

    return filtered


@pytest.fixture(scope="session")
def available_models():
    """Fixture that provides all available models.

    Returns:
        list: List of all available model dictionaries
    """
    return get_available_models()


@pytest.fixture(scope="session")
def production_model_ids(available_models):
    """Fixture that provides production model IDs.

    Returns:
        list: List of production model IDs (owned by Amazon)
    """
    return filter_models_by_capability(available_models, "production")


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line(
        "markers",
        "requires_capability: mark test to run only on models with specific capability",
    )


def pytest_generate_tests(metafunc):
    """Generate parametrized tests based on model capabilities.

    If a test has a 'model_id' parameter and is marked with @pytest.mark.requires_capability,
    it will be parametrized to run only on models that support that capability.

    Example:
        @pytest.mark.requires_capability("basic")
        def test_something(model_id):
            # Will run on all production models that support basic text

        @pytest.mark.requires_capability("tool_support")
        def test_tools(model_id):
            # Will run only on models with tool support
    """
    if "model_id" in metafunc.fixturenames:
        # Get the requires_capability marker
        marker = metafunc.definition.get_closest_marker("requires_capability")

        if marker:
            # Get capability from marker args
            capability = marker.args[0] if marker.args else "basic"

            # Fetch models and filter by capability
            all_models = get_available_models()
            model_ids = filter_models_by_capability(all_models, capability)

            if not model_ids:
                pytest.skip(f"No models found with capability: {capability}")

            # Parametrize the test with filtered model IDs
            metafunc.parametrize("model_id", model_ids)
