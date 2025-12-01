"""Configuration and fixtures for integration tests.

This module provides static model categorization and parametrizes tests
based on model capabilities.
"""

import os
import json
from typing import Literal
import pydantic
import pytest
import strands
from dotenv import load_dotenv
from strands import Agent

from amazon_nova import NovaAPIModel

# Load environment variables
load_dotenv()

# Static mapping of capabilities to model IDs
MODEL_CAPABILITIES = {
    "text": ["nova-micro-v1"],
    "multimodal": ["nova-lite-v1", "nova-premier-v1", "nova-pro-v1"],
    # "reasoning": [""],
    # "image_audio": ["nova-omni"],
}


@pytest.fixture
def yellow_img(pytestconfig):
    path = pytestconfig.rootdir / "tests/integration/yellow.png"
    with open(path, "rb") as fp:
        return fp.read()


@pytest.fixture
def test_document(pytestconfig):
    path = pytestconfig.rootdir / "tests/integration/test.txt"
    with open(path, "rb") as fp:
        return fp.read()


@pytest.fixture
def audio_data_content(pytestconfig):
    path = pytestconfig.rootdir / "tests/integration/audio_input_input.json"
    with open(path, "r") as f:
        data = json.load(f)
        return data


@pytest.fixture
def model():
    return NovaAPIModel(
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
        @pytest.mark.requires_capability("multimodal")
        def test_something(model_id):
            # Will run on all multimodal models

        @pytest.mark.requires_capability("text")
        def test_tools(model_id):
            # Will run only on text models
    """
    if "model_id" in metafunc.fixturenames:
        # Get the requires_capability marker
        marker = metafunc.definition.get_closest_marker("requires_capability")

        if marker:
            # Get capability from marker args
            capability = marker.args[0] if marker.args else "multimodal"

            # Get model IDs for this capability
            model_ids = MODEL_CAPABILITIES.get(capability, [])

            if not model_ids:
                pytest.skip(f"No models found with capability: {capability}")

            # Parametrize the test with filtered model IDs
            metafunc.parametrize("model_id", model_ids)
