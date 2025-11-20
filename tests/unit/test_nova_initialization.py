"""Comprehensive unit tests for NovaModel initialization."""

import os
from unittest.mock import patch

import pytest

from strands_nova import NovaModel


class TestBasicInitialization:
    """Test basic NovaModel initialization scenarios."""

    def test_initialization_with_minimal_params(self):
        """Test initialization with only required parameters."""
        model = NovaModel(api_key="test-api-key")
        assert model.api_key == "test-api-key"
        assert model.model == "nova-premier-v1"
        assert model.temperature == 0.7
        assert model.max_tokens == 4096
        assert model.top_p == 0.9
        assert model.base_url == "https://api.nova.amazon.com/v1/chat/completions"
        assert model.models_url == "https://api.nova.amazon.com/v1/models"
        assert model.stream_options == {"include_usage": True}
        assert model.stop == []
        assert model.reasoning_effort is None
        assert model.web_search_options is None

    def test_initialization_with_all_standard_params(self):
        """Test initialization with all standard parameters."""
        model = NovaModel(
            api_key="custom-key",
            model="nova-pro-v3",
            temperature=0.5,
            max_tokens=2048,
            top_p=0.95,
            stop=["END", "STOP"],
            base_url="https://custom.api.com/v1/chat",
            stream_options={"include_usage": False},
        )
        assert model.api_key == "custom-key"
        assert model.model == "nova-pro-v3"
        assert model.temperature == 0.5
        assert model.max_tokens == 2048
        assert model.top_p == 0.95
        assert model.stop == ["END", "STOP"]
        assert model.base_url == "https://custom.api.com/v1/chat"
        assert model.stream_options == {"include_usage": False}

    def test_initialization_with_reasoning_effort(self):
        """Test initialization with reasoning_effort parameter."""
        for effort in ["low", "medium", "high"]:
            model = NovaModel(
                api_key="test-key",
                model="mumbai-flintflex-reasoning-v3",
                reasoning_effort=effort,
            )
            assert model.reasoning_effort == effort

    def test_initialization_with_web_search_options(self):
        """Test initialization with web_search_options parameter."""
        web_options = {"search_context_size": "low", "custom_param": "value"}
        model = NovaModel(
            api_key="test-key",
            web_search_options=web_options,
        )
        assert model.web_search_options == web_options

    def test_initialization_with_additional_kwargs(self):
        """Test initialization with additional custom parameters."""
        model = NovaModel(
            api_key="test-key",
            custom_param1="value1",
            custom_param2=42,
            custom_param3=True,
        )
        assert model.additional_params["custom_param1"] == "value1"
        assert model.additional_params["custom_param2"] == 42
        assert model.additional_params["custom_param3"] is True


class TestAPIKeyHandling:
    """Test API key handling in various scenarios."""

    def test_initialization_with_explicit_api_key(self):
        """Test initialization with explicitly provided API key."""
        model = NovaModel(api_key="explicit-key")
        assert model.api_key == "explicit-key"

    def test_initialization_with_env_var_api_key(self):
        """Test initialization using NOVA_API_KEY environment variable."""
        with patch.dict(os.environ, {"NOVA_API_KEY": "env-key"}):
            model = NovaModel()
            assert model.api_key == "env-key"

    def test_explicit_api_key_overrides_env_var(self):
        """Test that explicit API key overrides environment variable."""
        with patch.dict(os.environ, {"NOVA_API_KEY": "env-key"}):
            model = NovaModel(api_key="explicit-key")
            assert model.api_key == "explicit-key"

    def test_initialization_without_api_key_raises_error(self):
        """Test that initialization without API key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Nova API key is required"):
                NovaModel()

    def test_initialization_with_empty_api_key_raises_error(self):
        """Test that initialization with empty API key raises ValueError."""
        with patch.dict(os.environ, {"NOVA_API_KEY": ""}):
            with pytest.raises(ValueError, match="Nova API key is required"):
                NovaModel()

    def test_initialization_with_none_api_key_uses_env(self):
        """Test that None API key falls back to environment variable."""
        with patch.dict(os.environ, {"NOVA_API_KEY": "env-key"}):
            model = NovaModel(api_key=None)
            assert model.api_key == "env-key"


class TestModelVariations:
    """Test initialization with different model names."""

    def test_initialization_with_nova_premier(self):
        """Test initialization with Nova Premier model."""
        model = NovaModel(api_key="test-key", model="nova-premier-v1")
        assert model.model == "nova-premier-v1"

    def test_initialization_with_nova_pro(self):
        """Test initialization with Nova Pro model."""
        model = NovaModel(api_key="test-key", model="Nova Pro v3 (6.x)")
        assert model.model == "Nova Pro v3 (6.x)"

    def test_initialization_with_reasoning_model(self):
        """Test initialization with reasoning model."""
        model = NovaModel(api_key="test-key", model="mumbai-flintflex-reasoning-v3")
        assert model.model == "mumbai-flintflex-reasoning-v3"

    def test_initialization_with_custom_model_name(self):
        """Test initialization with custom/unknown model name."""
        model = NovaModel(api_key="test-key", model="custom-model-name")
        assert model.model == "custom-model-name"


class TestParameterBoundaries:
    """Test initialization with boundary and edge case parameter values."""

    def test_initialization_with_zero_temperature(self):
        """Test initialization with temperature=0."""
        model = NovaModel(api_key="test-key", temperature=0.0)
        assert model.temperature == 0.0

    def test_initialization_with_max_temperature(self):
        """Test initialization with temperature=1.0."""
        model = NovaModel(api_key="test-key", temperature=1.0)
        assert model.temperature == 1.0

    def test_initialization_with_min_max_tokens(self):
        """Test initialization with minimal max_tokens."""
        model = NovaModel(api_key="test-key", max_tokens=1)
        assert model.max_tokens == 1

    def test_initialization_with_large_max_tokens(self):
        """Test initialization with very large max_tokens."""
        model = NovaModel(api_key="test-key", max_tokens=100000)
        assert model.max_tokens == 100000

    def test_initialization_with_zero_top_p(self):
        """Test initialization with top_p=0."""
        model = NovaModel(api_key="test-key", top_p=0.0)
        assert model.top_p == 0.0

    def test_initialization_with_max_top_p(self):
        """Test initialization with top_p=1.0."""
        model = NovaModel(api_key="test-key", top_p=1.0)
        assert model.top_p == 1.0

    def test_initialization_with_empty_stop_list(self):
        """Test initialization with empty stop sequences list."""
        model = NovaModel(api_key="test-key", stop=[])
        assert model.stop == []

    def test_initialization_with_many_stop_sequences(self):
        """Test initialization with many stop sequences."""
        stop_seqs = [f"STOP{i}" for i in range(100)]
        model = NovaModel(api_key="test-key", stop=stop_seqs)
        assert model.stop == stop_seqs


class TestURLConfiguration:
    """Test URL configuration scenarios."""

    def test_default_base_url(self):
        """Test that default base URL is set correctly."""
        model = NovaModel(api_key="test-key")
        assert model.base_url == "https://api.nova.amazon.com/v1/chat/completions"

    def test_custom_base_url(self):
        """Test initialization with custom base URL."""
        custom_url = "https://custom-endpoint.example.com/api/v2/chat"
        model = NovaModel(api_key="test-key", base_url=custom_url)
        assert model.base_url == custom_url

    def test_base_url_with_trailing_slash(self):
        """Test base URL with trailing slash."""
        model = NovaModel(api_key="test-key", base_url="https://api.example.com/")
        assert model.base_url == "https://api.example.com/"

    def test_base_url_without_protocol(self):
        """Test that base URL can be set without protocol (user responsibility)."""
        model = NovaModel(api_key="test-key", base_url="api.example.com/v1/chat")
        assert model.base_url == "api.example.com/v1/chat"


class TestStreamOptions:
    """Test stream options configuration."""

    def test_default_stream_options(self):
        """Test that default stream options include usage."""
        model = NovaModel(api_key="test-key")
        assert model.stream_options == {"include_usage": True}

    def test_custom_stream_options(self):
        """Test initialization with custom stream options."""
        custom_options = {"include_usage": False, "custom_option": "value"}
        model = NovaModel(api_key="test-key", stream_options=custom_options)
        assert model.stream_options == custom_options

    def test_stream_options_with_empty_dict(self):
        """Test initialization with empty stream options."""
        model = NovaModel(api_key="test-key", stream_options={})
        assert model.stream_options == {}

    def test_stream_options_with_none(self):
        """Test that None stream options uses default."""
        model = NovaModel(api_key="test-key", stream_options=None)
        assert model.stream_options == {"include_usage": True}


class TestStringRepresentation:
    """Test string representation of NovaModel."""

    def test_str_contains_model_name(self):
        """Test that __str__ includes model name."""
        model = NovaModel(api_key="test-key", model="nova-pro-v3")
        str_repr = str(model)
        assert "nova-pro-v3" in str_repr

    def test_str_contains_temperature(self):
        """Test that __str__ includes temperature."""
        model = NovaModel(api_key="test-key", temperature=0.5)
        str_repr = str(model)
        assert "0.5" in str_repr

    def test_str_contains_max_tokens(self):
        """Test that __str__ includes max_tokens."""
        model = NovaModel(api_key="test-key", max_tokens=2048)
        str_repr = str(model)
        assert "2048" in str_repr

    def test_str_format(self):
        """Test overall format of __str__."""
        model = NovaModel(api_key="test-key")
        str_repr = str(model)
        assert str_repr.startswith("NovaModel(")
        assert "model=" in str_repr
        assert "temperature=" in str_repr
        assert "max_tokens=" in str_repr
