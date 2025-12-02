"""Unit tests for NovaAPIModel initialization and configuration."""

import os
from unittest.mock import patch

import pytest

from strands_amazon_nova.nova import NovaAPIModel


class TestNovaModelInitialization:
    """Test NovaAPIModel initialization."""

    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        model = NovaAPIModel(model_id="nova-pro-v1", api_key="test-api-key")

        assert model.config["model_id"] == "nova-pro-v1"
        assert model.api_key == "test-api-key"
        assert model.base_url == "https://api.nova.amazon.com/v1"
        assert model.timeout == 300.0
        assert model._stream is True
        assert model.stream_options == {"include_usage": True}

    def test_init_with_env_api_key(self):
        """Test initialization with API key from environment variable."""
        with patch.dict(os.environ, {"NOVA_API_KEY": "env-api-key"}):
            model = NovaAPIModel(model_id="nova-lite-v2")

            assert model.api_key == "env-api-key"
            assert model.config["model_id"] == "nova-lite-v2"

    def test_init_without_api_key_raises_error(self):
        """Test initialization without API key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="api_key must be provided"):
                NovaAPIModel(model_id="nova-pro-v1")

    def test_init_with_custom_base_url(self):
        """Test initialization with custom base URL."""
        model = NovaAPIModel(
            model_id="nova-pro-v1",
            api_key="test-key",
            base_url="https://custom.api.com/v2",
        )

        assert model.base_url == "https://custom.api.com/v2"

    def test_init_base_url_strips_trailing_slash(self):
        """Test that trailing slash is stripped from base URL."""
        model = NovaAPIModel(
            model_id="nova-pro-v1",
            api_key="test-key",
            base_url="https://api.nova.amazon.com/v1/",
        )

        assert model.base_url == "https://api.nova.amazon.com/v1"

    def test_init_with_custom_timeout(self):
        """Test initialization with custom timeout."""
        model = NovaAPIModel(model_id="nova-pro-v1", api_key="test-key", timeout=600.0)

        assert model.timeout == 600.0

    def test_init_with_params(self):
        """Test initialization with model parameters."""
        params = {"max_tokens": 1000, "temperature": 0.7, "top_p": 0.9}
        model = NovaAPIModel(model_id="nova-pro-v1", api_key="test-key", params=params)

        assert model.config["params"] == params

    def test_init_with_stream_false(self):
        """Test initialization with streaming disabled."""
        model = NovaAPIModel(model_id="nova-pro-v1", api_key="test-key", stream=False)

        assert model._stream is False

    def test_init_with_custom_stream_options(self):
        """Test initialization with custom stream options."""
        stream_options = {"include_usage": False}
        model = NovaAPIModel(
            model_id="nova-pro-v1", api_key="test-key", stream_options=stream_options
        )

        assert model.stream_options == stream_options

    def test_init_with_extra_config(self):
        """Test initialization with extra configuration."""
        model = NovaAPIModel(
            model_id="nova-pro-v1", api_key="test-key", custom_field="custom_value"
        )

        assert model.config["custom_field"] == "custom_value"


class TestNovaModelConfiguration:
    """Test NovaAPIModel configuration methods."""

    def test_get_config(self):
        """Test getting model configuration."""
        params = {"max_tokens": 500}
        model = NovaAPIModel(model_id="nova-pro-v1", api_key="test-key", params=params)

        config = model.get_config()

        assert config["model_id"] == "nova-pro-v1"
        assert config["params"] == params

    def test_update_config(self):
        """Test updating model configuration."""
        model = NovaAPIModel(model_id="nova-pro-v1", api_key="test-key")

        new_params = {"temperature": 0.8, "max_tokens": 2000}
        model.update_config(model_id="nova-lite-v2", params=new_params)

        config = model.get_config()
        assert config["model_id"] == "nova-lite-v2"
        assert config["params"] == new_params

    def test_update_config_partial(self):
        """Test partial configuration update."""
        params = {"max_tokens": 500}
        model = NovaAPIModel(model_id="nova-pro-v1", api_key="test-key", params=params)

        model.update_config(model_id="nova-lite-v2")

        config = model.get_config()
        assert config["model_id"] == "nova-lite-v2"
        assert config["params"] == params  # Should remain unchanged
