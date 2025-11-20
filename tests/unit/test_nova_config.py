"""Comprehensive unit tests for NovaModel configuration management."""

from strands_nova import NovaModel


class TestUpdateConfig:
    """Test update_config method with various scenarios."""

    def test_update_temperature(self):
        """Test updating temperature configuration."""
        model = NovaModel(api_key="test-key", temperature=0.7)
        model.update_config(temperature=0.5)
        assert model.temperature == 0.5

    def test_update_max_tokens(self):
        """Test updating max_tokens configuration."""
        model = NovaModel(api_key="test-key", max_tokens=4096)
        model.update_config(max_tokens=2048)
        assert model.max_tokens == 2048

    def test_update_top_p(self):
        """Test updating top_p configuration."""
        model = NovaModel(api_key="test-key", top_p=0.9)
        model.update_config(top_p=0.95)
        assert model.top_p == 0.95

    def test_update_model_name(self):
        """Test updating model name."""
        model = NovaModel(api_key="test-key", model="nova-premier-v1")
        model.update_config(model="nova-pro-v3")
        assert model.model == "nova-pro-v3"

    def test_update_stop_sequences(self):
        """Test updating stop sequences."""
        model = NovaModel(api_key="test-key", stop=["END"])
        model.update_config(stop=["STOP", "EXIT"])
        assert model.stop == ["STOP", "EXIT"]

    def test_update_base_url(self):
        """Test updating base URL."""
        model = NovaModel(api_key="test-key")
        new_url = "https://new.api.com/v1/chat"
        model.update_config(base_url=new_url)
        assert model.base_url == new_url

    def test_update_stream_options(self):
        """Test updating stream options."""
        model = NovaModel(api_key="test-key")
        new_options = {"include_usage": False, "custom": "value"}
        model.update_config(stream_options=new_options)
        assert model.stream_options == new_options

    def test_update_reasoning_effort(self):
        """Test updating reasoning_effort."""
        model = NovaModel(api_key="test-key", reasoning_effort="low")
        model.update_config(reasoning_effort="high")
        assert model.reasoning_effort == "high"

    def test_update_web_search_options(self):
        """Test updating web_search_options."""
        model = NovaModel(api_key="test-key")
        web_opts = {"search_context_size": "medium"}
        model.update_config(web_search_options=web_opts)
        assert model.web_search_options == web_opts

    def test_update_multiple_params_at_once(self):
        """Test updating multiple parameters simultaneously."""
        model = NovaModel(api_key="test-key")
        model.update_config(
            temperature=0.3, max_tokens=1024, top_p=0.85, model="nova-lite-v1"
        )
        assert model.temperature == 0.3
        assert model.max_tokens == 1024
        assert model.top_p == 0.85
        assert model.model == "nova-lite-v1"

    def test_update_with_custom_param(self):
        """Test updating with custom parameter not in standard attrs."""
        model = NovaModel(api_key="test-key")
        model.update_config(custom_param="custom_value")
        assert model.additional_params["custom_param"] == "custom_value"

    def test_update_existing_custom_param(self):
        """Test updating an existing custom parameter."""
        model = NovaModel(api_key="test-key", extra_param="old")
        assert model.additional_params["extra_param"] == "old"
        model.update_config(extra_param="new")
        assert model.additional_params["extra_param"] == "new"

    def test_update_with_mix_of_standard_and_custom_params(self):
        """Test updating mix of standard and custom parameters."""
        model = NovaModel(api_key="test-key")
        model.update_config(
            temperature=0.6, custom_param1="value1", max_tokens=3000, custom_param2=42
        )
        assert model.temperature == 0.6
        assert model.max_tokens == 3000
        assert model.additional_params["custom_param1"] == "value1"
        assert model.additional_params["custom_param2"] == 42

    def test_update_with_empty_dict(self):
        """Test updating with empty configuration dict."""
        model = NovaModel(api_key="test-key", temperature=0.7)
        model.update_config()
        assert model.temperature == 0.7

    def test_update_api_key(self):
        """Test that API key can be updated (though not recommended)."""
        model = NovaModel(api_key="old-key")
        model.update_config(api_key="new-key")
        assert model.api_key == "new-key"


class TestGetConfig:
    """Test get_config method with various scenarios."""

    def test_get_config_returns_dict(self):
        """Test that get_config returns a dictionary."""
        model = NovaModel(api_key="test-key")
        config = model.get_config()
        assert isinstance(config, dict)

    def test_get_config_contains_standard_params(self):
        """Test that get_config includes all standard parameters."""
        model = NovaModel(api_key="test-key")
        config = model.get_config()
        assert "model" in config
        assert "temperature" in config
        assert "max_tokens" in config
        assert "top_p" in config
        assert "stop" in config
        assert "base_url" in config
        assert "stream_options" in config

    def test_get_config_values_match_model_state(self):
        """Test that get_config returns current model state."""
        model = NovaModel(
            api_key="test-key",
            model="nova-pro-v3",
            temperature=0.5,
            max_tokens=2048,
            top_p=0.95,
        )
        config = model.get_config()
        assert config["model"] == "nova-pro-v3"
        assert config["temperature"] == 0.5
        assert config["max_tokens"] == 2048
        assert config["top_p"] == 0.95

    def test_get_config_includes_reasoning_effort_when_set(self):
        """Test that get_config includes reasoning_effort when present."""
        model = NovaModel(api_key="test-key", reasoning_effort="medium")
        config = model.get_config()
        assert "reasoning_effort" in config
        assert config["reasoning_effort"] == "medium"

    def test_get_config_excludes_reasoning_effort_when_none(self):
        """Test that get_config excludes reasoning_effort when None."""
        model = NovaModel(api_key="test-key")
        config = model.get_config()
        assert "reasoning_effort" not in config

    def test_get_config_includes_web_search_options_when_set(self):
        """Test that get_config includes web_search_options when present."""
        web_opts = {"search_context_size": "low"}
        model = NovaModel(api_key="test-key", web_search_options=web_opts)
        config = model.get_config()
        assert "web_search_options" in config
        assert config["web_search_options"] == web_opts

    def test_get_config_excludes_web_search_options_when_none(self):
        """Test that get_config excludes web_search_options when None."""
        model = NovaModel(api_key="test-key")
        config = model.get_config()
        assert "web_search_options" not in config

    def test_get_config_includes_additional_params(self):
        """Test that get_config includes additional custom parameters."""
        model = NovaModel(
            api_key="test-key",
            custom_param1="value1",
            custom_param2=42,
        )
        config = model.get_config()
        assert config["custom_param1"] == "value1"
        assert config["custom_param2"] == 42

    def test_get_config_includes_stop_sequences(self):
        """Test that get_config includes stop sequences."""
        stop_seqs = ["END", "STOP"]
        model = NovaModel(api_key="test-key", stop=stop_seqs)
        config = model.get_config()
        assert config["stop"] == stop_seqs

    def test_get_config_includes_empty_stop_list(self):
        """Test that get_config includes empty stop list."""
        model = NovaModel(api_key="test-key", stop=[])
        config = model.get_config()
        assert config["stop"] == []

    def test_get_config_does_not_include_api_key(self):
        """Test that get_config does not expose API key."""
        model = NovaModel(api_key="secret-key")
        config = model.get_config()
        assert "api_key" not in config

    def test_get_config_after_update(self):
        """Test that get_config reflects updates."""
        model = NovaModel(api_key="test-key", temperature=0.7)
        model.update_config(temperature=0.5, max_tokens=1024)
        config = model.get_config()
        assert config["temperature"] == 0.5
        assert config["max_tokens"] == 1024

    def test_get_config_is_not_shallow_copy(self):
        """Test that modifying returned config doesn't affect model state."""
        model = NovaModel(api_key="test-key", temperature=0.7)
        config = model.get_config()
        config["temperature"] = 0.5
        assert model.temperature == 0.7

    def test_get_config_with_all_optional_params(self):
        """Test get_config with all optional parameters set."""
        model = NovaModel(
            api_key="test-key",
            model="custom-model",
            temperature=0.3,
            max_tokens=1500,
            top_p=0.85,
            reasoning_effort="high",
            web_search_options={"search_context_size": "high"},
            stop=["STOP1", "STOP2"],
            base_url="https://custom.api.com",
            stream_options={"include_usage": False},
            custom_param="custom_value",
        )
        config = model.get_config()
        assert config["model"] == "custom-model"
        assert config["temperature"] == 0.3
        assert config["max_tokens"] == 1500
        assert config["top_p"] == 0.85
        assert config["reasoning_effort"] == "high"
        assert config["web_search_options"]["search_context_size"] == "high"
        assert config["stop"] == ["STOP1", "STOP2"]
        assert config["base_url"] == "https://custom.api.com"
        assert config["stream_options"]["include_usage"] is False
        assert config["custom_param"] == "custom_value"


class TestConfigPersistence:
    """Test configuration persistence across operations."""

    def test_config_persists_after_multiple_updates(self):
        """Test that configuration remains consistent after multiple updates."""
        model = NovaModel(api_key="test-key", temperature=0.7)
        model.update_config(temperature=0.5)
        model.update_config(max_tokens=2048)
        model.update_config(top_p=0.95)

        assert model.temperature == 0.5
        assert model.max_tokens == 2048
        assert model.top_p == 0.95

    def test_config_state_consistency(self):
        """Test that get_config and direct attribute access are consistent."""
        model = NovaModel(
            api_key="test-key",
            model="nova-pro-v3",
            temperature=0.6,
            max_tokens=3000,
        )
        config = model.get_config()

        assert config["model"] == model.model
        assert config["temperature"] == model.temperature
        assert config["max_tokens"] == model.max_tokens

    def test_update_then_get_consistency(self):
        """Test consistency between update_config and get_config."""
        model = NovaModel(api_key="test-key")

        new_config = {
            "temperature": 0.4,
            "max_tokens": 2500,
            "model": "nova-lite-v1",
        }
        model.update_config(**new_config)

        retrieved_config = model.get_config()
        assert retrieved_config["temperature"] == new_config["temperature"]
        assert retrieved_config["max_tokens"] == new_config["max_tokens"]
        assert retrieved_config["model"] == new_config["model"]

    def test_additional_params_persistence(self):
        """Test that additional parameters persist correctly."""
        model = NovaModel(api_key="test-key", custom1="value1")
        model.update_config(custom2="value2")

        config = model.get_config()
        assert config["custom1"] == "value1"
        assert config["custom2"] == "value2"

    def test_overwriting_additional_params(self):
        """Test overwriting existing additional parameters."""
        model = NovaModel(api_key="test-key", custom_param="old")
        model.update_config(custom_param="new")

        config = model.get_config()
        assert config["custom_param"] == "new"
        assert model.additional_params["custom_param"] == "new"


class TestConfigEdgeCases:
    """Test edge cases in configuration management."""

    def test_update_with_none_values(self):
        """Test updating with None values."""
        model = NovaModel(api_key="test-key", reasoning_effort="medium")
        model.update_config(reasoning_effort=None)
        assert model.reasoning_effort is None

    def test_get_config_with_complex_nested_objects(self):
        """Test get_config with deeply nested configuration objects."""
        complex_web_opts = {
            "search_context_size": "high",
            "filters": {
                "domains": ["example.com"],
                "date_range": {"start": "2024-01-01", "end": "2024-12-31"},
            },
        }
        model = NovaModel(api_key="test-key", web_search_options=complex_web_opts)
        config = model.get_config()
        assert config["web_search_options"] == complex_web_opts

    def test_update_with_special_characters_in_strings(self):
        """Test updating with special characters in string values."""
        model = NovaModel(api_key="test-key")
        model.update_config(
            model="model-with-special!@#$%",
            base_url="https://api.example.com/v1/chat?key=value&param=test",
        )
        assert model.model == "model-with-special!@#$%"
        assert "key=value" in model.base_url

    def test_update_with_unicode_strings(self):
        """Test updating with unicode strings."""
        model = NovaModel(api_key="test-key")
        model.update_config(custom_param="Hello 世界 🌍")
        assert model.additional_params["custom_param"] == "Hello 世界 🌍"

    def test_update_config_with_boolean_values(self):
        """Test updating with boolean values."""
        model = NovaModel(api_key="test-key")
        model.update_config(enable_feature=True, disable_feature=False)
        assert model.additional_params["enable_feature"] is True
        assert model.additional_params["disable_feature"] is False

    def test_update_config_with_numeric_types(self):
        """Test updating with various numeric types."""
        model = NovaModel(api_key="test-key")
        model.update_config(
            int_param=42, float_param=3.14, negative_int=-10, negative_float=-2.5
        )
        assert model.additional_params["int_param"] == 42
        assert model.additional_params["float_param"] == 3.14
        assert model.additional_params["negative_int"] == -10
        assert model.additional_params["negative_float"] == -2.5

    def test_get_config_returns_current_state_not_cached(self):
        """Test that get_config always returns current state, not cached."""
        model = NovaModel(api_key="test-key", temperature=0.7)
        config1 = model.get_config()

        model.update_config(temperature=0.5)
        config2 = model.get_config()

        assert config1["temperature"] == 0.7
        assert config2["temperature"] == 0.5
