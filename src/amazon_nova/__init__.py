"""Amazon Nova API model provider for Strands Agents SDK.

This package provides integration between Amazon's Nova API (an OpenAI-compatible
API for Amazon's Nova family of models) and the Strands Agents SDK, enabling
access to Nova Pro, Nova Premier, reasoning models, and image generation capabilities.
"""

from .nova import (
    NovaAPIModel,
    NovaSystemTool,
    NovaAPIException,
    NovaValidationException,
    NovaModelNotFoundException,
    NovaModelException,
)

__version__ = "1.0.1"
__all__ = [
    "NovaAPIModel",
    "NovaSystemTool",
    "NovaAPIException",
    "NovaValidationException",
    "NovaModelNotFoundException",
    "NovaModelException",
]


# Convenience exports for common use cases
def create_nova_model(api_key: str, **kwargs) -> NovaAPIModel:
    """Create a Nova model instance with default settings.

    Args:
        api_key: Nova API key (required)
        **kwargs: Additional model parameters (model_id, params, base_url, timeout)

    Returns:
        Configured NovaAPIModel instance
    """
    return NovaAPIModel(api_key=api_key, **kwargs)
