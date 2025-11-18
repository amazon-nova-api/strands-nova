"""Amazon Nova API model provider for Strands Agents SDK.

This package provides integration between Amazon's Nova API (an OpenAI-compatible
API for Amazon's Nova family of models) and the Strands Agents SDK, enabling
access to Nova Pro, Nova Premier, reasoning models, and image generation capabilities.
"""

from typing import Optional

from .nova import NovaModel, NovaModelError

__version__ = "0.1.2"
__all__ = ["NovaModel", "NovaModelError"]


# Convenience exports for common use cases
def create_nova_model(api_key: Optional[str] = None, **kwargs) -> NovaModel:
    """Create a Nova model instance with default settings.

    Args:
        api_key: Nova API key (can also be set via NOVA_API_KEY env var)
        **kwargs: Additional model parameters

    Returns:
        Configured NovaModel instance
    """
    return NovaModel(api_key=api_key, **kwargs)
