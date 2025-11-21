"""Provider configuration for integration tests."""

import os
from typing import Callable, Optional
from pytest import mark
from strands.models import Model
from strands_nova import NovaModel


class ProviderInfo:
    """Provider-based info for providers that require an APIKey via environment variables."""

    def __init__(
        self,
        id: str,
        model_id: str,
        factory: Callable[[str], Model],
        environment_variable: Optional[str] = None,
    ) -> None:
        self.id = id
        self.model_id = model_id
        self.model_factory = factory
        self.mark = mark.skipif(
            environment_variable is not None and environment_variable not in os.environ,
            reason=f"{environment_variable} environment variable missing",
        )

    def create_model(self, model_id: str) -> Model:
        return self.model_factory(model_id)


# Nova provider configuration
nova = ProviderInfo(
    id="nova-api",
    model_id="nova-pro-v1",
    environment_variable="NOVA_API_KEY",
    factory=lambda model_id: NovaModel(
        model_id=model_id,
        api_key=os.getenv("NOVA_API_KEY"),
    ),
)
