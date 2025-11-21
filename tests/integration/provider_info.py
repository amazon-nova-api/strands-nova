"""Provider information for integration tests."""

import pytest


class ProviderInfo:
    """Provider information for parametrizing tests."""
    
    def __init__(self, name: str):
        self.name = name
        self.mark = pytest.mark.integration


# Nova provider marker
nova = ProviderInfo("nova")
