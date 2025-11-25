"""Script to display model categorization for testing.

Run this to see which models are in which categories.
"""

import os
import sys

import httpx
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.integration.conftest import categorize_model

load_dotenv()


def main():
    """Display model categories."""
    api_key = os.getenv("NOVA_API_KEY")
    if not api_key:
        print("Error: NOVA_API_KEY not found")
        return

    models_url = "https://api.nova.amazon.com/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    response = httpx.get(models_url, headers=headers, timeout=30.0)
    response.raise_for_status()
    data = response.json()
    print(data)

    models = [m for m in data.get("data", [])]

    print("=" * 80)
    print("NOVA MODEL CATEGORIZATION")
    print("=" * 80)

    # Group by category
    categories_map = {}
    for model in models:
        model_id = model.get("id")
        categories = categorize_model(model)

        for category in categories:
            if category not in categories_map:
                categories_map[category] = []
            categories_map[category].append(model_id)

    # Display categories
    for category in sorted(categories_map.keys()):
        print(f"\n{category.upper().replace('_', ' ')}:")
        print("-" * 80)
        for model_id in categories_map[category]:
            print(f"  • {model_id}")

    print("\n" + "=" * 80)
    print(f"Total production models: {len(models)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
