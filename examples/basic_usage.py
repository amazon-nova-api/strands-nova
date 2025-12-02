#!/usr/bin/env python
"""Basic usage example of strands-nova."""

from dotenv import load_dotenv
from strands import Agent

from strands_amazon_nova import NovaAPIModel


def yellow_img():
    """Load yellow image for testing."""
    with open("tests/integration/yellow.png", "rb") as fp:
        return fp.read()


def main():
    """Run basic example."""
    # Load environment variables
    load_dotenv()
    try:
        model = NovaAPIModel(
            model_id="nova-pro-v1",
            params={
                "temperature": 0.3,
                "max_tokens": 512,
            },
            stream=False,
        )
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Create agent
    agent = Agent(
        model=model, callback_handler=None, system_prompt="You are a helpful assistant."
    )

    # Simple conversation
    print("Question: What are the seven colors of the rainbow?")
    response = agent("What are the seven colors of the rainbow?")
    print(f"Response: {response.message}\n")
    print(f"Metadata: {response.metrics.accumulated_usage}\n")

    # Simple conversation
    print("Question: Is this image red, blue, or yellow?")
    content = [
        {"text": "Is this image red, blue, or yellow?"},
        {
            "image": {
                "format": "png",
                "source": {
                    "bytes": yellow_img(),
                },
            },
        },
    ]
    response = agent(content)
    print(f"Response: {response.message}\n")


if __name__ == "__main__":
    main()
