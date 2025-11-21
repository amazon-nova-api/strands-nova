#!/usr/bin/env python
"""Streaming example for strands-nova using Strands Agent."""

import asyncio
from dotenv import load_dotenv
from strands import Agent

from strands_nova import NovaModel


def yellow_img():
    """Load yellow image for testing."""
    with open("tests/integration/yellow.png", "rb") as fp:
        return fp.read()


async def basic_streaming():
    """Demonstrate basic streaming with Strands Agent."""

    # Initialize model
    model = NovaModel(
        model_id="nova-pro-v1",
        params={
            "temperature": 0.3,
            "max_tokens": 512,
        },
        stream=True,
        stream_options={"include_usage": True},
    )

    # Create agent
    agent = Agent(model=model, callback_handler=None)

    print("Question: What are the seven colors of the rainbow?\n")
    print("Response: ", end="", flush=True)
    async for event in agent.stream_async("What are the seven colors of the rainbow?"):
        if "data" in event:
            print(event["data"], end="", flush=True)
        # Print usage metadata when available
        elif "result" in event:
            result = event["result"]
            if hasattr(result, "metrics") and result.metrics:
                usage = result.metrics.accumulated_usage
                print(f"\n\nUsage: {usage}")
    print("\n")


async def image_streaming():
    """Demonstrate basic streaming with Strands Agent."""

    # Initialize model
    model = NovaModel(
        model_id="nova-pro-v1",
        params={
            "temperature": 0.3,
            "max_tokens": 512,
        },
        stream=True,
        stream_options={"include_usage": True},
    )

    # Create agent
    agent = Agent(model=model, callback_handler=None)

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
    print("Question: Is this image red, blue, or yellow??\n")
    print("Response: ", end="", flush=True)
    async for event in agent.stream_async(content):
        if "data" in event:
            print(event["data"], end="", flush=True)
    print("\n")


async def main():
    """Run streaming example"""
    # Load environment variables
    load_dotenv()

    # Run examples
    try:
        await basic_streaming()
        await image_streaming()
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
