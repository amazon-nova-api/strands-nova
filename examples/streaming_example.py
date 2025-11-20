#!/usr/bin/env python
"""Streaming example for strands-nova using Strands Agent."""

import asyncio
import os

from dotenv import load_dotenv
from strands import Agent

from strands_nova import NovaModel


async def basic_streaming():
    """Demonstrate basic streaming with Strands Agent."""

    # Initialize model
    model = NovaModel(
        model="nova-lite-v1", temperature=0.3, max_tokens=512, stream=True
    )

    # Create agent
    agent = Agent(model=model, callback_handler=None)

    print("Question: What is the Turing Test in Computer Science?\n")
    print("Response: ", end="")
    async for event in agent.stream_async(
        "What is the Turing Test in Computer Science?"
    ):
        if "data" in event:
            print(event["data"], end="", flush=True)
    print("\n")


async def main():
    """Run streaming example"""
    # Load environment variables
    load_dotenv()

    # Check for API key
    if not os.getenv("NOVA_API_KEY"):
        print("Error: NOVA_API_KEY not found in environment variables.")
        print("Please set: export NOVA_API_KEY='your-api-key'")
        print("Get your API key from: https://nova.amazon.com/apis")
        return

    # Run examples
    await basic_streaming()


if __name__ == "__main__":
    asyncio.run(main())
