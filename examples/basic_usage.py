#!/usr/bin/env python
"""Basic usage example of strands-nova."""

import asyncio
import os

from dotenv import load_dotenv
from strands import Agent

from strands_nova import NovaModel


async def main():
    """Run basic example."""
    # Load environment variables
    load_dotenv()

    # Check for API key
    if not os.getenv("NOVA_API_KEY"):
        print("Error: NOVA_API_KEY not found in environment variables.")
        print("Please set: export NOVA_API_KEY='your-api-key'")
        print("Get your API key from: https://nova.amazon.com/apis")
        return

    print("=== Basic Nova API Example ===\n")

    # Initialize model
    model = NovaModel(model="nova-premier-v1", temperature=0.7, max_tokens=1024)

    # Create agent
    agent = Agent(model=model)

    # Simple conversation
    print("Question: What are the key benefits of Amazon's Nova models?")
    response = await agent.invoke_async("What are the key benefits of Amazon's Nova models?")
    print(f"Response: {response.message}\n")

    # Technical question
    print("Question: Explain how transformer architecture works in simple terms.")
    response = await agent.invoke_async("Explain how transformer architecture works in simple terms.")
    print(f"Response: {response.message}\n")

    # Example with Nova Pro model
    print("\n=== Using Nova Pro v3 ===\n")
    model_pro = NovaModel(model="Nova Pro v3 (6.x)", temperature=0.5, max_tokens=512)
    agent_pro = Agent(model=model_pro)

    print("Question: Write a Python function to calculate fibonacci numbers.")
    response = await agent_pro.invoke_async("Write a Python function to calculate fibonacci numbers.")
    print(f"Response: {response.message}\n")


if __name__ == "__main__":
    asyncio.run(main())
