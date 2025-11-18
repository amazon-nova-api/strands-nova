#!/usr/bin/env python
"""Streaming example for strands-nova."""

import asyncio
import os
from dotenv import load_dotenv

from strands_nova import NovaModel


async def basic_streaming():
    """Demonstrate basic streaming."""
    print("=== Basic Streaming Example ===\n")
    
    model = NovaModel(
        model="nova-premier-v1",
        temperature=0.7,
        max_tokens=500
    )
    
    print("Question: Write a short story about a robot learning to paint.\n")
    print("Response: ", end="", flush=True)
    
    async for event in model.stream("Write a short story about a robot learning to paint."):
        if "contentBlockDelta" in event:
            delta = event["contentBlockDelta"]["delta"]
            if "text" in delta:
                print(delta["text"], end="", flush=True)
    
    print("\n")


async def streaming_with_system_prompt():
    """Demonstrate streaming with system prompt."""
    print("\n=== Streaming with System Prompt ===\n")
    
    model = NovaModel(
        model="nova-premier-v1",
        temperature=0.3
    )
    
    system_prompt = "You are a helpful coding assistant. Provide concise, well-commented code examples."
    
    print("Question: Write a Python function to merge two sorted lists.\n")
    print("Response: ", end="", flush=True)
    
    async for event in model.stream(
        "Write a Python function to merge two sorted lists.",
        system_prompt=system_prompt
    ):
        if "contentBlockDelta" in event:
            delta = event["contentBlockDelta"]["delta"]
            if "text" in delta:
                print(delta["text"], end="", flush=True)
    
    print("\n")


async def reasoning_model_example():
    """Demonstrate reasoning model with reasoning_effort parameter."""
    print("\n=== Reasoning Model Example ===\n")
    
    model = NovaModel(
        model="mumbai-flintflex-reasoning-v3",
        reasoning_effort="medium",
        temperature=0.5
    )
    
    print("Question: What is the sum of all prime numbers between 1 and 100?\n")
    print("Response: ", end="", flush=True)
    
    async for event in model.stream(
        "What is the sum of all prime numbers between 1 and 100?"
    ):
        if "contentBlockDelta" in event:
            delta = event["contentBlockDelta"]["delta"]
            if "text" in delta:
                print(delta["text"], end="", flush=True)
    
    print("\n")


async def web_search_example():
    """Demonstrate web search capabilities."""
    print("\n=== Web Search Example ===\n")
    
    model = NovaModel(
        model="nova-premier-v1",
        web_search_options={"search_context_size": "low"}
    )
    
    print("Question: What is the current Amazon stock price?\n")
    print("Response: ", end="", flush=True)
    
    async for event in model.stream(
        "What is the current Amazon stock price?",
        system_prompt="You are a helpful financial assistant."
    ):
        if "contentBlockDelta" in event:
            delta = event["contentBlockDelta"]["delta"]
            if "text" in delta:
                print(delta["text"], end="", flush=True)
    
    print("\n")


async def main():
    """Run all streaming examples."""
    # Load environment variables
    load_dotenv()
    
    # Check for API key
    if not os.getenv("NOVA_API_KEY"):
        print("Error: NOVA_API_KEY not found in environment variables.")
        print("Please set: export NOVA_API_KEY='your-api-key'")
        print("Get your API key from: https://internal.nova.amazon.com/apis")
        return
    
    # Run examples
    await basic_streaming()
    await streaming_with_system_prompt()
    
    # Optional: Run reasoning and web search examples
    # Note: These require specific model access
    try:
        await reasoning_model_example()
    except Exception as e:
        print(f"\nReasoning model example skipped: {e}\n")
    
    try:
        await web_search_example()
    except Exception as e:
        print(f"\nWeb search example skipped: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
