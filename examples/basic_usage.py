#!/usr/bin/env python
"""Basic usage example of strands-nova."""

import os

from dotenv import load_dotenv
from strands import Agent

from strands_nova import NovaModel


def main():
    """Run basic example."""
    # Load environment variables
    load_dotenv()

    # Check for API key
    if not os.getenv("NOVA_API_KEY"):
        print("Error: NOVA_API_KEY not found in environment variables.")
        print("Please set: export NOVA_API_KEY='your-api-key'")
        print("Get your API key from: https://nova.amazon.com/apis")
        return

    # Initialize model
    model = NovaModel(model="nova-premier-v1", temperature=0.3, max_tokens=512)

    # Create agent
    agent = Agent(model=model, callback_handler=None)

    # Simple conversation
    print("Question: What is the Turing Test in Computer Science?")
    response = agent("What is the Turing test in Computer Science?")
    print(f"Response: {response.message}\n")


if __name__ == "__main__":
    main()
