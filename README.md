# strands-nova

[![PyPI version](https://badge.fury.io/py/strands-nova.svg)](https://badge.fury.io/py/strands-nova)
[![Python Support](https://img.shields.io/pypi/pyversions/strands-nova.svg)](https://pypi.org/project/strands-nova/)
[![Tests](https://github.com/aidendef/strands-nova/actions/workflows/test.yml/badge.svg)](https://github.com/aidendef/strands-nova/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Amazon Nova API model provider for [Strands Agents SDK](https://github.com/strands-agents/sdk-python)

## Features

- **OpenAI-Compatible API** - Easy integration with familiar API format
- **Multiple Model Support** - Access Nova Pro, Nova Premier, reasoning models, and more
- **Full Streaming Support** - Real-time response streaming with SSE
- **Native Tool Calling** - Built-in function calling capabilities
- **Advanced Features** - Reasoning modes, web search, and image generation
- **Simple Authentication** - Just needs NOVA_API_KEY
- **Easy Integration** - Drop-in replacement for any Strands model provider
- **Type Safe** - Full type hints and mypy support

## Requirements

- Python 3.10+
- Strands Agents SDK 1.17.0+
- Nova API key from [nova.amazon.com](https://nova.amazon.com/apis)

## Installation

```bash
pip install strands-agents strands-nova
```

## Quick Start

### Basic Usage

```python
from strands_nova import NovaModel
from strands import Agent

# Initialize Nova model
model = NovaModel(
    api_key="your-nova-api-key",  # or set NOVA_API_KEY env var
    model="nova-premier-v1",
    temperature=0.7,
    max_tokens=2048
)

# Create an agent
agent = Agent(model=model)

# Use the agent
response = await agent.invoke_async("What are the benefits of using Amazon Nova models?")
print(response.message)
```

### Streaming Responses

```python
import asyncio
from strands_nova import NovaModel

async def stream_example():
    model = NovaModel(api_key="your-api-key")
    
    async for event in model.stream("Write a short story about AI"):
        if "contentBlockDelta" in event:
            delta = event["contentBlockDelta"]["delta"]
            if "text" in delta:
                print(delta["text"], end="", flush=True)

asyncio.run(stream_example())
```

### With System Prompt

```python
model = NovaModel(api_key="your-api-key")

async for event in model.stream(
    "Explain Python list comprehensions",
    system_prompt="You are a helpful coding assistant. Provide concise answers."
):
    # Process streaming events
    pass
```

## Configuration

### Environment Variables

```bash
export NOVA_API_KEY="your-api-key"
```

Get your API key from: https://nova.amazon.com/apis

### Model Parameters

```python
model = NovaModel(
    model="nova-premier-v1",     # Model ID
    temperature=0.7,             # Sampling temperature (0.0-1.0)
    max_tokens=4096,             # Maximum tokens to generate
    top_p=0.9,                   # Nucleus sampling parameter
    stop=["\\n\\n"],             # Stop sequences
    reasoning_effort="medium",   # For reasoning models: low/medium/high (NOTE: Not currently supported)
    web_search_options={         # Enable web search
        "search_context_size": "low"
    }
)
```

### Available Models

- **nova-premier-v1** - Most capable model with 1M token context window
- **nova-pro-v1** - Multimodal model (text, images, videos) with 300K context
- **nova-lite-v1** - Multimodal model with 300K context
- **nova-micro-v1** - Text-only model with 128K context
- **Nova Pro v3** - Latest Nova Pro version
- **nova-orchestrator-v1** - Image generation capabilities
- **nova-deep-research-v1** - Long research tasks

**Note**: Reasoning models are not yet available. Waiting on model name from AWS.

Discover available models:
```bash
curl -L 'https://api.nova.amazon.com/v1/models' \
-H 'Authorization: Bearer YOUR_API_KEY'
```

### Dynamic Configuration

```python
# Update configuration at runtime
model.update_config(
    temperature=0.2,
    max_tokens=1024
)

# Get current configuration
config = model.get_config()
print(config)
```

## Examples

### Reasoning Model

**Note**: Reasoning models are not yet available. The `reasoning_effort` parameter is implemented but not supported by current models.

```python
from strands_nova import NovaModel
from strands import Agent

# TODO: Update with actual reasoning model name when available from AWS
model = NovaModel(
    api_key="your-api-key",
    model="nova-premier-v1",  # Placeholder until reasoning model is available
    # reasoning_effort="high"  # Not currently supported - will raise error
)
agent = Agent(model=model)

response = await agent.invoke_async(
    "What is the sum of all prime numbers between 1 and 100?"
)
```

### Web Search Integration

```python
model = NovaModel(
    api_key="your-api-key",
    model="nova-premier-v1",
    web_search_options={"search_context_size": "low"}
)
agent = Agent(model=model)

response = await agent.invoke_async("What is the current Amazon stock price?")
```

### Tool/Function Calling

```python
from strands_nova import NovaModel
from strands import Agent

model = NovaModel(api_key="your-api-key")

# Define tool specs using Strands SDK format
tool_specs = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "inputSchema": {  # Note: Strands SDK uses "inputSchema" not "parameters"
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and country, e.g., Seattle, USA"
                }
            },
            "required": ["location"]
        }
    }
]

# Use with agent
agent = Agent(model=model, tools=tool_specs)
response = await agent.invoke_async("What's the weather in Tokyo?")
```

### Multi-turn Conversations

```python
from strands import Agent

agent = Agent(model=NovaModel(api_key="your-api-key"))

# First turn
response1 = await agent.invoke_async("Tell me a story about a robot.")
print(response1.message)

# Second turn (with context)
response2 = await agent.invoke_async("What happened next?")
print(response2.message)
```

## Testing

### Run Unit Tests

```bash
pip install -e ".[dev]"
pytest tests/unit/ -v
```

### Run Integration Tests

Requires `NOVA_API_KEY` environment variable:

```bash
export NOVA_API_KEY="your-api-key"
pytest tests/integration/ -v
```

## API Reference

### NovaModel

Main model class that implements the Strands `Model` abstract base class.

#### Methods

- `stream(messages, tool_specs, system_prompt, tool_choice, system_prompt_content, **kwargs)` - Stream responses from Nova API
  - `messages` - Messages to process
  - `tool_specs` - List of tool specifications (Strands SDK format with `inputSchema`)
  - `system_prompt` - Optional system message (text)
  - `tool_choice` - Optional tool choice strategy (auto/any/specific tool)
  - `system_prompt_content` - Optional structured system prompt content blocks
- `get_config()` - Get current model configuration
- `update_config(**kwargs)` - Update model configuration
- `structured_output()` - Not yet implemented (raises NotImplementedError)

#### Parameters

- `model` - Model identifier (e.g., "nova-premier-v1")
- `api_key` - Nova API key (or set NOVA_API_KEY env var)
- `temperature` - Controls randomness (0.0-1.0, default: 0.7)
- `max_tokens` - Maximum tokens in response (default: 4096)
- `top_p` - Nucleus sampling threshold (default: 0.9)
- `reasoning_effort` - For reasoning models: "low", "medium", or "high" (NOTE: Not currently supported)
- `web_search_options` - Dict with web search config (e.g., {"search_context_size": "low"})
- `stop` - List of stop sequences

### NovaModelError

Exception raised for Nova API-specific errors.

## Advanced Features

### Reasoning Models

**Note**: Reasoning models are not yet available. Waiting on model name from AWS.

The `reasoning_effort` parameter is implemented but currently not supported by available models:

```python
# TODO: Update when reasoning model is available
model = NovaModel(
    model="nova-premier-v1",  # Placeholder
    # reasoning_effort="high"  # Not currently supported - will raise error
)
```

### Web Search

Enable web search for up-to-date information:

```python
model = NovaModel(
    model="nova-premier-v1",
    web_search_options={
        "search_context_size": "low"  # low, medium, or high
    }
)
```

### Image Generation

Use orchestrator model for image generation:

```python
model = NovaModel(model="nova-orchestrator-v1")
response = await agent.invoke_async("Draw me a picture of a sunset over mountains")
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Support

- **Issue Tracker**: [GitHub Issues](https://github.com/aidendef/strands-nova/issues)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Links

- [Strands Agents SDK](https://github.com/strands-agents/sdk-python)
- [Nova API Documentation](https://api.nova.amazon.com/docs)
- [Get API Key](https://nova.amazon.com/apis)
- [PyPI Package](https://pypi.org/project/strands-nova/)