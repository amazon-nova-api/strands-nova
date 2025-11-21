# Strands Nova

Amazon Nova API model provider for Strands Agents SDK.

This package provides integration between Amazon's Nova API (an OpenAI-compatible API for Amazon's Nova family of models) and the Strands Agents SDK, enabling access to Nova Pro, Nova Premier, reasoning models, and other capabilities.

## Installation

```bash
pip install strands-nova
```

Or with development dependencies:

```bash
pip install strands-nova[dev]
```

## Quick Start

```python
import asyncio
import os
from strands_nova import NovaModel

async def main():
    # Initialize the Nova model
    model = NovaModel(
        api_key=os.getenv("NOVA_API_KEY"),
        model_id="nova-lite-v2",  # Required parameter
        params={
            "max_tokens": 1000,
            "temperature": 0.7,
        }
    )
    
    # Simple conversation
    messages = [
        {
            "role": "user",
            "content": [{"text": "Hello! How are you?"}]
        }
    ]
    
    # Stream the response
    async for chunk in model.stream(messages=messages):
        if "contentBlockDelta" in chunk:
            delta = chunk["contentBlockDelta"]["delta"]
            if "text" in delta:
                print(delta["text"], end="", flush=True)

asyncio.run(main())
```

## Features

- ✅ **Direct Nova API Integration**: Uses httpx for direct API calls (no OpenAI SDK dependency)
- ✅ **Streaming Support**: Full support for streaming responses with Server-Sent Events (SSE)
- ✅ **Tool Calling**: Complete support for function/tool calling
- ✅ **Multi-modal Inputs**: Support for text, images, documents, and audio
- ✅ **Error Handling**: Proper handling of Nova API errors including throttling and context overflow
- ✅ **Reasoning Content**: Support for models with reasoning capabilities
- ✅ **Strands Compatible**: Fully compatible with the Strands Agents SDK

## Supported Models

- `nova-pro-v1` - High-performance model
- `nova-lite-v2` - Lightweight, fast model
- `nova-micro-v1` - Ultra-lightweight model
- `nova-premier-v1` - Premium tier model
- `nova-2o-omni` - Omni-modal model
- `nova-deep-research-v1` - Research-focused model

To see all available models for your account:

```bash
curl -L 'https://api.nova.amazon.com/v1/models' \
  -H 'Authorization: Bearer YOUR_API_KEY'
```

## Configuration

### Basic Configuration

```python
from strands_nova import NovaModel

model = NovaModel(
    api_key="your-api-key",              # Required: Nova API key
    model_id="nova-pro-v1",              # Required: Model ID
    base_url="https://api.nova.amazon.com/v1",  # Optional, default shown
    timeout=300.0,                       # Optional, request timeout in seconds
    params={                             # Optional: Model parameters
        "max_tokens": 4096,              # Maximum tokens to generate
        "max_completion_tokens": 4096,   # Alternative to max_tokens
        "temperature": 0.7,              # Sampling temperature (0.0-1.0)
        "top_p": 0.9,                    # Nucleus sampling (0.0-1.0)
        "reasoning_effort": "medium",    # For reasoning models: "low", "medium", "high"
        "metadata": {},                  # Additional metadata
        "web_search_options": {}         # Web search config (in review)
    }
)
```

**Supported Parameters in `params`:**
- `max_tokens` (int): Maximum tokens to generate (deprecated, use max_completion_tokens)
- `max_completion_tokens` (int): Maximum tokens to generate
- `temperature` (float): Controls randomness (0.0 = deterministic, 1.0 = maximum randomness)
- `top_p` (float): Nucleus sampling threshold
- `reasoning_effort` (str): For reasoning models - "low", "medium", or "high"
- `metadata` (dict): Additional request metadata
- `web_search_options` (dict): Web search configuration (currently in review)
- Any additional parameters will be passed through to the API for future extensibility

### Using with Strands Agents

```python
from strands import Agent
from strands_nova import NovaModel

model = NovaModel(
    api_key="your-api-key",
    model_id="nova-pro-v1",  # Required
    params={
        "temperature": 0.7,
        "max_tokens": 2048
    }
)

agent = Agent(
    model=model,
    system_prompt="You are a helpful assistant."
)

# Use the agent
response = await agent.run("Tell me about quantum computing")
```

## Advanced Features

### Tool Calling

```python
tool_specs = [
    {
        "name": "get_weather",
        "description": "Get weather for a location",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }
]

messages = [
    {
        "role": "user",
        "content": [{"text": "What's the weather in Paris?"}]
    }
]

async for chunk in model.stream(
    messages=messages,
    tool_specs=tool_specs,
    tool_choice={"auto": {}}
):
    # Process chunks
    pass
```

### Multi-modal Inputs

```python
import base64

# Image input
with open("image.png", "rb") as f:
    image_bytes = f.read()

messages = [
    {
        "role": "user",
        "content": [
            {"text": "What's in this image?"},
            {
                "image": {
                    "format": "png",
                    "source": {"bytes": image_bytes}
                }
            }
        ]
    }
]

async for chunk in model.stream(messages=messages):
    # Process response
    pass
```

### Error Handling

```python
from strands.types.exceptions import (
    ContextWindowOverflowException,
    ModelThrottledException
)

try:
    async for chunk in model.stream(messages=messages):
        # Process chunks
        pass
except ContextWindowOverflowException as e:
    print(f"Context window exceeded: {e}")
    # Reduce message history or content size
except ModelThrottledException as e:
    print(f"Rate limited: {e}")
    # Implement retry with backoff
except Exception as e:
    print(f"API error: {e}")
```

## API Reference

### NovaModel

#### Constructor

```python
NovaModel(
    api_key: str,
    model_id: str,
    base_url: str = "https://api.nova.amazon.com/v1",
    timeout: float = 300.0,
    params: Optional[dict[str, Any]] = None,
    **extra_config: Any
)
```

**Parameters:**
- `api_key` (str, required): Nova API key for authentication
- `model_id` (str, required): Model ID (e.g., "nova-pro-v1", "nova-lite-v2")
- `base_url` (str, optional): Base URL for Nova API (default: "https://api.nova.amazon.com/v1")
- `timeout` (float, optional): Request timeout in seconds (default: 300.0)
- `params` (dict, optional): Model parameters (see NovaModelParams for typed options)
- `**extra_config` (Any): Additional configuration for future extensibility

#### Methods

##### stream()

Stream conversation with the Nova model.

```python
async def stream(
    messages: Messages,
    tool_specs: Optional[list[ToolSpec]] = None,
    system_prompt: Optional[str] = None,
    *,
    tool_choice: ToolChoice | None = None,
    **kwargs
) -> AsyncGenerator[StreamEvent, None]
```

##### update_config()

Update the model configuration.

```python
def update_config(**model_config: NovaConfig) -> None
```

##### get_config()

Get the current model configuration.

```python
def get_config() -> NovaConfig
```

## Getting Your API Key

1. Visit [https://nova.amazon.com/dev-apis](https://nova.amazon.com/dev-apis)
2. Sign in with your Amazon account
3. Generate your API key
4. Set it as an environment variable: `export NOVA_API_KEY=your-key-here`

## Configuration Types

### NovaModelParams

The `NovaModelParams` TypedDict provides type hints for all supported parameters:

```python
class NovaModelParams(TypedDict, total=False):
    max_tokens: int                      # Deprecated, use max_completion_tokens
    max_completion_tokens: int           # Maximum tokens to generate
    temperature: float                   # Sampling temperature (0.0-1.0)
    top_p: float                        # Nucleus sampling (0.0-1.0)
    reasoning_effort: str               # "low", "medium", or "high"
    metadata: dict[str, Any]            # Additional metadata
    web_search_options: dict[str, Any]  # Web search config (in review)
```

**Note:** The configuration is extensible - any additional parameters not explicitly typed will be passed through to the Nova API, allowing support for future API additions without package updates.

### Auto-set Parameters

These parameters are automatically configured:
- `stream`: Always set to `true` for streaming support
- `stream_options`: Automatically includes `{"include_usage": True}` for token usage tracking

## Error Codes

- `400` - ValidationException (including context length errors)
- `404` - ModelNotFoundException
- `429` - ThrottlingException (rate limits)
- `500` - ModelException

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check .
black --check .

# Type checking
mypy src/strands_nova
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues with the Nova API itself, contact: nova-api-support@amazon.com
For issues with this package, please open a GitHub issue.

## Acknowledgments

This package is built on top of the [Strands Agents SDK](https://github.com/strands-agents/sdk-python).
