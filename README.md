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
from strands_nova import NovaAPIModel

async def main():
    # Initialize the Nova model
    model = NovaAPIModel(
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
- ✅ **System Tools**: Support for Nova system tools (grounding, code interpreter)
- ✅ **Structured Output**: Type-safe structured output using Pydantic models
- ✅ **Multi-modal Inputs**: Support for text, images, documents, and audio
- ✅ **Error Handling**: Proper handling of Nova API errors including throttling and context overflow
- ✅ **Reasoning Content**: Support for models with reasoning capabilities
- ✅ **Strands Compatible**: Fully compatible with the Strands Agents SDK

## Supported Models

To see all available models for your account:

```bash
curl -L 'https://api.nova.amazon.com/v1/models' \
  -H 'Authorization: Bearer YOUR_API_KEY'
```

## Configuration

### Basic Configuration

```python
from strands_nova import NovaAPIModel

model = NovaAPIModel(
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
        "system_tools": ["nova_grounding", "nova_code_interpreter"] # Available system tools from Nova API
        "metadata": {},                  # Additional metadata
    }
)
```

**Supported Parameters in `params`:**
- `max_tokens` (int): Maximum tokens to generate (deprecated, use max_completion_tokens)
- `max_completion_tokens` (int): Maximum tokens to generate
- `temperature` (float): Controls randomness (0.0 = deterministic, 1.0 = maximum randomness)
- `top_p` (float): Nucleus sampling threshold
- `reasoning_effort` (str): For reasoning models - "low", "medium", or "high"
- `system_tools` (list): Available system tools from the Nova API - currently nova_grounding and nova_code_interpreter
- `metadata` (dict): Additional request metadata
- Any additional parameters will be passed through to the API for future extensibility

### Using with Strands Agents

```python
from strands import Agent
from strands_nova import NovaAPIModel

model = NovaAPIModel(
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

### Error Handling

The package provides custom exceptions for Nova-specific errors, along with support for Strands SDK exceptions:

```python
from strands.types.exceptions import (
    ContextWindowOverflowException,
    ModelThrottledException
)
from strands_nova import (
    NovaAPIException,           # Base exception for all Nova API errors
    NovaValidationException,    # HTTP 400 validation errors
    NovaModelNotFoundException, # HTTP 404 model not found
    NovaModelException          # HTTP 500 internal model errors
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
except NovaValidationException as e:
    print(f"Validation error: {e}")
    # Check request parameters
except NovaModelNotFoundException as e:
    print(f"Model not found: {e}")
    # Verify model_id is correct
except NovaModelException as e:
    print(f"Model error: {e}")
    # Nova API internal error
except NovaAPIException as e:
    print(f"Nova API error: {e}")
    # Catch-all for other Nova errors
except Exception as e:
    print(f"Unexpected error: {e}")
```

**Custom Nova Exceptions:**
- `NovaAPIException`: Base exception for all Nova API errors
- `NovaValidationException`: Raised for HTTP 400 validation errors (malformed requests, invalid parameters)
- `NovaModelNotFoundException`: Raised for HTTP 404 errors (model not found or inaccessible)
- `NovaModelException`: Raised for HTTP 500 errors (internal model service errors)

**Strands SDK Exceptions:**
- `ContextWindowOverflowException`: Raised when input exceeds the model's context window
- `ModelThrottledException`: Raised when requests are rate-limited (HTTP 429)

## Getting Your API Key

1. Visit [https://nova.amazon.com/dev-apis](https://nova.amazon.com/dev-apis)
2. Sign in with your Amazon account
3. Generate your API key
4. Set it as an environment variable: `export NOVA_API_KEY=your-key-here`

## Configuration Types

### NovaAPIModelParams

The `NovaAPIModelParams` TypedDict provides type hints for all supported parameters:

```python
class NovaAPIModelParams(TypedDict, total=False):
    max_tokens: int                          # Deprecated, use max_completion_tokens
    max_completion_tokens: int               # Maximum tokens to generate
    temperature: float                       # Sampling temperature (0.0-1.0)
    top_p: float                            # Nucleus sampling (0.0-1.0)
    reasoning_effort: str                   # "low", "medium", or "high"
    metadata: dict[str, Any]                # Additional metadata
    system_tools: list[Union[NovaSystemTool, str]]  # Built-in system tools
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check .

# Type checking
mypy src/strands_nova
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues with the Nova API itself, contact: <email>
For issues with this package, please open a GitHub issue.

## Acknowledgments

This package is built for [Strands Agents SDK](https://github.com/strands-agents/sdk-python).
