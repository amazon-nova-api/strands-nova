"""Nova model provider for Strands Agents SDK.

This module implements a custom model provider for Amazon Nova models,
providing integration with the Strands Agents framework through OpenAI-compatible
APIs.

Key Features:
    - Streaming and non-streaming responses
    - Tool calling support
    - Structured output via Pydantic models
    - Multi-modal content (text, images, audio, documents)
    - Reasoning content support
    - Comprehensive error handling with custom exceptions

Example:
    Basic usage with an agent:
        >>> from strands_nova import NovaAPIModel
        >>> from strands import Agent
        >>>
        >>> model = NovaAPIModel(
        ...     model_id="nova-pro-v1",
        ...     api_key="your-api-key"
        ... )
        >>> agent = Agent(model=model)
        >>> response = agent("Hello, world!")

    With structured output:
        >>> from pydantic import BaseModel
        >>>
        >>> class PersonInfo(BaseModel):
        ...     name: str
        ...     age: int
        ...
        >>> result = agent.structured_output(
        ...     PersonInfo,
        ...     "Extract: John is 30 years old"
        ... )
License: MIT
"""

import base64
import json
import logging
import mimetypes
import os
from enum import Enum
from typing import (
    Any,
    AsyncGenerator,
    Optional,
    Type,
    TypedDict,
    TypeVar,
    Union,
    cast,
    Dict,
)

import httpx
from pydantic import BaseModel
from typing_extensions import Unpack, override

from strands.types.content import ContentBlock, Messages, SystemContentBlock
from strands.types.exceptions import (
    ContextWindowOverflowException,
    ModelThrottledException,
)
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolChoice, ToolResult, ToolSpec, ToolUse
from strands.tools import convert_pydantic_to_tool_spec
from strands.models import Model

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# ==================== Custom Exceptions ====================


class NovaAPIException(Exception):
    """Base exception for all Nova API errors.

    This serves as the parent class for all Nova-specific exceptions,
    allowing consumers to catch all Nova errors with a single except clause.
    """

    pass


class NovaValidationException(NovaAPIException):
    """Raised when Nova API returns a validation error (HTTP 400).

    This typically indicates malformed requests, invalid parameters,
    or unsupported operations.
    """

    pass


class NovaModelNotFoundException(NovaAPIException):
    """Raised when the specified model is not found (HTTP 404).

    This occurs when the model_id doesn't exist or is not accessible
    with the provided credentials.
    """

    pass


class NovaModelException(NovaAPIException):
    """Raised when Nova API encounters a model error (HTTP 500).

    This indicates an internal error within the Nova model service.
    """

    pass


# ==================== Enums ====================


class NovaSystemTool(str, Enum):
    """Predefined Nova system tools.

    These are built-in tools provided by the Nova API. You can also pass
    custom tool names as strings if needed.

    Attributes:
        GROUNDING: Provides grounded responses with citations and sources.
        CODE_INTERPRETER: Interprets and executes code in a sandboxed environment.
    """

    GROUNDING = "nova_grounding"
    CODE_INTERPRETER = "nova_code_interpreter"


# ==================== Message Formatter ====================


class NovaMessageFormatter:
    """Handles formatting of Strands messages to Nova API format.

    This class encapsulates all message formatting logic, providing a clean
    separation of concerns and making the code more maintainable and testable.
    """

    @staticmethod
    def format_content_block(content: ContentBlock, **kwargs: Any) -> dict[str, Any]:
        """Format a Strands content block to Nova-compatible format.

        Converts various content types (text, image, audio, document) into the
        format expected by the Nova API, including proper base64 encoding and
        MIME type handling.

        Args:
            content: A Strands ContentBlock containing one of: text, image, audio, or document.
            **kwargs: Additional keyword arguments for future extensibility.

        Returns:
            Nova-compatible content block dictionary.

        Raises:
            TypeError: If the content block type is not supported.

        Example:
            >>> formatter = NovaMessageFormatter()
            >>> content = {"text": "Hello, world!"}
            >>> formatted = formatter.format_content_block(content)
            >>> formatted
            {'text': 'Hello, world!', 'type': 'text'}
        """
        if "document" in content:
            mime_type = mimetypes.types_map.get(
                f".{content['document']['format']}", "application/octet-stream"
            )
            file_data = base64.b64encode(content["document"]["source"]["bytes"]).decode(
                "utf-8"
            )
            return {
                "file": {
                    "file_data": f"data:{mime_type};base64,{file_data}",
                    "filename": content["document"]["name"],
                },
                "type": "file",
            }

        if "image" in content:
            mime_type = mimetypes.types_map.get(
                f".{content['image']['format']}", "application/octet-stream"
            )
            image_data = base64.b64encode(content["image"]["source"]["bytes"]).decode(
                "utf-8"
            )
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{image_data}",
                },
            }

        if "audio" in content:
            # Handle both raw bytes and already-encoded base64 strings
            audio_source = cast(Dict[str, Any], content)
            audio_bytes = audio_source["audio"]["source"]["bytes"]
            if isinstance(audio_bytes, str):
                # Already base64 encoded
                audio_data = audio_bytes
            else:
                # Raw bytes, need to encode
                audio_data = base64.b64encode(audio_bytes).decode("utf-8")

            audio_format = audio_source["audio"]["format"]

            return {
                "type": "input_audio",
                "input_audio": {
                    "data": audio_data,
                    "format": audio_format,
                },
            }

        if "text" in content:
            return {"text": content["text"], "type": "text"}

        raise TypeError(f"content_type=<{next(iter(content))}> | unsupported type")

    @staticmethod
    def format_tool_call(tool_use: ToolUse, **kwargs: Any) -> dict[str, Any]:
        """Format a Strands tool use to Nova-compatible tool call format.

        Args:
            tool_use: Tool use object from Strands containing tool invocation details.
            **kwargs: Additional keyword arguments for future extensibility.

        Returns:
            Nova-compatible tool call dictionary in OpenAI format.

        Example:
            >>> formatter = NovaMessageFormatter()
            >>> tool_use = {
            ...     "name": "calculator",
            ...     "toolUseId": "123",
            ...     "input": {"operation": "add", "a": 1, "b": 2}
            ... }
            >>> formatted = formatter.format_tool_call(tool_use)
        """
        return {
            "function": {
                "arguments": json.dumps(tool_use["input"]),
                "name": tool_use["name"],
            },
            "id": tool_use["toolUseId"],
            "type": "function",
        }

    @staticmethod
    def format_tool_result(tool_result: ToolResult, **kwargs: Any) -> dict[str, Any]:
        """Format a Strands tool result to Nova-compatible format.

        Args:
            tool_result: Tool result from tool execution.
            **kwargs: Additional keyword arguments for future extensibility.

        Returns:
            Nova-compatible tool message dictionary.

        Example:
            >>> formatter = NovaMessageFormatter()
            >>> tool_result = {
            ...     "toolUseId": "123",
            ...     "content": [{"text": "Result: 3"}]
            ... }
            >>> formatted = formatter.format_tool_result(tool_result)
        """
        contents = cast(
            list[ContentBlock],
            [
                {"text": json.dumps(content["json"])} if "json" in content else content
                for content in tool_result["content"]
            ],
        )

        return {
            "role": "tool",
            "tool_call_id": tool_result["toolUseId"],
            "content": [
                NovaMessageFormatter.format_content_block(content)
                for content in contents
            ],
        }

    @staticmethod
    def format_system_messages(
        system_prompt: Optional[str] = None,
        *,
        system_prompt_content: Optional[list[SystemContentBlock]] = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Format system messages for Nova API.

        Nova uses OpenAI-style system messages in the messages array.
        This method converts Strands system prompts to the appropriate format.

        Args:
            system_prompt: Simple text system prompt.
            system_prompt_content: Structured system content blocks.
            **kwargs: Additional keyword arguments for future extensibility.

        Returns:
            List of formatted system message dictionaries.

        Note:
            If both system_prompt and system_prompt_content are provided,
            system_prompt_content takes precedence.
        """
        if system_prompt and system_prompt_content is None:
            system_prompt_content = [{"text": system_prompt}]

        return [
            {"role": "system", "content": content["text"]}
            for content in system_prompt_content or []
            if "text" in content
        ]

    @staticmethod
    def format_regular_messages(
        messages: Messages, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Format regular conversation messages for Nova API.

        Processes user and assistant messages, handling text content, tool calls,
        and tool results. Filters out unsupported content types like reasoningContent
        in multi-turn conversations.

        Args:
            messages: List of Strands message objects.
            **kwargs: Additional keyword arguments for future extensibility.

        Returns:
            List of formatted message dictionaries.

        Warning:
            reasoningContent is not supported in multi-turn conversations and will
            be filtered out with a warning log.
        """
        formatted_messages = []

        for message in messages:
            contents = message["content"]

            # Warn about unsupported content in multi-turn
            if any("reasoningContent" in content for content in contents):
                logger.warning(
                    "reasoningContent is not supported in multi-turn conversations "
                    "with the Chat Completions API and will be filtered out."
                )

            # Format text, image, audio, document content
            formatted_contents = [
                NovaMessageFormatter.format_content_block(content)
                for content in contents
                if not any(
                    block_type in content
                    for block_type in ["toolResult", "toolUse", "reasoningContent"]
                )
            ]

            # Format tool calls
            formatted_tool_calls = [
                NovaMessageFormatter.format_tool_call(content["toolUse"])
                for content in contents
                if "toolUse" in content
            ]

            # Format tool results (these become separate messages)
            formatted_tool_messages = [
                NovaMessageFormatter.format_tool_result(content["toolResult"])
                for content in contents
                if "toolResult" in content
            ]

            # Build the main message
            formatted_message = {
                "role": message["role"],
                "content": formatted_contents,
                **(
                    {"tool_calls": formatted_tool_calls} if formatted_tool_calls else {}
                ),
            }
            formatted_messages.append(formatted_message)
            formatted_messages.extend(formatted_tool_messages)

        return formatted_messages


# ==================== Main Model Class ====================


class NovaAPIModel(Model):
    """Nova model provider implementation using httpx for direct API access.

    This class provides a complete implementation of the Strands Model interface
    for Amazon Nova models, supporting both streaming and non-streaming responses,
    tool calling, and structured output.

    Attributes:
        base_url: Base URL for Nova API endpoints.
        api_key: Authentication key for Nova API.
        timeout: Request timeout in seconds.
        config: Model configuration including model_id and parameters.

    Example:
        >>> model = NovaAPIModel(
        ...     model_id="nova-pro-v1",
        ...     api_key="your-key",
        ...     params={"temperature": 0.7}
        ... )
        >>> # Use with Strands Agent
        >>> from strands import Agent
        >>> agent = Agent(model=model)
    """

    base_url: str
    api_key: str
    timeout: float

    class NovaAPIModelParams(TypedDict, total=False):
        """Nova API model parameters.

        All parameters are optional and will be passed through to the Nova API.
        Values outside specified ranges may result in API validation errors.

        Attributes:
            max_tokens: Maximum tokens to generate (deprecated, use max_completion_tokens).
            max_completion_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (0.0-1.0). Higher = more random.
            top_p: Nucleus sampling parameter (0.0-1.0). Higher = more diverse.
            reasoning_effort: Reasoning effort level: "low", "medium", or "high".
            metadata: Additional metadata to include with the request.
            system_tools: List of system tools to enable (NovaSystemTool enum or strings).
        """

        max_tokens: int
        max_completion_tokens: int
        temperature: float
        top_p: float
        reasoning_effort: str  # "low" | "medium" | "high"
        metadata: dict[str, Any]
        system_tools: list[Union[NovaSystemTool, str]]

    class NovaConfig(TypedDict, total=False):
        """Configuration options for Nova models.

        Attributes:
            model_id: Model identifier (required). Examples: "nova-pro-v1", "nova-lite-v2".
                See https://api.nova.amazon.com/v1/models for available models.
            params: Model parameters dictionary (optional).
                Supports NovaAPIModelParams fields plus any custom parameters.
        """

        model_id: str
        params: Optional[dict[str, Any]]

    def __init__(
        self,
        model_id: str,
        api_key: Optional[str] = None,
        base_url: str = "https://api.nova.amazon.com/v1",
        timeout: float = 300.0,
        params: Union[NovaAPIModelParams, dict[str, Any], None] = None,
        stream: bool = True,
        stream_options: Optional[dict[str, Any]] = None,
        **extra_config: Any,
    ) -> None:
        """Initialize Nova provider instance.

        Args:
            model_id: Model ID (required). Must be a non-empty string.
            api_key: Nova API key. If None, reads from NOVA_API_KEY environment variable.
            base_url: Base URL for Nova API (default: https://api.nova.amazon.com/v1).
            timeout: Request timeout in seconds (default: 300.0). Must be positive.
            params: Model parameters (optional). Supports typed NovaAPIModelParams or custom dict.
            stream: Whether to stream responses (default: True).
            stream_options: Stream options dict (default: {"include_usage": True}).
            **extra_config: Additional configuration options for future extensibility.

        Raises:
            ValueError: If model_id is empty, api_key is not found, or parameters are invalid.

        Example:
            >>> # With explicit API key
            >>> model = NovaAPIModel(
            ...     model_id="nova-pro-v1",
            ...     api_key="your-key"
            ... )
            >>>
            >>> # With environment variable
            >>> import os
            >>> os.environ["NOVA_API_KEY"] = "your-key"
            >>> model = NovaAPIModel(model_id="nova-pro-v1")
        """
        # Validate model_id
        if not model_id or not isinstance(model_id, str):
            raise ValueError("model_id must be a non-empty string")

        # Validate timeout
        if timeout <= 0:
            raise ValueError(f"timeout must be positive, got {timeout}")

        # Infer API key from environment variable if not provided
        if api_key is None:
            api_key = os.getenv("NOVA_API_KEY")
            if api_key is None:
                raise ValueError(
                    "api_key must be provided either as an argument or through "
                    "the NOVA_API_KEY environment variable"
                )

        # Validate API key
        if not api_key or not isinstance(api_key, str):
            raise ValueError("api_key must be a non-empty string")

        # Validate and process params
        params = params or {}
        if not isinstance(params, dict):
            raise ValueError("params must be a dictionary")

        # Validate temperature if provided
        if "temperature" in params:
            temp = params["temperature"]
            if not isinstance(temp, (int, float)) or not 0.0 <= temp <= 1.0:
                raise ValueError(
                    f"temperature must be a number between 0.0 and 1.0, got {temp}"
                )

        # Validate top_p if provided
        if "top_p" in params:
            top_p = params["top_p"]
            if not isinstance(top_p, (int, float)) or not 0.0 <= top_p <= 1.0:
                raise ValueError(
                    f"top_p must be a number between 0.0 and 1.0, got {top_p}"
                )

        # Validate reasoning_effort if provided
        if "reasoning_effort" in params:
            effort = params["reasoning_effort"]
            if effort not in ["low", "medium", "high"]:
                raise ValueError(
                    f"reasoning_effort must be 'low', 'medium', or 'high', got {effort}"
                )

        # Validate stream_options
        if stream_options is not None and not isinstance(stream_options, dict):
            raise ValueError("stream_options must be a dictionary")

        # Initialize configuration
        self.config: dict[str, Any] = {
            "model_id": model_id,
            "params": params,
        }
        self.config.update(extra_config)

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._stream = stream
        self.stream_options = (
            stream_options if stream_options is not None else {"include_usage": True}
        )

        logger.debug(
            "model_id=<%s> base_url=<%s> timeout=<%s> stream=<%s> | initialized Nova model",
            model_id,
            self.base_url,
            timeout,
            stream,
        )

    @override
    def update_config(self, **model_config: Unpack[NovaConfig]) -> None:  # type: ignore[override]
        """Update the Nova model configuration.

        This method allows dynamic configuration changes at runtime, which can be
        useful when tools need to modify model behavior during agent execution.

        Args:
            **model_config: Configuration overrides (model_id, params, etc.).

        Example:
            >>> model = NovaAPIModel(model_id="nova-pro-v1", api_key="key")
            >>> model.update_config(params={"temperature": 0.9})
            >>> # Temperature is now 0.9 for subsequent requests
        """
        logger.debug("updating config with: %s", model_config)
        self.config.update(model_config)

    @override
    def get_config(self) -> NovaConfig:
        """Get the current Nova model configuration.

        Returns:
            NovaConfig: Current model configuration including model_id and params.

        Example:
            >>> model = NovaAPIModel(model_id="nova-pro-v1", api_key="key")
            >>> config = model.get_config()
            >>> print(config["model_id"])
            'nova-pro-v1'
        """
        return cast(NovaAPIModel.NovaConfig, self.config)

    def _format_request_tool_choice(
        self, tool_choice: ToolChoice | None
    ) -> dict[str, Any]:
        """Format a Strands tool choice to Nova-compatible format.

        Converts Strands tool choice configuration to OpenAI-style tool_choice format.

        Args:
            tool_choice: Tool choice configuration from Strands.

        Returns:
            Nova-compatible tool_choice dictionary.

        Note:
            - {"auto": {}} maps to "auto" (model decides)
            - {"any": {}} maps to "required" (must use a tool)
            - {"tool": {"name": "x"}} maps to specific function selection
            - None or unknown formats default to "auto"
        """
        if not tool_choice:
            return {}

        match tool_choice:
            case {"auto": _}:
                logger.debug("tool_choice=auto | model decides whether to use tools")
                return {"tool_choice": "auto"}
            case {"any": _}:
                logger.debug("tool_choice=required | model must use a tool")
                return {"tool_choice": "required"}
            case {"tool": {"name": tool_name}}:
                logger.debug("tool_choice=%s | forcing specific tool", tool_name)
                return {
                    "tool_choice": {"type": "function", "function": {"name": tool_name}}
                }
            case _:
                logger.warning(
                    "tool_choice=<%s> | unexpected format, defaulting to 'auto'",
                    tool_choice,
                )
                return {"tool_choice": "auto"}

    def format_request(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        tool_choice: ToolChoice | None = None,
        *,
        system_prompt_content: list[SystemContentBlock] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Format a complete Nova API chat request.

        This is the central request formatting method that coordinates all message
        formatting, tool specification conversion, and parameter application.

        Args:
            messages: Conversation messages to process.
            tool_specs: Available tools for the model to use.
            system_prompt: Simple text system prompt.
            tool_choice: Tool selection strategy.
            system_prompt_content: Structured system content (overrides system_prompt).
            **kwargs: Additional keyword arguments for future extensibility.

        Returns:
            Complete Nova API request dictionary ready for HTTP POST.

        Raises:
            TypeError: If a message contains unsupported content types.

        Example:
            >>> request = model.format_request(
            ...     messages=[{"role": "user", "content": [{"text": "Hi"}]}],
            ...     system_prompt="You are a helpful assistant"
            ... )
        """
        # Format messages using the formatter class
        formatted_messages = NovaMessageFormatter.format_system_messages(
            system_prompt, system_prompt_content=system_prompt_content
        )
        formatted_messages.extend(
            NovaMessageFormatter.format_regular_messages(messages)
        )

        # Filter out empty messages (messages with no content and no tool_calls)
        formatted_messages = [
            msg for msg in formatted_messages if msg["content"] or "tool_calls" in msg
        ]

        # Build the request
        request = {
            "messages": formatted_messages,
            "model": self.config["model_id"],
            "stream": self._stream,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": tool_spec["name"],
                        "description": tool_spec["description"],
                        "parameters": tool_spec["inputSchema"]["json"],
                    },
                }
                for tool_spec in tool_specs or []
            ],
            **self._format_request_tool_choice(tool_choice),
            **cast(dict[str, Any], self.config.get("params", {})),
        }

        if self.stream_options:
            request["stream_options"] = self.stream_options

        return request

    def format_chunk(self, event: dict[str, Any], **kwargs: Any) -> StreamEvent:
        """Format a Nova response event into a standardized Strands StreamEvent.

        Converts Nova's streaming event format to the Strands StreamEvent protocol,
        handling all event types including text, tools, reasoning, and metadata.

        Args:
            event: Raw event from Nova API with 'chunk_type' and 'data'.
            **kwargs: Additional keyword arguments for future extensibility.

        Returns:
            Formatted StreamEvent conforming to Strands protocol.

        Raises:
            RuntimeError: If chunk_type is not recognized.

        Supported chunk_types:
            - message_start: Beginning of a response
            - content_start: Start of a content block
            - content_delta: Incremental content update
            - content_stop: End of a content block
            - message_stop: End of response with stop reason
            - metadata: Usage and metrics information
        """
        match event["chunk_type"]:
            case "message_start":
                return {"messageStart": {"role": "assistant"}}

            case "content_start":
                if event["data_type"] == "tool":
                    return {
                        "contentBlockStart": {
                            "start": {
                                "toolUse": {
                                    "name": event["data"]["name"],
                                    "toolUseId": event["data"]["id"],
                                }
                            }
                        }
                    }
                return {"contentBlockStart": {"start": {}}}

            case "content_delta":
                if event["data_type"] == "tool":
                    return {
                        "contentBlockDelta": {
                            "delta": {
                                "toolUse": {"input": event["data"]["arguments"] or ""}
                            }
                        }
                    }
                if event["data_type"] == "reasoning_content":
                    return {
                        "contentBlockDelta": {
                            "delta": {"reasoningContent": {"text": event["data"]}}
                        }
                    }
                return {"contentBlockDelta": {"delta": {"text": event["data"]}}}

            case "content_stop":
                return {"contentBlockStop": {}}

            case "message_stop":
                match event["data"]:
                    case "tool_calls":
                        return {"messageStop": {"stopReason": "tool_use"}}
                    case "length":
                        return {"messageStop": {"stopReason": "max_tokens"}}
                    case _:
                        return {"messageStop": {"stopReason": "end_turn"}}

            case "metadata":
                return {
                    "metadata": {
                        "usage": {
                            "inputTokens": event["data"]["prompt_tokens"],
                            "outputTokens": event["data"]["completion_tokens"],
                            "totalTokens": event["data"]["total_tokens"],
                        },
                        "metrics": {
                            "latencyMs": 0,  # TODO: Implement actual latency tracking
                        },
                    },
                }

            case _:
                raise RuntimeError(f"chunk_type=<{event['chunk_type']}> | unknown type")

    def _handle_api_error(self, response: httpx.Response) -> None:
        """Handle Nova API errors and raise appropriate custom exceptions.

        Analyzes HTTP status codes and error messages to raise specific exception
        types that consumers can catch and handle appropriately.

        Args:
            response: HTTP response from Nova API containing an error.

        Raises:
            ContextWindowOverflowException: HTTP 400 with context/length errors.
            NovaValidationException: HTTP 400 with other validation errors.
            NovaModelNotFoundException: HTTP 404 - model not found.
            ModelThrottledException: HTTP 429 - rate limit exceeded.
            NovaModelException: HTTP 500 - internal model error.
            NovaAPIException: Other unexpected API errors.

        Nova API Error Mapping:
            - 400: ValidationException, ContextWindowOverflow
            - 404: ModelNotFoundException
            - 429: ThrottlingException
            - 500: ModelException, InternalServerException
            - Others: Generic NovaAPIException
        """
        try:
            error_data = response.json()
            error_message = error_data.get("error", {}).get("message", str(error_data))
        except Exception:
            error_message = response.text

        logger.error(
            "status_code=<%s> error_message=<%s> | Nova API error",
            response.status_code,
            error_message,
        )

        if response.status_code == 400:
            # Check if it's context window related
            if "context" in error_message.lower() or "length" in error_message.lower():
                logger.warning("Context window overflow detected")
                raise ContextWindowOverflowException(error_message)
            raise NovaValidationException(f"Nova API validation error: {error_message}")
        elif response.status_code == 404:
            raise NovaModelNotFoundException(f"Nova model not found: {error_message}")
        elif response.status_code == 429:
            logger.warning("Rate limit exceeded")
            raise ModelThrottledException(error_message)
        elif response.status_code == 500:
            raise NovaModelException(f"Nova API model error: {error_message}")
        else:
            raise NovaAPIException(
                f"Nova API error ({response.status_code}): {error_message}"
            )

    async def _parse_sse_stream(
        self, response: httpx.Response
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Parse Server-Sent Events (SSE) stream from Nova API.

        Handles the SSE protocol used by Nova for streaming responses, parsing
        each event and yielding structured data.

        Args:
            response: Streaming HTTP response from Nova API.

        Yields:
            Parsed JSON events from the SSE stream.

        Note:
            - Ignores empty lines and non-data lines
            - Stops on [DONE] sentinel
            - Logs warnings for malformed JSON
        """
        async for line in response.aiter_lines():
            line = line.strip()
            if not line:
                continue

            if line.startswith("data: "):
                data = line[6:]  # Remove "data: " prefix
                if data == "[DONE]":
                    logger.debug("Received [DONE] sentinel, ending stream")
                    break

                try:
                    yield json.loads(data)
                except json.JSONDecodeError as e:
                    logger.warning(
                        "data=<%s> error=<%s> | Failed to parse SSE data",
                        data[:100] if len(data) > 100 else data,
                        str(e),
                    )
                    continue

    @override
    async def stream(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        *,
        tool_choice: ToolChoice | None = None,
        system_prompt_content: Optional[list[SystemContentBlock]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream conversation with the Nova model.

        This is the core streaming method that handles both streaming and non-streaming
        responses from the Nova API, properly formatting requests and responses according
        to Strands protocols.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.
            tool_choice: Selection strategy for tool invocation.
            system_prompt_content: Structured system content blocks (overrides system_prompt).
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Formatted StreamEvent objects conforming to Strands protocol.

        Raises:
            ContextWindowOverflowException: If the input exceeds the model's context window.
            ModelThrottledException: If the request is throttled by Nova (rate limits).
            NovaValidationException: If the request contains invalid parameters.
            NovaModelNotFoundException: If the specified model doesn't exist.
            NovaModelException: If the model encounters an internal error.
            NovaAPIException: For other API errors.

        Example:
            >>> async for event in model.stream(
            ...     messages=[{"role": "user", "content": [{"text": "Hi"}]}],
            ...     system_prompt="You are helpful"
            ... ):
            ...     print(event)
        """
        logger.debug(
            "messages_count=<%s> tool_specs_count=<%s> system_prompt_len=<%s> | formatting request",
            len(messages),
            len(tool_specs) if tool_specs else 0,
            len(system_prompt) if system_prompt else 0,
        )

        request = self.format_request(
            messages,
            tool_specs,
            system_prompt,
            tool_choice,
            system_prompt_content=system_prompt_content,
        )

        logger.debug("formatted request | invoking Nova model")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    json=request,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                ) as response:
                    if response.status_code != 200:
                        await response.aread()
                        self._handle_api_error(response)

                    logger.debug("received response from Nova API")

                    # Handle non-streaming response
                    if not self._stream:
                        response_data = await response.aread()
                        response_json = json.loads(response_data)
                        yield self.format_chunk({"chunk_type": "message_start"})

                        if response_json.get("choices"):
                            choice = response_json["choices"][0]
                            message = choice.get("message", {})

                            # Handle text content
                            if message.get("content"):
                                yield self.format_chunk(
                                    {"chunk_type": "content_start", "data_type": "text"}
                                )
                                yield self.format_chunk(
                                    {
                                        "chunk_type": "content_delta",
                                        "data_type": "text",
                                        "data": message["content"],
                                    }
                                )
                                yield self.format_chunk(
                                    {"chunk_type": "content_stop", "data_type": "text"}
                                )

                            # Handle tool calls
                            if message.get("tool_calls"):
                                for tool_call in message["tool_calls"]:
                                    tool_call_data = {
                                        "id": tool_call.get("id", ""),
                                        "type": tool_call.get("type", "function"),
                                        "name": tool_call.get("function", {}).get(
                                            "name", ""
                                        ),
                                        "arguments": tool_call.get("function", {}).get(
                                            "arguments", ""
                                        ),
                                    }
                                    yield self.format_chunk(
                                        {
                                            "chunk_type": "content_start",
                                            "data_type": "tool",
                                            "data": tool_call_data,
                                        }
                                    )
                                    yield self.format_chunk(
                                        {
                                            "chunk_type": "content_delta",
                                            "data_type": "tool",
                                            "data": tool_call_data,
                                        }
                                    )
                                    yield self.format_chunk(
                                        {
                                            "chunk_type": "content_stop",
                                            "data_type": "tool",
                                        }
                                    )

                            # Handle finish reason
                            finish_reason = choice.get("finish_reason", "stop")
                            if finish_reason == "tool_calls":
                                yield self.format_chunk(
                                    {"chunk_type": "message_stop", "data": "tool_calls"}
                                )
                            elif finish_reason == "length":
                                yield self.format_chunk(
                                    {"chunk_type": "message_stop", "data": "length"}
                                )
                            else:
                                yield self.format_chunk(
                                    {"chunk_type": "message_stop", "data": "stop"}
                                )

                        # Handle usage metadata if present
                        if response_json.get("usage"):
                            yield self.format_chunk(
                                {
                                    "chunk_type": "metadata",
                                    "data": response_json["usage"],
                                }
                            )

                        logger.debug("completed non-streaming response from Nova model")
                        return

                    # Handle streaming response
                    yield self.format_chunk({"chunk_type": "message_start"})

                    tool_calls: dict[int, dict[str, Any]] = {}
                    data_type = None
                    finish_reason = None

                    async for event in self._parse_sse_stream(response):
                        # Check for usage data (appears in streaming if include_usage is true)
                        if event.get("usage"):
                            yield self.format_chunk(
                                {"chunk_type": "metadata", "data": event["usage"]}
                            )

                        if not event.get("choices"):
                            continue

                        choice = event["choices"][0]
                        delta = choice.get("delta", {})

                        # Handle reasoning content
                        if delta.get("reasoning_content"):
                            chunks, data_type = self._stream_switch_content(
                                "reasoning_content", data_type
                            )
                            for chunk in chunks:
                                yield chunk
                            yield self.format_chunk(
                                {
                                    "chunk_type": "content_delta",
                                    "data_type": data_type,
                                    "data": delta["reasoning_content"],
                                }
                            )

                        # Handle regular content
                        if delta.get("content"):
                            chunks, data_type = self._stream_switch_content(
                                "text", data_type
                            )
                            for chunk in chunks:
                                yield chunk
                            yield self.format_chunk(
                                {
                                    "chunk_type": "content_delta",
                                    "data_type": data_type,
                                    "data": delta["content"],
                                }
                            )

                        # Handle tool calls
                        if delta.get("tool_calls"):
                            for tool_call in delta["tool_calls"]:
                                index = tool_call.get("index", 0)
                                if index not in tool_calls:
                                    tool_calls[index] = {
                                        "id": tool_call.get("id", ""),
                                        "type": tool_call.get("type", "function"),
                                        "name": tool_call.get("function", {}).get(
                                            "name", ""
                                        ),
                                        "arguments": "",
                                    }

                                # Accumulate arguments
                                if (
                                    "function" in tool_call
                                    and "arguments" in tool_call["function"]
                                ):
                                    tool_calls[index]["arguments"] += tool_call[
                                        "function"
                                    ]["arguments"]

                        # Handle finish reason
                        if choice.get("finish_reason") and not finish_reason:
                            finish_reason = choice["finish_reason"]
                            if data_type:
                                yield self.format_chunk(
                                    {
                                        "chunk_type": "content_stop",
                                        "data_type": data_type,
                                    }
                                )

                    # Emit tool calls after stream completes
                    for tool_call_data in tool_calls.values():
                        yield self.format_chunk(
                            {
                                "chunk_type": "content_start",
                                "data_type": "tool",
                                "data": tool_call_data,
                            }
                        )
                        yield self.format_chunk(
                            {
                                "chunk_type": "content_delta",
                                "data_type": "tool",
                                "data": tool_call_data,
                            }
                        )
                        yield self.format_chunk(
                            {"chunk_type": "content_stop", "data_type": "tool"}
                        )

                    yield self.format_chunk(
                        {"chunk_type": "message_stop", "data": finish_reason or "stop"}
                    )

                    logger.debug("completed streaming response from Nova model")

            except httpx.TimeoutException as _:
                logger.error(
                    "request_timeout=<%s> | Nova API request timed out", self.timeout
                )
                raise
            except httpx.RequestError as e:
                logger.error("request_error=<%s> | Nova API request failed", str(e))
                raise

    def _stream_switch_content(
        self, data_type: str, prev_data_type: str | None
    ) -> tuple[list[StreamEvent], str]:
        """Handle switching between different content types in a stream.

        When streaming responses contain multiple content types (e.g., text followed
        by reasoning content), this method ensures proper event boundaries by emitting
        stop events for the previous content and start events for the new content.

        Args:
            data_type: The content type being switched to.
            prev_data_type: The previous content type (None if this is the first content).

        Returns:
            Tuple containing:
                - List of StreamEvents to emit for the transition
                - The new data_type value

        Example:
            Switching from text to reasoning content will emit:
            1. contentBlockStop for text
            2. contentBlockStart for reasoning_content
        """
        chunks = []
        if data_type != prev_data_type:
            if prev_data_type is not None:
                chunks.append(
                    self.format_chunk(
                        {"chunk_type": "content_stop", "data_type": prev_data_type}
                    )
                )
            chunks.append(
                self.format_chunk(
                    {"chunk_type": "content_start", "data_type": data_type}
                )
            )

        return chunks, data_type

    @override
    async def structured_output(
        self,
        output_model: Type[T],
        prompt: Messages,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, Union[T, Any]], None]:
        """Get structured output from the Nova model using tool calling.

        This method leverages Nova's tool calling capabilities to extract structured
        data conforming to a Pydantic model schema. The model is forced to use the
        tool, ensuring the output matches the expected structure.

        Args:
            output_model: Pydantic model class defining the output structure.
            prompt: The prompt messages to send to the model.
            system_prompt: System prompt to provide context to the model.
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Dictionary with "output" key containing validated Pydantic model instance.

        Raises:
            ContextWindowOverflowException: If the input exceeds the model's context window.
            ModelThrottledException: If the request is throttled by Nova (rate limits).
            NovaValidationException: If the request contains invalid parameters.
            NovaModelNotFoundException: If the specified model doesn't exist.
            NovaModelException: If the model encounters an internal error.
            ValueError: If the response doesn't contain valid tool calls or parsing fails.

        Example:
            >>> from pydantic import BaseModel
            >>>
            >>> class Person(BaseModel):
            ...     name: str
            ...     age: int
            >>>
            >>> async for event in model.structured_output(
            ...     Person,
            ...     [{"role": "user", "content": [{"text": "John is 30"}]}]
            ... ):
            ...     if "output" in event:
            ...         person = event["output"]
            ...         print(f"{person.name}, {person.age}")
        """
        logger.debug(
            "using tool calling for structured output with model=<%s>",
            output_model.__name__,
        )

        tool_spec = convert_pydantic_to_tool_spec(output_model)
        request = self.format_request(
            prompt, [tool_spec], system_prompt, cast(ToolChoice, {"any": {}})
        )

        # Force non-streaming for structured output
        request["stream"] = False

        logger.debug("invoking Nova model for structured output")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=request,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                )

                if response.status_code != 200:
                    self._handle_api_error(response)

            except httpx.TimeoutException as _:
                logger.error(
                    "request_timeout=<%s> | Nova API request timed out", self.timeout
                )
                raise
            except httpx.RequestError as e:
                logger.error("request_error=<%s> | Nova API request failed", str(e))
                raise

            logger.debug("received structured output response from Nova model")

            # Parse response
            response_data = response.json()

            if not response_data.get("choices"):
                raise ValueError("No choices found in the response")

            if len(response_data["choices"]) > 1:
                logger.warning("Multiple choices found, using first choice only")

            choice = response_data["choices"][0]

            if choice.get("finish_reason") != "tool_calls":
                raise ValueError(
                    f"Expected tool_calls finish reason, got: {choice.get('finish_reason')}"
                )

            message = choice.get("message", {})
            tool_calls = message.get("tool_calls", [])

            if not tool_calls:
                raise ValueError("No tool_calls found in message")

            try:
                # Parse the tool call content as JSON
                tool_call = tool_calls[0]
                tool_call_data = json.loads(tool_call["function"]["arguments"])

                # Instantiate and validate the output model
                structured_output = output_model(**tool_call_data)

                logger.debug("successfully parsed structured output")
                yield {"output": structured_output}

            except (json.JSONDecodeError, TypeError, ValueError) as e:
                logger.error("failed to parse tool call arguments: %s", str(e))
                raise ValueError(
                    f"Failed to parse or validate structured output: {e}"
                ) from e

        logger.debug("completed structured output from Nova model")
