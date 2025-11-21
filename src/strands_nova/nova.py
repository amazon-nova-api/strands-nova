import base64
import json
import logging
import mimetypes
import os
from typing import Any, AsyncGenerator, Optional, Type, TypedDict, TypeVar, Union, cast

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


class NovaModel(Model):
    """Nova model provider implementation using httpx for direct API access."""

    base_url: str
    api_key: str
    timeout: float

    class NovaModelParams(TypedDict, total=False):
        """Nova API model parameters.

        Attributes:
            max_tokens: Maximum number of tokens to generate (deprecated, use max_completion_tokens).
            max_completion_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (0.0 to 1.0). Controls randomness in generation.
            top_p: Nucleus sampling parameter (0.0 to 1.0). Controls diversity.
            reasoning_effort: Reasoning effort level for reasoning models. Must be one of: "low", "medium", "high".
            metadata: Additional metadata to include with the request.
            web_search_options: Web search configuration (currently in review).
        """

        max_tokens: int
        max_completion_tokens: int
        temperature: float
        top_p: float
        reasoning_effort: str  # "low" | "medium" | "high"
        metadata: dict[str, Any]
        web_search_options: dict[str, Any]

    class NovaConfig(TypedDict, total=False):
        """Configuration options for Nova models.

        Attributes:
            model_id: Model ID (e.g., "nova-pro-v1", "nova-lite-v2").
                For a complete list of supported models, see https://api.nova.amazon.com/v1/models.
            params: Model parameters (e.g., max_tokens, temperature, reasoning_effort).
                For a complete list of supported parameters, see Nova API documentation.
                Any additional parameters not explicitly typed will be passed through to the API.
        """

        model_id: str
        params: Optional[dict[str, Any]]

    def __init__(
        self,
        model_id: str,
        api_key: Optional[str] = None,
        base_url: str = "https://api.nova.amazon.com/v1",
        timeout: float = 300.0,
        params: Optional[dict[str, Any]] = None,
        stream: bool = True,
        stream_options: Optional[dict[str, Any]] = None,
        **extra_config: Any,
    ) -> None:
        """Initialize Nova provider instance.

        Args:
            model_id: Model ID (e.g., "nova-pro-v1", "nova-lite-v2") (required).
            api_key: Nova API key for authentication. If not provided, will be inferred from NOVA_API_KEY environment variable.
            base_url: Base URL for Nova API (default: https://api.nova.amazon.com/v1).
            timeout: Request timeout in seconds (default: 300.0).
            params: Model parameters (max_tokens, temperature, etc.).
            _stream: Whether to stream responses (default: True).
            stream_options: Stream options like include_usage (default: {"include_usage": True}).
            **extra_config: Additional configuration options for future extensibility.

        Raises:
            ValueError: If api_key is not provided and NOVA_API_KEY environment variable is not set.
        """
        # Infer API key from environment variable if not provided
        if api_key is None:
            api_key = os.getenv("NOVA_API_KEY")
            if api_key is None:
                raise ValueError(
                    "api_key must be provided either as an argument or through the NOVA_API_KEY environment variable"
                )

        self.config: dict[str, Any] = {
            "model_id": model_id,
            "params": params or {},
        }
        # Add any extra config for extensibility
        self.config.update(extra_config)

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._stream = stream
        self.stream_options = (
            stream_options if stream_options is not None else {"include_usage": True}
        )

        logger.debug("config=<%s> | initializing Nova model", self.config)

    @override
    def update_config(self, **model_config: Unpack[NovaConfig]) -> None:  # type: ignore[override]
        """Update the Nova model configuration with the provided arguments.

        Args:
            **model_config: Configuration overrides.
        """
        self.config.update(model_config)

    @override
    def get_config(self) -> NovaConfig:
        """Get the Nova model configuration.

        Returns:
            The Nova model configuration.
        """
        return cast(NovaModel.NovaConfig, self.config)

    @classmethod
    def format_request_message_content(
        cls, content: ContentBlock, **kwargs: Any
    ) -> dict[str, Any]:
        """Format a Nova compatible content block.

        Args:
            content: Message content.
            **kwargs: Additional keyword arguments for future extensibility.

        Returns:
            Nova compatible content block.

        Raises:
            TypeError: If the content block type cannot be converted to a Nova-compatible format.
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
            audio_source = content["audio"]["source"]["bytes"]
            if isinstance(audio_source, str):
                # Already base64 encoded
                audio_data = audio_source
            else:
                # Raw bytes, need to encode
                audio_data = base64.b64encode(audio_source).decode("utf-8")
            
            audio_format = content["audio"]["format"]

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

    @classmethod
    def format_request_message_tool_call(
        cls, tool_use: ToolUse, **kwargs: Any
    ) -> dict[str, Any]:
        """Format a Nova compatible tool call.

        Args:
            tool_use: Tool use requested by the model.
            **kwargs: Additional keyword arguments for future extensibility.

        Returns:
            Nova compatible tool call.
        """
        return {
            "function": {
                "arguments": json.dumps(tool_use["input"]),
                "name": tool_use["name"],
            },
            "id": tool_use["toolUseId"],
            "type": "function",
        }

    @classmethod
    def format_request_tool_message(
        cls, tool_result: ToolResult, **kwargs: Any
    ) -> dict[str, Any]:
        """Format a Nova compatible tool message.

        Args:
            tool_result: Tool result collected from a tool execution.
            **kwargs: Additional keyword arguments for future extensibility.

        Returns:
            Nova compatible tool message.
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
                cls.format_request_message_content(content) for content in contents
            ],
        }

    @classmethod
    def _format_request_tool_choice(
        cls, tool_choice: ToolChoice | None
    ) -> dict[str, Any]:
        """Format a tool choice for Nova compatibility.

        Args:
            tool_choice: Tool choice configuration in Bedrock format.

        Returns:
            Nova compatible tool choice format.
        """
        if not tool_choice:
            return {}

        match tool_choice:
            case {"auto": _}:
                return {"tool_choice": "auto"}
            case {"any": _}:
                return {"tool_choice": "required"}
            case {"tool": {"name": tool_name}}:
                return {
                    "tool_choice": {"type": "function", "function": {"name": tool_name}}
                }
            case _:
                return {"tool_choice": "auto"}

    @classmethod
    def _format_system_messages(
        cls,
        system_prompt: Optional[str] = None,
        *,
        system_prompt_content: Optional[list[SystemContentBlock]] = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Format system messages for Nova.

        Args:
            system_prompt: System prompt to provide context to the model.
            system_prompt_content: System prompt content blocks to provide context to the model.
            **kwargs: Additional keyword arguments for future extensibility.

        Returns:
            List of formatted system messages.
        """
        if system_prompt and system_prompt_content is None:
            system_prompt_content = [{"text": system_prompt}]

        return [
            {"role": "system", "content": content["text"]}
            for content in system_prompt_content or []
            if "text" in content
        ]

    @classmethod
    def _format_regular_messages(
        cls, messages: Messages, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Format regular messages for Nova.

        Args:
            messages: List of message objects to be processed by the model.
            **kwargs: Additional keyword arguments for future extensibility.

        Returns:
            List of formatted messages.
        """
        formatted_messages = []

        for message in messages:
            contents = message["content"]

            if any("reasoningContent" in content for content in contents):
                logger.warning(
                    "reasoningContent is not supported in multi-turn conversations with the Chat Completions API."
                )

            formatted_contents = [
                cls.format_request_message_content(content)
                for content in contents
                if not any(
                    block_type in content
                    for block_type in ["toolResult", "toolUse", "reasoningContent"]
                )
            ]
            formatted_tool_calls = [
                cls.format_request_message_tool_call(content["toolUse"])
                for content in contents
                if "toolUse" in content
            ]
            formatted_tool_messages = [
                cls.format_request_tool_message(content["toolResult"])
                for content in contents
                if "toolResult" in content
            ]

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

    @classmethod
    def format_request_messages(
        cls,
        messages: Messages,
        system_prompt: Optional[str] = None,
        *,
        system_prompt_content: Optional[list[SystemContentBlock]] = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Format a Nova compatible messages array.

        Args:
            messages: List of message objects to be processed by the model.
            system_prompt: System prompt to provide context to the model.
            system_prompt_content: System prompt content blocks to provide context to the model.
            **kwargs: Additional keyword arguments for future extensibility.

        Returns:
            A Nova compatible messages array.
        """
        formatted_messages = cls._format_system_messages(
            system_prompt, system_prompt_content=system_prompt_content
        )
        formatted_messages.extend(cls._format_regular_messages(messages))

        return [
            message
            for message in formatted_messages
            if message["content"] or "tool_calls" in message
        ]

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
        """Format a Nova compatible chat request.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.
            tool_choice: Selection strategy for tool invocation.
            system_prompt_content: System prompt content blocks to provide context to the model.
            **kwargs: Additional keyword arguments for future extensibility.

        Returns:
            A Nova compatible chat request.

        Raises:
            TypeError: If a message contains a content block type that cannot be converted to a Nova-compatible
                format.
        """
        request = {
            "messages": self.format_request_messages(
                messages, system_prompt, system_prompt_content=system_prompt_content
            ),
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
            **(self._format_request_tool_choice(tool_choice)),
            **cast(dict[str, Any], self.config.get("params", {})),
        }

        # Add stream_options if configured
        if self.stream_options:
            request["stream_options"] = self.stream_options

        return request

    def format_chunk(self, event: dict[str, Any], **kwargs: Any) -> StreamEvent:
        """Format a Nova response event into a standardized message chunk.

        Args:
            event: A response event from the Nova model.
            **kwargs: Additional keyword arguments for future extensibility.

        Returns:
            The formatted chunk.

        Raises:
            RuntimeError: If chunk_type is not recognized.
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
                            "latencyMs": 0,
                        },
                    },
                }

            case _:
                raise RuntimeError(f"chunk_type=<{event['chunk_type']}> | unknown type")

    def _handle_api_error(self, response: httpx.Response) -> None:
        """Handle Nova API errors and raise appropriate exceptions.

        Args:
            response: The HTTP response from Nova API.

        Raises:
            ContextWindowOverflowException: If the input exceeds the model's context window.
            ModelThrottledException: If the request is throttled (rate limits).
            Exception: For other API errors.
        """
        try:
            error_data = response.json()
            error_message = error_data.get("error", {}).get("message", str(error_data))
        except Exception:
            error_message = response.text

        if response.status_code == 400:
            # ValidationException - check if it's context length related
            if "context" in error_message.lower() or "length" in error_message.lower():
                logger.warning("Nova API threw context window overflow error")
                raise ContextWindowOverflowException(error_message)
            raise Exception(f"Nova API validation error: {error_message}")
        elif response.status_code == 404:
            # ModelNotFoundException
            raise Exception(f"Nova API model not found: {error_message}")
        elif response.status_code == 429:
            # ThrottlingException
            logger.warning("Nova API threw rate limit error")
            raise ModelThrottledException(error_message)
        elif response.status_code == 500:
            # ModelException
            raise Exception(f"Nova API model error: {error_message}")
        else:
            raise Exception(f"Nova API error ({response.status_code}): {error_message}")

    async def _parse_sse_stream(
        self, response: httpx.Response
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Parse Server-Sent Events (SSE) stream from Nova API.

        Args:
            response: The streaming HTTP response.

        Yields:
            Parsed JSON events from the SSE stream.
        """
        async for line in response.aiter_lines():
            line = line.strip()
            if not line:
                continue

            if line.startswith("data: "):
                data = line[6:]  # Remove "data: " prefix
                if data == "[DONE]":
                    break

                try:
                    yield json.loads(data)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse SSE data: {data}, error: {e}")
                    continue

    @override
    async def stream(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        *,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream conversation with the Nova model.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.
            tool_choice: Selection strategy for tool invocation.
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Formatted message chunks from the model.

        Raises:
            ContextWindowOverflowException: If the input exceeds the model's context window.
            ModelThrottledException: If the request is throttled by Nova (rate limits).
        """
        logger.debug("formatting request")
        request = self.format_request(messages, tool_specs, system_prompt, tool_choice)
        logger.debug("formatted request=<%s>", request)

        logger.debug("invoking Nova model")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                # Use stream() context manager for actual streaming
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
                        self._handle_api_error(response)

                    logger.debug("got response from Nova model")

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

                        logger.debug("finished non-streaming response from Nova model")
                        return

                    # Handle streaming response
                    yield self.format_chunk({"chunk_type": "message_start"})

                    tool_calls: dict[int, dict[str, Any]] = {}
                    data_type = None
                    finish_reason = None

                    usage_data = None
                    async for event in self._parse_sse_stream(response):
                        # Check for usage data (appears in streaming if include_usage is true)
                        if event.get("usage"):
                            usage_data = event["usage"]
                            yield self.format_chunk(
                                {"chunk_type": "metadata", "data": usage_data}
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

                        # Handle finish reason - don't break, continue to capture usage data
                        if choice.get("finish_reason") and not finish_reason:
                            finish_reason = choice["finish_reason"]
                            if data_type:
                                yield self.format_chunk(
                                    {
                                        "chunk_type": "content_stop",
                                        "data_type": data_type,
                                    }
                                )

                    # Emit tool calls
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

                    logger.debug("finished streaming response from Nova model")

            except httpx.TimeoutException as e:
                logger.error(f"Nova API request timed out: {e}")
                raise
            except httpx.RequestError as e:
                logger.error(f"Nova API request error: {e}")
                raise

    def _stream_switch_content(
        self, data_type: str, prev_data_type: str | None
    ) -> tuple[list[StreamEvent], str]:
        """Handle switching to a new content stream.

        Args:
            data_type: The next content data type.
            prev_data_type: The previous content data type.

        Returns:
            Tuple containing:
            - Stop block for previous content and the start block for the next content.
            - Next content data type.
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

        Args:
            output_model: The output model to use for the agent.
            prompt: The prompt messages to use for the agent.
            system_prompt: System prompt to provide context to the model.
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Model events with the last being the structured output.

        Raises:
            ContextWindowOverflowException: If the input exceeds the model's context window.
            ModelThrottledException: If the request is throttled by Nova (rate limits).
            ValueError: If the response doesn't contain valid tool calls or parsing fails.
        """
        logger.debug("using tool calling for structured output")
        tool_spec = convert_pydantic_to_tool_spec(output_model)
        request = self.format_request(
            prompt, [tool_spec], system_prompt, cast(ToolChoice, {"any": {}})
        )

        # Structured output must be non-streaming, override stream settings
        request["stream"] = False
        request.pop("stream_options", None)

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

            except httpx.TimeoutException as e:
                logger.error(f"Nova API request timed out: {e}")
                raise
            except httpx.RequestError as e:
                logger.error(f"Nova API request error: {e}")
                raise

            logger.debug("got structured output response from Nova model")

            # Parse response
            response_data = response.json()

            if not response_data.get("choices"):
                raise ValueError("No choices found in the response")

            if len(response_data["choices"]) > 1:
                raise ValueError("Multiple choices found in the response")

            choice = response_data["choices"][0]

            if choice.get("finish_reason") != "tool_calls":
                raise ValueError("No tool_calls found in response")

            message = choice.get("message", {})
            tool_calls = message.get("tool_calls", [])

            if not tool_calls:
                raise ValueError("No tool_calls found in message")

            try:
                # Parse the tool call content as JSON
                tool_call = tool_calls[0]
                tool_call_data = json.loads(tool_call["function"]["arguments"])
                # Instantiate the output model with the parsed data
                structured_output = output_model(**tool_call_data)
                yield {"output": structured_output}
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                raise ValueError(
                    f"Failed to parse or load content into model: {e}"
                ) from e

        logger.debug("finished structured output from Nova model")
