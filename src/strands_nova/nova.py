"""Amazon Nova API model provider for Strands Agents SDK."""

import json
import logging
import os
from typing import Any, AsyncGenerator, AsyncIterable, Dict, List, Optional, Type, Union

import httpx
from pydantic import BaseModel
from strands.models.model import Model
from strands.types.content import ContentBlock, Messages, SystemContentBlock
from strands.types.exceptions import ContextWindowOverflowException, ModelThrottledException
from strands.types.streaming import (
    ContentBlockDelta,
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    MessageStartEvent,
    MessageStopEvent,
    MetadataEvent,
    StreamEvent,
)
from strands.types.tools import ToolChoice, ToolSpec

logger = logging.getLogger(__name__)


class NovaModelError(Exception):
    """Exception for Nova API errors."""

    pass


class NovaModel(Model):
    """Amazon Nova API model provider implementation.
    
    Nova API is an OpenAI-compatible API that provides access to Amazon's
    Nova family of models, including Nova Pro, Nova Premier, and specialized
    models for reasoning and image generation.
    """

    def __init__(
        self,
        model: str = "nova-premier-v1",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        top_p: float = 0.9,
        reasoning_effort: Optional[str] = None,
        web_search_options: Optional[Dict[str, Any]] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        """Initialize Nova model.

        Args:
            model: Model ID (e.g., "nova-premier-v1", "Nova Pro v3 (6.x)", 
                   "mumbai-flintflex-reasoning-v3")
            api_key: Nova API key (can be set via NOVA_API_KEY env var)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum number of tokens to generate
            top_p: Nucleus sampling parameter
            reasoning_effort: For reasoning models: "low", "medium", or "high".
                NOTE: Currently not supported by available models.
                TODO: Waiting on reasoning model name from AWS.
            web_search_options: Web search configuration (e.g., {"search_context_size": "low"})
            stop: List of stop sequences
            **kwargs: Additional parameters
        """
        self.model = model
        self.api_key = api_key or os.getenv("NOVA_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Nova API key is required. Set NOVA_API_KEY environment variable or pass api_key parameter."
            )

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.reasoning_effort = reasoning_effort
        self.web_search_options = web_search_options
        self.stop = stop or []
        self.base_url = "https://api.nova.amazon.com/v1/chat/completions"
        self.models_url = "https://api.nova.amazon.com/v1/models"

        # Store additional kwargs for future use
        self.additional_params = kwargs

    def update_config(self, **model_config: Any) -> None:
        """Update the model configuration with the provided arguments.

        Args:
            **model_config: Configuration overrides.
        """
        for key, value in model_config.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.additional_params[key] = value

    def get_config(self) -> Dict[str, Any]:
        """Return the model configuration.

        Returns:
            The model's configuration.
        """
        config = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stop": self.stop,
            **self.additional_params,
        }
        
        if self.reasoning_effort:
            config["reasoning_effort"] = self.reasoning_effort
        if self.web_search_options:
            config["web_search_options"] = self.web_search_options
            
        return config

    def _convert_messages_to_nova_format(
        self, messages: Union[Messages, str], system_prompt: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Convert Strands messages to Nova/OpenAI format.

        Args:
            messages: Messages to be processed by the model.
            system_prompt: Optional system message.

        Returns:
            List of messages in Nova/OpenAI format.
        """
        nova_messages = []

        # Add system prompt if provided
        if system_prompt:
            nova_messages.append({"role": "system", "content": system_prompt})

        # Handle both Messages type and simple string
        if isinstance(messages, str):
            nova_messages.append({"role": "user", "content": messages})
        elif hasattr(messages, "__iter__"):
            for msg in messages:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    role = msg["role"]
                    content_blocks = msg["content"]

                    # Convert content blocks to simple string format
                    if isinstance(content_blocks, list):
                        # Extract text from content blocks
                        text_parts = []
                        for block in content_blocks:
                            if isinstance(block, dict) and "text" in block:
                                text_parts.append(block["text"])
                            elif isinstance(block, str):
                                text_parts.append(block)
                        content = "\n".join(text_parts) if text_parts else ""
                    else:
                        content = str(content_blocks)

                    nova_messages.append({"role": role, "content": content})
                else:
                    # Fallback for other message formats
                    nova_messages.append(msg)

        return nova_messages

    def _convert_tool_specs_to_nova_format(
        self, tool_specs: Optional[List[ToolSpec]]
    ) -> Optional[List[Dict[str, Any]]]:
        """Convert Strands tool specs to Nova/OpenAI format.

        Strands SDK uses 'inputSchema' while OpenAI/Nova uses 'parameters'.

        Args:
            tool_specs: List of tool specifications.

        Returns:
            List of tools in Nova/OpenAI format, or None if no tools provided.
        """
        if not tool_specs:
            return None

        nova_tools = []
        for tool_spec in tool_specs:
            nova_tool = {
                "type": "function",
                "function": {
                    "name": tool_spec["name"],
                    "description": tool_spec["description"],
                    "parameters": tool_spec["inputSchema"],  # Convert inputSchema -> parameters
                },
            }
            nova_tools.append(nova_tool)
            logger.debug(f"Converted tool spec: {tool_spec['name']}")

        return nova_tools

    def _format_tool_choice(
        self, tool_choice: Optional[ToolChoice]
    ) -> Union[str, Dict[str, Any]]:
        """Convert Strands ToolChoice to Nova/OpenAI format.

        Strands formats:
        - {"auto": {}} -> "auto"
        - {"any": {}} -> "required"
        - {"tool": {"name": "foo"}} -> {"type": "function", "function": {"name": "foo"}}

        Args:
            tool_choice: Strands-format tool choice.

        Returns:
            Nova/OpenAI-format tool choice.
        """
        if tool_choice is None:
            return "auto"

        if "auto" in tool_choice:
            return "auto"

        if "any" in tool_choice:
            return "required"  # OpenAI/Nova equivalent of "any"

        if "tool" in tool_choice:
            tool_name = tool_choice["tool"]["name"]
            return {
                "type": "function",
                "function": {"name": tool_name}
            }

        logger.warning(f"Unknown tool_choice format: {tool_choice}, defaulting to 'auto'")
        return "auto"

    async def stream(
        self,
        messages: Union[Messages, str],
        tool_specs: Optional[List[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        *,
        tool_choice: Optional[ToolChoice] = None,
        system_prompt_content: Optional[List[SystemContentBlock]] = None,
        **kwargs: Any,
    ) -> AsyncIterable[StreamEvent]:
        """Stream responses from Nova model.

        Args:
            messages: Messages to be processed by the model.
            tool_specs: List of tool specifications for function calling.
            system_prompt: Optional system message (text).
            tool_choice: Optional tool choice strategy (auto/any/specific tool).
            system_prompt_content: Optional structured system prompt content blocks.
            **kwargs: Additional parameters (temperature, max_tokens, etc.).

        Yields:
            Formatted message chunks from the model.
        """
        # Build system messages from both sources
        system_messages = []

        # Add text system prompt
        if system_prompt:
            system_messages.append({"role": "system", "content": system_prompt})
            logger.debug("Added text system prompt")

        # Add structured system prompt content
        if system_prompt_content:
            for block in system_prompt_content:
                if "text" in block:
                    system_messages.append({
                        "role": "system",
                        "content": block["text"]
                    })
            logger.debug(f"Added {len(system_prompt_content)} system prompt content blocks")

        # Convert messages to Nova format
        nova_messages = self._convert_messages_to_nova_format(messages)

        # Prepend system messages
        nova_messages = system_messages + nova_messages

        # Build request body
        request_body: Dict[str, Any] = {
            "model": self.model,
            "messages": nova_messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "top_p": kwargs.get("top_p", self.top_p),
            "stream": True,
        }

        # Add stop sequences if provided
        if self.stop or kwargs.get("stop"):
            request_body["stop"] = kwargs.get("stop", self.stop)

        # Add tools if provided
        if tool_specs:
            nova_tools = self._convert_tool_specs_to_nova_format(tool_specs)
            if nova_tools:
                request_body["tools"] = nova_tools
                # Use proper tool_choice conversion
                request_body["tool_choice"] = self._format_tool_choice(
                    kwargs.get("tool_choice", tool_choice)
                )
                logger.debug(f"Added {len(nova_tools)} tools with choice: {request_body['tool_choice']}")

        # Add reasoning effort for reasoning models
        # TODO: Waiting on reasoning model name from AWS. Currently not supported by available models.
        # Will raise "invalid reasoning config" error if used with current models.
        if self.reasoning_effort or kwargs.get("reasoning_effort"):
            reasoning = kwargs.get("reasoning_effort", self.reasoning_effort)
            request_body["reasoning_effort"] = reasoning
            logger.debug(f"Added reasoning_effort: {reasoning} (may not be supported by current models)")

        # Add web search options if configured
        if self.web_search_options or kwargs.get("web_search_options"):
            request_body["web_search_options"] = kwargs.get(
                "web_search_options", self.web_search_options
            )

        # Add stream options to include usage info
        request_body["stream_options"] = {"include_usage": True}

        # Add any additional parameters
        for key, value in self.additional_params.items():
            if key not in request_body:
                request_body[key] = value

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
                async with client.stream(
                    "POST",
                    self.base_url,
                    json=request_body,
                    headers=headers,
                ) as response:
                    # Handle authentication errors (Nova returns 500 for invalid API keys)
                    if response.status_code in [401, 500]:
                        error_text = await response.aread()
                        error_body = error_text.decode('utf-8')

                        # Check if it's an auth error (HTML response with "Login Provider Error")
                        if "Login Provider Error" in error_body or response.status_code == 401:
                            logger.error("Authentication failed - invalid Nova API key")
                            raise NovaModelError(
                                "Authentication failed - check your Nova API key (NOVA_API_KEY environment variable)"
                            )

                        # Other 500 errors
                        logger.error(f"Nova API error (500): {error_body[:200]}")
                        raise NovaModelError(f"Nova API error: {error_body[:200]}")

                    # Handle model not found (404)
                    if response.status_code == 404:
                        error_text = await response.aread()
                        error_body = error_text.decode('utf-8')
                        logger.error(f"Model not found: {self.model}")
                        raise NovaModelError(
                            f"Model '{self.model}' not found or access denied. {error_body}"
                        )

                    # Handle rate limiting (429)
                    if response.status_code == 429:
                        error_text = await response.aread()
                        logger.warning("Nova API rate limit exceeded")
                        raise ModelThrottledException(
                            f"Nova API rate limit exceeded: {error_text.decode('utf-8')}"
                        )

                    # Handle bad requests (400)
                    if response.status_code == 400:
                        error_text = await response.aread()
                        error_body = error_text.decode('utf-8')

                        # Check if it's a context length error
                        # TODO: Need to verify exact error message format for context overflow
                        if any(keyword in error_body.lower() for keyword in ["context", "maximum", "token limit", "exceeds"]):
                            logger.error(f"Context window exceeded: {error_body}")
                            raise ContextWindowOverflowException(
                                f"Context window exceeded: {error_body}"
                            )

                        logger.error(f"Bad request: {error_body}")
                        raise NovaModelError(f"Bad request: {error_body}")

                    # Handle other errors
                    if response.status_code != 200:
                        error_text = await response.aread()
                        logger.error(f"Nova API request failed with status {response.status_code}")
                        raise NovaModelError(
                            f"Nova API request failed with status {response.status_code}: {error_text.decode('utf-8')}"
                        )

                    # Emit message start
                    logger.debug("Stream started, emitting messageStart")
                    message_start: MessageStartEvent = {"role": "assistant"}
                    yield {"messageStart": message_start}

                    # Track content block state
                    content_block_started = False
                    tool_call_started = False
                    accumulated_tool_calls: Dict[int, Dict[str, Any]] = {}

                    # Process SSE stream (OpenAI format)
                    buffer = b""
                    async for chunk in response.aiter_bytes():
                        buffer += chunk
                        # Split by double newline which separates SSE events
                        events = buffer.split(b"\n\n")
                        # Keep the last incomplete event in buffer
                        buffer = events[-1]

                        # Process complete events
                        for event_data in events[:-1]:
                            if not event_data:
                                continue

                            # Parse SSE event
                            lines = event_data.split(b"\n")
                            data_line = None
                            for line in lines:
                                if line.startswith(b"data:"):
                                    data_line = line[5:].strip()
                                    break

                            if not data_line or data_line == b"[DONE]":
                                continue

                            try:
                                data_str = data_line.decode("utf-8")
                                data = json.loads(data_str)

                                # Handle OpenAI-format streaming response
                                if "choices" in data and len(data["choices"]) > 0:
                                    choice = data["choices"][0]

                                    # Handle content delta
                                    if "delta" in choice and "content" in choice["delta"]:
                                        content = choice["delta"]["content"]
                                        if content:
                                            # Emit content block start on first content
                                            if not content_block_started:
                                                logger.debug("Emitting contentBlockStart")
                                                start_event: ContentBlockStartEvent = {"start": {}}
                                                yield {"contentBlockStart": start_event}
                                                content_block_started = True

                                            # Emit content delta
                                            delta: ContentBlockDelta = {"text": content}
                                            delta_event: ContentBlockDeltaEvent = {"delta": delta}
                                            stream_event: StreamEvent = {"contentBlockDelta": delta_event}
                                            yield stream_event

                                    # Handle tool calls
                                    if "delta" in choice and "tool_calls" in choice["delta"]:
                                        tool_calls = choice["delta"]["tool_calls"]

                                        for tool_call in tool_calls:
                                            index = tool_call.get("index", 0)

                                            # Initialize tool call accumulator
                                            if index not in accumulated_tool_calls:
                                                accumulated_tool_calls[index] = {
                                                    "id": None,
                                                    "name": None,
                                                    "arguments": ""
                                                }

                                            # First chunk of a tool call (has id and name)
                                            if tool_call.get("id"):
                                                accumulated_tool_calls[index]["id"] = tool_call["id"]

                                            if tool_call.get("function", {}).get("name"):
                                                accumulated_tool_calls[index]["name"] = tool_call["function"]["name"]

                                                # Emit content block start for tool
                                                logger.debug(f"Emitting contentBlockStart for tool: {tool_call['function']['name']}")
                                                tool_start_event: ContentBlockStartEvent = {
                                                    "start": {
                                                        "toolUse": {
                                                            "name": tool_call["function"]["name"],
                                                            "toolUseId": tool_call["id"],
                                                        }
                                                    }
                                                }
                                                yield {"contentBlockStart": tool_start_event}
                                                tool_call_started = True

                                            # Accumulate arguments
                                            if "function" in tool_call and "arguments" in tool_call["function"]:
                                                arguments = tool_call["function"]["arguments"]
                                                if arguments:
                                                    accumulated_tool_calls[index]["arguments"] += arguments

                                                    # Emit tool input delta
                                                    tool_delta: ContentBlockDelta = {
                                                        "toolUse": {
                                                            "input": arguments
                                                        }
                                                    }
                                                    tool_delta_event: ContentBlockDeltaEvent = {"delta": tool_delta}
                                                    yield {"contentBlockDelta": tool_delta_event}

                                    # Handle finish
                                    if "finish_reason" in choice and choice["finish_reason"]:
                                        finish_reason = choice["finish_reason"]

                                        # Emit content block stop if we started a content block
                                        if content_block_started or tool_call_started:
                                            logger.debug("Emitting contentBlockStop")
                                            stop_block_event: ContentBlockStopEvent = {}
                                            yield {"contentBlockStop": stop_block_event}
                                            content_block_started = False
                                            tool_call_started = False

                                        # Map OpenAI finish reasons to Strands format
                                        stop_reason = "end_turn"
                                        if finish_reason == "length":
                                            stop_reason = "max_tokens"
                                        elif finish_reason == "tool_calls":
                                            stop_reason = "tool_use"

                                        logger.debug(f"Emitting messageStop with reason: {stop_reason}")
                                        stop_event: MessageStopEvent = {"stopReason": stop_reason}
                                        final_event: StreamEvent = {"messageStop": stop_event}
                                        yield final_event

                                # Handle usage metadata
                                if "usage" in data:
                                    logger.debug(f"Emitting metadata with usage: {data['usage']}")
                                    metadata_event: MetadataEvent = {
                                        "usage": {
                                            "inputTokens": data["usage"]["prompt_tokens"],
                                            "outputTokens": data["usage"]["completion_tokens"],
                                            "totalTokens": data["usage"]["total_tokens"],
                                        }
                                    }
                                    yield {"metadata": metadata_event}

                            except (json.JSONDecodeError, KeyError, UnicodeDecodeError) as e:
                                # Log error but continue processing
                                logger.debug(f"Error parsing SSE event: {e}")
                                continue

        except httpx.TimeoutException as e:
            logger.error(f"Request timeout: {str(e)}")
            raise ModelThrottledException(f"Request timeout: {str(e)}")
        except httpx.NetworkError as e:
            logger.error(f"Network error: {str(e)}")
            raise NovaModelError(f"Network error: {str(e)}")
        except httpx.RequestError as e:
            logger.error(f"Request error: {str(e)}")
            raise NovaModelError(f"Request error: {str(e)}")

    async def structured_output(
        self,
        output_model: Type[BaseModel],
        prompt: Messages,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[Dict[str, Union[BaseModel, Any]], None]:
        """Get structured output from the model.

        Note: This is not yet implemented for Nova models.

        TODO: Investigate if Nova API supports structured output via response_format parameter.
        If supported, implement similar to OpenAI's structured outputs.

        Args:
            output_model: The output model to use for the agent.
            prompt: The prompt messages to use for the agent.
            system_prompt: System prompt to provide context to the model.
            **kwargs: Additional keyword arguments.

        Raises:
            NotImplementedError: Structured output is not yet supported for Nova models.
        """
        raise NotImplementedError(
            "Structured output is not yet supported for Nova models. "
            "TODO: Check if Nova API supports response_format parameter."
        )
        # Make this a generator (unreachable code, but satisfies type hint)
        yield  # pragma: no cover

    def __str__(self) -> str:
        """String representation of the model."""
        return f"NovaModel(model='{self.model}', temperature={self.temperature}, max_tokens={self.max_tokens})"
