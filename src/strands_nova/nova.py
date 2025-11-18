"""Amazon Nova API model provider for Strands Agents SDK."""

import json
import logging
import os
from typing import Any, AsyncGenerator, AsyncIterable, Dict, List, Optional, Type, Union

import httpx
from pydantic import BaseModel
from strands.models.model import Model
from strands.types.content import Messages
from strands.types.streaming import ContentBlockDelta, ContentBlockDeltaEvent, MessageStopEvent, StreamEvent
from strands.types.tools import ToolSpec

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
            reasoning_effort: For reasoning models: "low", "medium", or "high"
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
                    "name": tool_spec.get("name", ""),
                    "description": tool_spec.get("description", ""),
                    "parameters": tool_spec.get("parameters", {}),
                },
            }
            nova_tools.append(nova_tool)

        return nova_tools

    async def stream(
        self,
        messages: Union[Messages, str],
        tool_specs: Optional[List[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterable[StreamEvent]:
        """Stream responses from Nova model.

        Args:
            messages: Messages to be processed by the model.
            tool_specs: List of tool specifications for function calling.
            system_prompt: Optional system message.
            **kwargs: Additional parameters.

        Yields:
            Formatted message chunks from the model.
        """
        # Convert messages to Nova format
        nova_messages = self._convert_messages_to_nova_format(messages, system_prompt)

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
                request_body["tool_choice"] = kwargs.get("tool_choice", "auto")

        # Add reasoning effort for reasoning models
        if self.reasoning_effort or kwargs.get("reasoning_effort"):
            request_body["reasoning_effort"] = kwargs.get(
                "reasoning_effort", self.reasoning_effort
            )

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

        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            async with client.stream(
                "POST",
                self.base_url,
                json=request_body,
                headers=headers,
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    error_msg = f"Nova API request failed with status {response.status_code}: {error_text.decode('utf-8')}"
                    raise NovaModelError(error_msg)

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
                                        delta: ContentBlockDelta = {"text": content}
                                        delta_event: ContentBlockDeltaEvent = {"delta": delta}
                                        stream_event: StreamEvent = {"contentBlockDelta": delta_event}
                                        yield stream_event

                                # Handle tool calls
                                if "delta" in choice and "tool_calls" in choice["delta"]:
                                    # Tool call handling - may need to be enhanced based on Strands SDK requirements
                                    pass

                                # Check for finish reason
                                if "finish_reason" in choice and choice["finish_reason"]:
                                    finish_reason = choice["finish_reason"]
                                    # Map OpenAI finish reasons to Strands format
                                    stop_reason = "end_turn"
                                    if finish_reason == "length":
                                        stop_reason = "max_tokens"
                                    elif finish_reason == "tool_calls":
                                        stop_reason = "tool_use"

                                    stop_event: MessageStopEvent = {"stopReason": stop_reason}
                                    final_event: StreamEvent = {"messageStop": stop_event}
                                    yield final_event
                                    break

                        except (json.JSONDecodeError, KeyError, UnicodeDecodeError) as e:
                            # Log error but continue processing
                            logger.debug(f"Error parsing SSE event: {e}")
                            continue

    async def structured_output(
        self,
        output_model: Type[BaseModel],
        prompt: Messages,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[Dict[str, Union[BaseModel, Any]], None]:
        """Get structured output from the model.

        Note: This is not yet implemented for Nova models.

        Args:
            output_model: The output model to use for the agent.
            prompt: The prompt messages to use for the agent.
            system_prompt: System prompt to provide context to the model.
            **kwargs: Additional keyword arguments.

        Raises:
            NotImplementedError: Structured output is not yet supported for Nova models.
        """
        raise NotImplementedError("Structured output is not yet supported for Nova models")
        # Make this a generator (unreachable code, but satisfies type hint)
        yield  # pragma: no cover

    def __str__(self) -> str:
        """String representation of the model."""
        return f"NovaModel(model='{self.model}', temperature={self.temperature}, max_tokens={self.max_tokens})"
