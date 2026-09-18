"""Core harness runner and lifecycle management."""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

from ..plugins.base_plugin import (
    BaseProviderPlugin,
    GenerationChunk,
    GenerationResponse,
    ToolCall,
)


@dataclass
class HarnessEvent:
    """An event emitted during harness execution."""
    event_type: str
    data: Any


@dataclass
class ToolExecution:
    """Represents a tool execution during agent run."""
    tool_name: str
    tool_input: dict[str, Any]
    tool_output: Any | None = None
    success: bool = True
    error: str | None = None


@dataclass
class RunContext:
    """Context for a single harness run."""
    session_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    events: list[HarnessEvent] = field(default_factory=list)
    tool_executions: list[ToolExecution] = field(default_factory=list)
    final_response: str = ""
    finish_reason: str | None = None


class ToolRegistry:
    """Registry for available tools."""

    def __init__(self):
        self._tools: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, func: Callable[..., Any]) -> None:
        """Register a tool function."""
        self._tools[name] = func

    def get(self, name: str) -> Callable[..., Any] | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())


class HarnessRunner:
    """Runs the DeepSeek Harness agent loop."""

    def __init__(self, provider: BaseProviderPlugin):
        self._provider = provider
        self._tool_registry = ToolRegistry()
        self._running = False
        self._max_iterations = 10  # Prevent infinite loops

    @property
    def provider(self) -> BaseProviderPlugin:
        """Get the current provider plugin."""
        return self._provider

    @provider.setter
    def provider(self, value: BaseProviderPlugin) -> None:
        """Set a new provider plugin."""
        self._provider = value

    def register_tool(self, name: str, func: Callable[..., Any]) -> None:
        """Register a tool with the harness."""
        self._tool_registry.register(name, func)

    async def execute_tool(self, tool_name: str, tool_input: dict[str, Any]) -> Any:
        """Execute a tool and return the result."""
        tool_func = self._tool_registry.get(tool_name)
        if tool_func is None:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        if asyncio.iscoroutinefunction(tool_func):
            return await tool_func(**tool_input)
        else:
            return tool_func(**tool_input)

    async def run(
        self,
        prompt: str,
        session_id: str | None = None,
        system_message: str | None = None,
    ) -> RunContext:
        """Run a complete agent turn."""
        session_id = session_id or f"session-{uuid.uuid4().hex}"
        ctx = RunContext(session_id=session_id)

        # Initialize messages
        if system_message:
            ctx.messages.append({"role": "system", "content": system_message})
        ctx.messages.append({"role": "user", "content": prompt})

        self._running = True
        iteration = 0

        try:
            while self._running and iteration < self._max_iterations:
                iteration += 1

                # Generate response
                response = await self._provider.generate(
                    messages=ctx.messages,
                )

                # Add assistant message to history
                ctx.messages.append({
                    "role": "assistant",
                    "content": response.content,
                })

                ctx.final_response = response.content
                ctx.finish_reason = response.finish_reason

                # Check for tool calls
                if response.tool_calls:
                    for tool_call in response.tool_calls:
                        ctx.events.append(HarnessEvent(
                            event_type="tool_start",
                            data={"tool": tool_call.tool_name, "input": tool_call.tool_input},
                        ))

                        try:
                            tool_output = await self.execute_tool(
                                tool_call.tool_name,
                                tool_call.tool_input,
                            )
                            tool_execution = ToolExecution(
                                tool_name=tool_call.tool_name,
                                tool_input=tool_call.tool_input,
                                tool_output=tool_output,
                                success=True,
                            )
                        except Exception as e:
                            tool_output = {"error": str(e)}
                            tool_execution = ToolExecution(
                                tool_name=tool_call.tool_name,
                                tool_input=tool_call.tool_input,
                                tool_output=None,
                                success=False,
                                error=str(e),
                            )

                        ctx.tool_executions.append(tool_execution)
                        ctx.events.append(HarnessEvent(
                            event_type="tool_end",
                            data={"tool": tool_call.tool_name, "output": tool_output},
                        ))

                        # Add tool result to messages
                        ctx.messages.append({
                            "role": "tool",
                            "tool_call_id": str(uuid.uuid4()),
                            "name": tool_call.tool_name,
                            "content": str(tool_output),
                        })

                    # Continue loop to process tool results
                    continue
                else:
                    # No tool calls, we're done
                    break

        finally:
            self._running = False

        return ctx

    async def run_stream(
        self,
        prompt: str,
        session_id: str | None = None,
        system_message: str | None = None,
    ) -> AsyncIterator[HarnessEvent]:
        """Run an agent turn and stream events."""
        session_id = session_id or f"session-{uuid.uuid4().hex}"
        messages: list[dict[str, Any]] = []

        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        self._running = True
        iteration = 0

        try:
            while self._running and iteration < self._max_iterations:
                iteration += 1
                accumulated_content = ""

                # Stream generation
                async for chunk in self._provider.generate_stream(messages=messages):
                    if chunk.content:
                        accumulated_content += chunk.content
                        yield HarnessEvent(
                            event_type="token",
                            data=chunk.content,
                        )

                # Add assistant message
                if accumulated_content:
                    messages.append({"role": "assistant", "content": accumulated_content})

                # For now, tool calls are handled in non-streaming mode
                # In a full implementation, you'd parse tool calls from the stream
                yield HarnessEvent(
                    event_type="complete",
                    data={"response": accumulated_content},
                )
                break

        finally:
            self._running = False

    async def shutdown(self) -> None:
        """Shutdown the harness runner."""
        self._running = False
        await self._provider.shutdown()
