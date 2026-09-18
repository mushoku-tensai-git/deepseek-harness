"""Base plugin interface for DeepSeek Harness plugin architecture."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator


@dataclass
class PluginMetadata:
    """Metadata describing a plugin."""
    name: str
    version: str
    description: str
    author: str | None = None


@dataclass
class ToolCall:
    """Represents a tool call during agent execution."""
    tool_name: str
    tool_input: dict[str, Any]
    tool_output: Any | None = None
    success: bool = True


@dataclass
class GenerationChunk:
    """A chunk of generated content."""
    content: str
    finish_reason: str | None = None
    usage: dict[str, int] | None = None


@dataclass
class GenerationResponse:
    """Complete response from a generation request."""
    content: str
    finish_reason: str | None = None
    usage: dict[str, int] | None = None
    tool_calls: list[ToolCall] | None = None


class BasePlugin(ABC):
    """Abstract base class for all DeepSeek Harness plugins."""

    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the plugin. Called once when plugin is loaded."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the plugin. Called once when plugin is unloaded."""
        pass

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the plugin is available and ready to use."""
        pass


class BaseProviderPlugin(BasePlugin):
    """Abstract base class for LLM provider plugins."""

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> GenerationResponse:
        """Generate a completion synchronously."""
        pass

    @abstractmethod
    async def generate_stream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        """Generate a completion as a stream of chunks."""
        pass

    @abstractmethod
    async def execute_tool(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        """Execute a tool and return the result."""
        pass
