"""DeepSeek Harness for LS Companion - A plugin-based AI agent framework."""
from __future__ import annotations

from .core import (
    HarnessEvent,
    HarnessRunner,
    RunContext,
    ToolExecution,
    ToolRegistry,
)
from .plugins import (
    BasePlugin,
    BaseProviderPlugin,
    GenerationChunk,
    GenerationResponse,
    PluginMetadata,
    ToolCall,
)

__version__ = "1.0.0"
__all__ = [
    # Core
    "HarnessEvent",
    "HarnessRunner",
    "RunContext",
    "ToolExecution",
    "ToolRegistry",
    # Plugins
    "BasePlugin",
    "BaseProviderPlugin",
    "GenerationChunk",
    "GenerationResponse",
    "PluginMetadata",
    "ToolCall",
]
