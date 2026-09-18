"""DeepSeek Harness plugin architecture for LS Companion."""
from __future__ import annotations

from .base_plugin import (
    BasePlugin,
    BaseProviderPlugin,
    GenerationChunk,
    GenerationResponse,
    PluginMetadata,
    ToolCall,
)
from .security_plugin import (
    ActionType,
    create_security_plugin,
    Decision,
    PermissionDeniedException,
    PermissionRequest,
    RiskLevel,
    Rule,
    SecurityPlugin,
)

__all__ = [
    "BasePlugin",
    "BaseProviderPlugin",
    "GenerationChunk",
    "GenerationResponse",
    "PluginMetadata",
    "ToolCall",
    # Security plugin exports
    "SecurityPlugin",
    "create_security_plugin",
    "PermissionDeniedException",
    "PermissionRequest",
    "RiskLevel",
    "ActionType",
    "Decision",
    "Rule",
]
