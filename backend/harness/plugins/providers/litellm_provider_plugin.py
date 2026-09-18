"""LiteLLM provider plugin for DeepSeek Harness.

This plugin uses LiteLLM to enable hot-swapping between multiple LLM providers:
- Ollama
- OpenRouter
- OpenAI
- Anthropic
- Custom OpenAI-compatible endpoints

Configuration is read from:
- Windows: %APPDATA%/LadeStack/LSCompanion/config.json
- Linux: ~/.config/ladestack/lscompanion/config.json
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator

import litellm
from litellm import acompletion

from ..base_plugin import (
    BaseProviderPlugin,
    GenerationChunk,
    GenerationResponse,
    PluginMetadata,
    ToolCall,
)


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""
    model_name: str
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.7
    max_tokens: int | None = None
    provider_type: str | None = None  # e.g., "openai", "ollama", "anthropic"


def get_config_path() -> Path:
    """Get the platform-specific config file path."""
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA", "")
        if appdata:
            return Path(appdata) / "LadeStack" / "LSCompanion" / "config.json"
        # Fallback to user profile
        userprofile = os.environ.get("USERPROFILE", "")
        if userprofile:
            return Path(userprofile) / "AppData" / "Roaming" / "LadeStack" / "LSCompanion" / "config.json"
    # Linux and macOS
    xdg_config = os.environ.get("XDG_CONFIG_HOME", "")
    if xdg_config:
        return Path(xdg_config) / "ladestack" / "lscompanion" / "config.json"
    # Default to home directory
    home = Path.home()
    return home / ".config" / "ladestack" / "lscompanion" / "config.json"


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load configuration from JSON file."""
    path = config_path or get_config_path()
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def save_config(config: dict[str, Any], config_path: Path | None = None) -> None:
    """Save configuration to JSON file."""
    path = config_path or get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def get_provider_from_model(model_name: str) -> str:
    """Infer provider type from model name."""
    model_lower = model_name.lower()
    if model_lower.startswith("ollama/"):
        return "ollama"
    elif model_lower.startswith("openrouter/"):
        return "openrouter"
    elif model_lower.startswith("anthropic/"):
        return "anthropic"
    elif model_lower.startswith("openai/"):
        return "openai"
    elif "claude" in model_lower:
        return "anthropic"
    elif "gpt" in model_lower:
        return "openai"
    return "openai"  # Default to OpenAI-compatible


class LiteLLMProviderPlugin(BaseProviderPlugin):
    """LiteLLM-based provider plugin supporting multiple LLM backends."""

    def __init__(self, config: ProviderConfig | None = None):
        self._config = config
        self._initialized = False
        self._litellm_configured = False

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="litellm-provider",
            version="1.0.0",
            description="LiteLLM-powered multi-provider LLM plugin supporting Ollama, OpenRouter, OpenAI, Anthropic, and custom endpoints",
            author="LS Companion",
        )

    async def initialize(self) -> None:
        """Initialize the LiteLLM provider plugin."""
        if self._initialized:
            return
        
        # Load config if not provided
        if self._config is None:
            config_data = load_config()
            provider_config = config_data.get("provider", {})
            self._config = ProviderConfig(
                model_name=provider_config.get("model_name", "gpt-4o"),
                base_url=provider_config.get("base_url"),
                api_key=provider_config.get("api_key"),
                temperature=provider_config.get("temperature", 0.7),
                max_tokens=provider_config.get("max_tokens"),
                provider_type=provider_config.get("provider_type"),
            )

        # Configure LiteLLM
        if self._config.base_url:
            litellm.api_base = self._config.base_url
        
        if self._config.api_key:
            # Set appropriate key based on provider
            provider_type = self._config.provider_type or get_provider_from_model(self._config.model_name)
            if provider_type == "anthropic":
                litellm.anthropic_key = self._config.api_key
            elif provider_type == "ollama":
                pass  # Ollama typically doesn't need API key
            else:
                litellm.api_key = self._config.api_key

        self._litellm_configured = True
        self._initialized = True

    async def shutdown(self) -> None:
        """Shutdown the plugin."""
        self._initialized = False
        self._litellm_configured = False

    async def is_available(self) -> bool:
        """Check if the plugin is available."""
        return self._initialized and self._litellm_configured

    def _build_model_string(self) -> str:
        """Build the full model string for LiteLLM."""
        if not self._config:
            raise RuntimeError("Plugin not initialized with config")
        
        model = self._config.model_name
        provider_type = self._config.provider_type
        
        # If model already has provider prefix, use as-is
        if "/" in model:
            return model
        
        # Add provider prefix if known
        if provider_type:
            return f"{provider_type}/{model}"
        
        # Try to infer from model name
        inferred = get_provider_from_model(model)
        return f"{inferred}/{model}"

    async def generate(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> GenerationResponse:
        """Generate a completion synchronously."""
        if not self._initialized:
            await self.initialize()

        model_str = model or self._build_model_string()
        temp = temperature if temperature is not None else (self._config.temperature if self._config else 0.7)
        max_tok = max_tokens or (self._config.max_tokens if self._config else None)

        response = await acompletion(
            model=model_str,
            messages=messages,
            temperature=temp,
            max_tokens=max_tok,
            **kwargs,
        )

        content = ""
        finish_reason = None
        usage = None

        if response.choices:
            choice = response.choices[0]
            if hasattr(choice.message, "content") and choice.message.content:
                content = choice.message.content
            if hasattr(choice, "finish_reason") and choice.finish_reason:
                finish_reason = choice.finish_reason

        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                "total_tokens": getattr(response.usage, "total_tokens", 0),
            }

        tool_calls = None
        if response.choices and hasattr(response.choices[0].message, "tool_calls") and response.choices[0].message.tool_calls:
            tool_calls = [
                ToolCall(
                    tool_name=tc.function.name if hasattr(tc.function, "name") else "unknown",
                    tool_input=json.loads(tc.function.arguments) if hasattr(tc.function, "arguments") else {},
                )
                for tc in response.choices[0].message.tool_calls
            ]

        return GenerationResponse(
            content=content,
            finish_reason=finish_reason,
            usage=usage,
            tool_calls=tool_calls,
        )

    async def generate_stream(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        """Generate a completion as a stream of chunks."""
        if not self._initialized:
            await self.initialize()

        model_str = model or self._build_model_string()
        temp = temperature if temperature is not None else (self._config.temperature if self._config else 0.7)
        max_tok = max_tokens or (self._config.max_tokens if self._config else None)

        stream = await acompletion(
            model=model_str,
            messages=messages,
            temperature=temp,
            max_tokens=max_tok,
            stream=True,
            **kwargs,
        )

        async for chunk in stream:
            content = ""
            finish_reason = None
            usage = None

            if chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    content = delta.content
                if hasattr(chunk.choices[0], "finish_reason") and chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason

            if hasattr(chunk, "usage") and chunk.usage:
                usage = {
                    "prompt_tokens": getattr(chunk.usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(chunk.usage, "completion_tokens", 0),
                    "total_tokens": getattr(chunk.usage, "total_tokens", 0),
                }

            yield GenerationChunk(
                content=content,
                finish_reason=finish_reason,
                usage=usage,
            )

    async def execute_tool(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        """Execute a tool and return the result.
        
        This is a placeholder implementation. Actual tool execution
        should be delegated to the harness's tool registry.
        """
        # Placeholder - tools are executed by the harness core
        # This method exists for provider plugins that support native tool calling
        raise NotImplementedError(
            "Tool execution is handled by the harness core. "
            "Use the harness's tool registry for executing tools."
        )

    def update_config(self, config: ProviderConfig) -> None:
        """Update the provider configuration at runtime."""
        self._config = config
        self._litellm_configured = False
        # Re-initialize with new config
        # Note: This is called from sync context, so we don't await here
        # The next generate/generate_stream call will re-initialize
