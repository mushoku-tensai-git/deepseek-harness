"""JSON configuration manager for LS Companion."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ProviderSettings:
    """Settings for the LLM provider."""
    model_name: str = "gpt-4o"
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.7
    max_tokens: int | None = None
    provider_type: str | None = None


@dataclass
class DaemonSettings:
    """Settings for the daemon server."""
    host: str = "127.0.0.1"
    port: int = 8765
    log_level: str = "info"


@dataclass
class Config:
    """Complete configuration for LS Companion."""
    provider: ProviderSettings = field(default_factory=ProviderSettings)
    daemon: DaemonSettings = field(default_factory=DaemonSettings)
    extra: dict[str, Any] = field(default_factory=dict)


def get_config_path() -> Path:
    """Get the platform-specific config file path.
    
    Returns:
        - Windows: %APPDATA%/LadeStack/LSCompanion/config.json
        - Linux: ~/.config/ladestack/lscompanion/config.json
        - macOS: ~/Library/Application Support/LadeStack/LSCompanion/config.json
    """
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA", "")
        if appdata:
            return Path(appdata) / "LadeStack" / "LSCompanion" / "config.json"
        userprofile = os.environ.get("USERPROFILE", "")
        if userprofile:
            return Path(userprofile) / "AppData" / "Roaming" / "LadeStack" / "LSCompanion" / "config.json"
    elif sys.platform == "darwin":
        home = Path.home()
        return home / "Library" / "Application Support" / "LadeStack" / "LSCompanion" / "config.json"
    else:
        # Linux and other Unix-like systems
        xdg_config = os.environ.get("XDG_CONFIG_HOME", "")
        if xdg_config:
            return Path(xdg_config) / "ladestack" / "lscompanion" / "config.json"
        home = Path.home()
        return home / ".config" / "ladestack" / "lscompanion" / "config.json"


class ConfigManager:
    """Manages loading, saving, and accessing configuration."""

    def __init__(self, config_path: Path | None = None):
        self._config_path = config_path or get_config_path()
        self._config: Config | None = None

    @property
    def config_path(self) -> Path:
        """Get the config file path."""
        return self._config_path

    def load(self) -> Config:
        """Load configuration from file."""
        if not self._config_path.exists():
            # Return defaults if no config exists
            self._config = Config()
            return self._config

        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            # Return defaults on error
            self._config = Config()
            return self._config

        # Parse provider settings
        provider_data = data.get("provider", {})
        provider = ProviderSettings(
            model_name=provider_data.get("model_name", "gpt-4o"),
            base_url=provider_data.get("base_url"),
            api_key=provider_data.get("api_key"),
            temperature=provider_data.get("temperature", 0.7),
            max_tokens=provider_data.get("max_tokens"),
            provider_type=provider_data.get("provider_type"),
        )

        # Parse daemon settings
        daemon_data = data.get("daemon", {})
        daemon = DaemonSettings(
            host=daemon_data.get("host", "127.0.0.1"),
            port=daemon_data.get("port", 8765),
            log_level=daemon_data.get("log_level", "info"),
        )

        # Get extra settings
        extra = {k: v for k, v in data.items() if k not in ("provider", "daemon")}

        self._config = Config(provider=provider, daemon=daemon, extra=extra)
        return self._config

    def save(self, config: Config | None = None) -> None:
        """Save configuration to file."""
        if config is not None:
            self._config = config

        if self._config is None:
            self._config = Config()

        # Ensure parent directory exists
        self._config_path.parent.mkdir(parents=True, exist_ok=True)

        data: dict[str, Any] = {
            "provider": {
                "model_name": self._config.provider.model_name,
                "base_url": self._config.provider.base_url,
                "api_key": self._config.provider.api_key,
                "temperature": self._config.provider.temperature,
                "max_tokens": self._config.provider.max_tokens,
                "provider_type": self._config.provider.provider_type,
            },
            "daemon": {
                "host": self._config.daemon.host,
                "port": self._config.daemon.port,
                "log_level": self._config.daemon.log_level,
            },
            **self._config.extra,
        }

        with open(self._config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def get(self) -> Config:
        """Get current configuration, loading if necessary."""
        if self._config is None:
            return self.load()
        return self._config

    def update_provider(self, **kwargs: Any) -> Config:
        """Update provider settings and save."""
        config = self.get()
        for key, value in kwargs.items():
            if hasattr(config.provider, key):
                setattr(config.provider, key, value)
        self.save(config)
        return config

    def update_daemon(self, **kwargs: Any) -> Config:
        """Update daemon settings and save."""
        config = self.get()
        for key, value in kwargs.items():
            if hasattr(config.daemon, key):
                setattr(config.daemon, key, value)
        self.save(config)
        return config

    def reload(self) -> Config:
        """Reload configuration from file."""
        return self.load()


# Global config manager instance
_config_manager: ConfigManager | None = None


def get_config_manager(config_path: Path | None = None) -> ConfigManager:
    """Get or create the global config manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    return _config_manager


def load_config(config_path: Path | None = None) -> Config:
    """Load and return configuration."""
    manager = get_config_manager(config_path)
    return manager.load()


def save_config(config: Config, config_path: Path | None = None) -> None:
    """Save configuration."""
    manager = get_config_manager(config_path)
    manager.save(config)
