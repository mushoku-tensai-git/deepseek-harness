"""LS Companion Backend - Configuration and Daemon."""
from __future__ import annotations

from .config import (
    Config,
    ConfigManager,
    DaemonSettings,
    ProviderSettings,
    get_config_manager,
    get_config_path,
    load_config,
    save_config,
)

__version__ = "1.0.0"
__all__ = [
    "Config",
    "ConfigManager",
    "DaemonSettings",
    "ProviderSettings",
    "get_config_manager",
    "get_config_path",
    "load_config",
    "save_config",
]
