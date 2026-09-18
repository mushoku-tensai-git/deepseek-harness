"""FastAPI / WebSocket daemon entrypoint for LS Companion.

This daemon provides:
- WebSocket server at ws://127.0.0.1:<PORT>/ws/agent
- REST endpoints for configuration management
- Integration with DeepSeek Harness plugin architecture
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import Config, ConfigManager, ProviderSettings, get_config_manager
from harness.core.runner import HarnessEvent, HarnessRunner
from harness.plugins.base_plugin import BaseProviderPlugin
from harness.plugins.providers.litellm_provider_plugin import (
    LiteLLMProviderPlugin,
    ProviderConfig,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ls-companion-daemon")


class ProviderUpdateRequest(BaseModel):
    """Request body for updating provider configuration."""
    model_name: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    provider_type: str | None = None


class PromptRequest(BaseModel):
    """Request body for sending a prompt via WebSocket."""
    prompt: str
    session_id: str | None = None
    system_message: str | None = None


@dataclass
class AppState:
    """Application state shared across the daemon."""
    config_manager: ConfigManager
    provider: BaseProviderPlugin | None = None
    runner: HarnessRunner | None = None
    websocket_connections: set[WebSocket] = frozenset()


async def create_provider(config: ProviderSettings) -> BaseProviderPlugin:
    """Create and initialize a provider plugin from config."""
    provider_config = ProviderConfig(
        model_name=config.model_name,
        base_url=config.base_url,
        api_key=config.api_key,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        provider_type=config.provider_type,
    )
    provider = LiteLLMProviderPlugin(config=provider_config)
    await provider.initialize()
    return provider


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[dict[str, Any]]:
    """Manage application lifecycle."""
    config_manager = get_config_manager()
    config = config_manager.load()
    
    # Initialize provider
    provider = await create_provider(config.provider)
    runner = HarnessRunner(provider=provider)
    
    state = {
        "config_manager": config_manager,
        "provider": provider,
        "runner": runner,
        "websocket_connections": set(),
    }
    
    logger.info(f"Daemon started with provider: {provider.metadata.name}")
    
    yield state
    
    # Shutdown
    if state["runner"]:
        await state["runner"].shutdown()
    logger.info("Daemon shutdown complete")


app = FastAPI(
    title="LS Companion Daemon",
    description="Headless AI agent daemon for LS Companion desktop shell",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ready", "harness": "deepseek-plugin-engine"}


@app.post("/api/config/provider")
async def update_provider(request: ProviderUpdateRequest) -> dict[str, Any]:
    """Hot-reload the active provider plugin configuration."""
    # Access state safely - may not be available in all contexts (e.g., tests)
    try:
        state = app.state._state  # type: ignore
        config_manager = state["config_manager"]
        runner = state["runner"]
        current_provider = state["provider"]
    except (AttributeError, KeyError):
        # Fallback to direct config loading if state not available
        config_manager = get_config_manager()
        runner = None
        current_provider = None
    
    # Update config file
    update_data = request.model_dump(exclude_none=True)
    if update_data:
        config_manager.update_provider(**update_data)
    
    # Reload config
    new_config = config_manager.load()
    
    # Create new provider
    new_provider = await create_provider(new_config.provider)
    
    # Shutdown old provider
    if current_provider:
        await current_provider.shutdown()
    
    # Replace provider
    try:
        state = app.state._state  # type: ignore
        state["provider"] = new_provider
        if state["runner"]:
            state["runner"].provider = new_provider
    except (AttributeError, KeyError):
        pass  # State not available (e.g., in tests)
    
    logger.info(f"Provider updated to: {new_provider.metadata.name}")
    
    return {
        "status": "success",
        "provider": new_provider.metadata.name,
        "model": new_config.provider.model_name,
    }


@app.get("/api/config")
async def get_current_config() -> dict[str, Any]:
    """Get current configuration."""
    # Access state safely - may not be available in all contexts (e.g., tests)
    try:
        state = app.state._state  # type: ignore
        config_manager = state["config_manager"]
        config = config_manager.get()
    except (AttributeError, KeyError):
        # Fallback to direct config loading if state not available
        config_manager = get_config_manager()
        config = config_manager.load()
    
    return {
        "provider": {
            "model_name": config.provider.model_name,
            "base_url": config.provider.base_url,
            "temperature": config.provider.temperature,
            "max_tokens": config.provider.max_tokens,
            "provider_type": config.provider.provider_type,
        },
        "daemon": {
            "host": config.daemon.host,
            "port": config.daemon.port,
            "log_level": config.daemon.log_level,
        },
    }


def event_to_dict(event: HarnessEvent) -> dict[str, Any]:
    """Convert a HarnessEvent to a WebSocket-compatible dict."""
    return {"event": event.event_type, "data": event.data}


async def handle_prompt_stream(
    websocket: WebSocket,
    prompt: str,
    session_id: str | None,
    system_message: str | None,
    runner: HarnessRunner,
) -> None:
    """Handle a prompt by streaming events back via WebSocket."""
    try:
        async for event in runner.run_stream(
            prompt=prompt,
            session_id=session_id,
            system_message=system_message,
        ):
            event_dict = event_to_dict(event)
            await websocket.send_json(event_dict)
    except Exception as e:
        logger.error(f"Error during prompt streaming: {e}")
        await websocket.send_json({
            "event": "error",
            "data": {"message": str(e)},
        })
    finally:
        await websocket.send_json({"event": "complete", "data": None})


@app.websocket("/ws/agent")
async def websocket_agent(websocket: WebSocket) -> None:
    """WebSocket endpoint for agent communication."""
    await websocket.accept()
    
    state = app.state._state  # type: ignore
    state["websocket_connections"].add(websocket)
    
    logger.info("New WebSocket connection established")
    
    try:
        while True:
            # Wait for incoming message
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_json({
                    "event": "error",
                    "data": {"message": "Invalid JSON"},
                })
                continue
            
            msg_type = message.get("type")
            
            if msg_type == "prompt":
                prompt_request = PromptRequest(
                    prompt=message.get("prompt", ""),
                    session_id=message.get("session_id"),
                    system_message=message.get("system_message"),
                )
                
                if not prompt_request.prompt:
                    await websocket.send_json({
                        "event": "error",
                        "data": {"message": "Prompt is required"},
                    })
                    continue
                
                # Handle the prompt and stream response
                if state["runner"]:
                    await handle_prompt_stream(
                        websocket=websocket,
                        prompt=prompt_request.prompt,
                        session_id=prompt_request.session_id,
                        system_message=prompt_request.system_message,
                        runner=state["runner"],
                    )
                else:
                    await websocket.send_json({
                        "event": "error",
                        "data": {"message": "Harness runner not initialized"},
                    })
            
            elif msg_type == "ping":
                await websocket.send_json({"event": "pong"})
            
            else:
                await websocket.send_json({
                    "event": "error",
                    "data": {"message": f"Unknown message type: {msg_type}"},
                })
    
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "event": "error",
                "data": {"message": str(e)},
            })
        except Exception:
            pass
    finally:
        state["websocket_connections"].discard(websocket)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="LS Companion Daemon - Headless AI agent server"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to run the WebSocket server on (default: 8765)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="Logging level (default: info)",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    # Set log level
    logging.getLogger().setLevel(args.log_level.upper())
    
    # Set config path if provided
    if args.config:
        global _config_manager
        _config_manager = ConfigManager(Path(args.config))
    
    # Import uvicorn here to avoid issues if not installed
    import uvicorn
    
    logger.info(f"Starting LS Companion Daemon on {args.host}:{args.port}")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
