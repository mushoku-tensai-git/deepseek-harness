"""Security & Permission Plugin for DeepSeek Harness in LS Companion.

This plugin implements a strict approval gate for all tool executions,
with 3-tier scoped permission storage and async approval handshake.
"""
from __future__ import annotations

import asyncio
import fnmatch
import json
import os
import sys
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from ..plugins.base_plugin import BasePlugin, PluginMetadata


class RiskLevel(Enum):
    """Risk level classification for tool actions."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ActionType(Enum):
    """Types of actions that require permission."""
    SHELL = "shell"
    FILE_WRITE = "file_write"
    FILE_READ = "file_read"
    GUI_CLICK = "gui_click"
    SYSTEM_SETTING = "system_setting"
    DIRECTORY_LIST = "directory_list"
    SCREENSHOT = "screenshot"


class Decision(Enum):
    """Permission decision types."""
    ALLOW_ONCE = "allow_once"
    ALLOW_WORKSPACE = "allow_workspace"
    ALLOW_GLOBAL = "allow_global"
    DENY = "deny"


@dataclass
class PermissionRequest:
    """Represents a permission request pending user approval."""
    request_id: str
    action_type: ActionType
    target: str
    risk_level: RiskLevel
    diff: str | None
    explainer: str
    workspace: str
    tool_name: str
    arguments: dict[str, Any]
    future: asyncio.Future[Any] = field(default_factory=lambda: asyncio.get_event_loop().create_future())


@dataclass
class Rule:
    """A permission rule for matching actions."""
    pattern: str
    action_type: str | None  # e.g., "cmd", "file_write", "file_read"
    decision: Decision
    scope: str  # "global" or "workspace"
    description: str | None = None

    def matches(self, action_type: ActionType, target: str) -> bool:
        """Check if this rule matches the given action."""
        # If action_type is specified in rule, it must match
        if self.action_type is not None:
            rule_action_map = {
                "cmd": ActionType.SHELL,
                "shell": ActionType.SHELL,
                "file_write": ActionType.FILE_WRITE,
                "file_read": ActionType.FILE_READ,
                "gui_click": ActionType.GUI_CLICK,
                "system_setting": ActionType.SYSTEM_SETTING,
                "directory_list": ActionType.DIRECTORY_LIST,
                "screenshot": ActionType.SCREENSHOT,
            }
            rule_action = rule_action_map.get(self.action_type.lower())
            if rule_action != action_type:
                return False

        # Check if target matches the pattern (supports glob wildcards)
        return fnmatch.fnmatch(target, self.pattern)


@dataclass
class RulesFile:
    """Structure of a rules JSON file."""
    rules: list[dict[str, Any]] = field(default_factory=list)
    version: str = "1.0"


class PermissionDeniedException(Exception):
    """Exception raised when permission is denied for a tool execution."""

    def __init__(self, message: str, request_id: str | None = None):
        super().__init__(message)
        self.request_id = request_id


def get_global_rules_path() -> Path:
    """Get the platform-specific global rules file path.
    
    Returns:
        - Windows: %APPDATA%/LadeStack/LSCompanion/global_rules.json
        - Linux: ~/.config/ladestack/lscompanion/global_rules.json
        - macOS: ~/Library/Application Support/LadeStack/LSCompanion/global_rules.json
    """
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA", "")
        if appdata:
            return Path(appdata) / "LadeStack" / "LSCompanion" / "global_rules.json"
        userprofile = os.environ.get("USERPROFILE", "")
        if userprofile:
            return Path(userprofile) / "AppData" / "Roaming" / "LadeStack" / "LSCompanion" / "global_rules.json"
    elif sys.platform == "darwin":
        home = Path.home()
        return home / "Library" / "Application Support" / "LadeStack" / "LSCompanion" / "global_rules.json"
    else:
        # Linux and other Unix-like systems
        xdg_config = os.environ.get("XDG_CONFIG_HOME", "")
        if xdg_config:
            return Path(xdg_config) / "ladestack" / "lscompanion" / "global_rules.json"
        home = Path.home()
        return home / ".config" / "ladestack" / "lscompanion" / "global_rules.json"


def get_workspace_rules_path(workspace_path: str) -> Path:
    """Get the workspace-specific rules file path.
    
    Args:
        workspace_path: Path to the current workspace
        
    Returns:
        Path to <workspace>/.ladestack/rules.json
    """
    return Path(workspace_path) / ".ladestack" / "rules.json"


# Read-only tools that are auto-approved
READ_ONLY_TOOLS = {
    "read_file",
    "list_dir",
    "take_screenshot",
    "get_file_info",
    "search_files",
    "grep",
    "cat",
    "ls",
    "pwd",
    "whoami",
    "date",
    "echo",
}

# Tools that modify files
FILE_WRITE_TOOLS = {
    "write_file",
    "append_file",
    "delete_file",
    "rename_file",
    "copy_file",
    "move_file",
    "create_directory",
    "delete_directory",
    "edit_file",
    "patch_file",
}

# Tools that execute commands
SHELL_TOOLS = {
    "run_shell",
    "execute_command",
    "shell",
    "bash",
    "sh",
    "cmd",
    "powershell",
}

# Tools that interact with GUI
GUI_TOOLS = {
    "click_element",
    "type_text",
    "press_key",
    "mouse_move",
    "double_click",
    "right_click",
    "drag_drop",
}

# Tools that modify system settings
SYSTEM_SETTING_TOOLS = {
    "set_system_setting",
    "modify_registry",
    "change_environment",
    "install_package",
    "uninstall_package",
    "configure_service",
}


class SecurityPlugin(BasePlugin):
    """Security & Permission Plugin for DeepSeek Harness.
    
    Implements a strict approval gate for all tool executions with:
    - Pre-tool execution interceptor hook
    - 3-tier scoped permission storage (Global, Workspace, Prompt)
    - Async approval handshake via WebSocket events
    - Thread-safe async locks for concurrent access
    """

    # Class-level lock for thread-safe access to shared state
    _class_lock: asyncio.Lock | None = None

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="security-permission-plugin",
            version="1.0.0",
            description="Central Security & Permission Plugin for LS Companion",
            author="LS Companion Team",
        )

    def __init__(
        self,
        global_rules_path: Path | None = None,
        send_permission_event: Callable[[dict[str, Any]], None] | None = None,
    ):
        """Initialize the security plugin.
        
        Args:
            global_rules_path: Optional custom path for global rules
            send_permission_event: Callback to send permission events to WebSocket clients
        """
        self._global_rules_path = global_rules_path or get_global_rules_path()
        self._send_permission_event = send_permission_event
        self._lock = asyncio.Lock()
        self._pending_requests: dict[str, PermissionRequest] = {}
        self._initialized = False
        self._global_rules_cache: list[Rule] = []
        self._workspace_rules_cache: dict[str, list[Rule]] = {}
        self._cache_loaded = False

    @classmethod
    async def get_class_lock(cls) -> asyncio.Lock:
        """Get or create the class-level lock."""
        if cls._class_lock is None:
            cls._class_lock = asyncio.Lock()
        return cls._class_lock

    async def initialize(self) -> None:
        """Initialize the plugin. Load rules from disk."""
        async with self._lock:
            if self._initialized:
                return
            
            await self._load_global_rules()
            self._initialized = True

    async def shutdown(self) -> None:
        """Shutdown the plugin. Clear pending requests."""
        async with self._lock:
            # Cancel all pending requests
            for request in self._pending_requests.values():
                if not request.future.done():
                    request.future.cancel()
            self._pending_requests.clear()
            self._initialized = False

    async def is_available(self) -> bool:
        """Check if the plugin is available and ready."""
        return self._initialized

    def set_permission_callback(
        self, 
        callback: Callable[[dict[str, Any]], None]
    ) -> None:
        """Set the callback for sending permission events.
        
        Args:
            callback: Function to call with permission event data
        """
        self._send_permission_event = callback

    async def _load_global_rules(self) -> list[Rule]:
        """Load global rules from disk."""
        rules: list[Rule] = []
        
        if not self._global_rules_path.exists():
            self._global_rules_cache = rules
            return rules

        try:
            with open(self._global_rules_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for rule_data in data.get("rules", []):
                rule = Rule(
                    pattern=rule_data.get("pattern", ""),
                    action_type=rule_data.get("action_type"),
                    decision=Decision(rule_data.get("decision", "deny")),
                    scope="global",
                    description=rule_data.get("description"),
                )
                rules.append(rule)
        except (json.JSONDecodeError, IOError, ValueError):
            pass

        self._global_rules_cache = rules
        return rules

    async def _load_workspace_rules(self, workspace_path: str) -> list[Rule]:
        """Load workspace-specific rules from disk."""
        rules: list[Rule] = []
        rules_path = get_workspace_rules_path(workspace_path)

        if not rules_path.exists():
            self._workspace_rules_cache[workspace_path] = rules
            return rules

        try:
            with open(rules_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for rule_data in data.get("rules", []):
                rule = Rule(
                    pattern=rule_data.get("pattern", ""),
                    action_type=rule_data.get("action_type"),
                    decision=Decision(rule_data.get("decision", "deny")),
                    scope="workspace",
                    description=rule_data.get("description"),
                )
                rules.append(rule)
        except (json.JSONDecodeError, IOError, ValueError):
            pass

        self._workspace_rules_cache[workspace_path] = rules
        return rules

    async def _save_global_rule(self, rule: Rule) -> None:
        """Save a rule to the global rules file."""
        async with self._lock:
            # Ensure parent directory exists
            self._global_rules_path.parent.mkdir(parents=True, exist_ok=True)

            # Load existing rules
            data: dict[str, Any] = {"rules": [], "version": "1.0"}
            if self._global_rules_path.exists():
                try:
                    with open(self._global_rules_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except (json.JSONDecodeError, IOError):
                    pass

            # Add new rule
            rule_data = {
                "pattern": rule.pattern,
                "action_type": rule.action_type,
                "decision": rule.decision.value,
                "scope": "global",
            }
            if rule.description:
                rule_data["description"] = rule.description
            
            data["rules"].append(rule_data)

            # Save
            with open(self._global_rules_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            # Update cache
            self._global_rules_cache.append(rule)

    async def _save_workspace_rule(self, workspace_path: str, rule: Rule) -> None:
        """Save a rule to the workspace rules file."""
        async with self._lock:
            rules_path = get_workspace_rules_path(workspace_path)
            
            # Ensure parent directory exists
            rules_path.parent.mkdir(parents=True, exist_ok=True)

            # Load existing rules
            data: dict[str, Any] = {"rules": [], "version": "1.0"}
            if rules_path.exists():
                try:
                    with open(rules_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except (json.JSONDecodeError, IOError):
                    pass

            # Add new rule
            rule_data = {
                "pattern": rule.pattern,
                "action_type": rule.action_type,
                "decision": rule.decision.value,
                "scope": "workspace",
            }
            if rule.description:
                rule_data["description"] = rule.description
            
            data["rules"].append(rule_data)

            # Save
            with open(rules_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            # Update cache
            if workspace_path not in self._workspace_rules_cache:
                self._workspace_rules_cache[workspace_path] = []
            self._workspace_rules_cache[workspace_path].append(rule)

    def _classify_tool(self, tool_name: str) -> tuple[ActionType, RiskLevel]:
        """Classify a tool by its action type and risk level.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tuple of (ActionType, RiskLevel)
        """
        tool_lower = tool_name.lower()
        
        if tool_lower in READ_ONLY_TOOLS:
            return ActionType.FILE_READ if tool_lower in ("read_file", "get_file_info", "grep", "cat") else ActionType.SCREENSHOT if tool_lower == "take_screenshot" else ActionType.DIRECTORY_LIST, RiskLevel.LOW
        
        if tool_lower in FILE_WRITE_TOOLS:
            return ActionType.FILE_WRITE, RiskLevel.MEDIUM
        
        if tool_lower in SHELL_TOOLS:
            return ActionType.SHELL, RiskLevel.HIGH
        
        if tool_lower in GUI_TOOLS:
            return ActionType.GUI_CLICK, RiskLevel.HIGH
        
        if tool_lower in SYSTEM_SETTING_TOOLS:
            return ActionType.SYSTEM_SETTING, RiskLevel.HIGH

        # Default: treat unknown tools as medium risk
        return ActionType.SHELL, RiskLevel.MEDIUM

    def _generate_explainer(
        self,
        action_type: ActionType,
        target: str,
        tool_name: str,
        arguments: dict[str, Any],
        risk_level: RiskLevel,
    ) -> str:
        """Generate a plain-language explanation of the action.
        
        Args:
            action_type: Type of action
            target: Target of the action (file path, command, etc.)
            tool_name: Name of the tool
            arguments: Tool arguments
            risk_level: Risk level of the action
            
        Returns:
            Human-readable explanation string
        """
        explanations = {
            ActionType.SHELL: f"This will execute the command `{target}` on your system. This could modify files, install software, or change system configuration.",
            ActionType.FILE_WRITE: f"This will write to or modify the file `{target}`. This could overwrite existing data or create new files.",
            ActionType.FILE_READ: f"This will read the contents of `{target}`. This is a safe, read-only operation.",
            ActionType.GUI_CLICK: f"This will simulate a GUI interaction (click, type, etc.) targeting `{target}`. This could trigger application actions.",
            ActionType.SYSTEM_SETTING: f"This will modify a system setting: `{target}`. This could affect system behavior or security.",
            ActionType.DIRECTORY_LIST: f"This will list the contents of directory `{target}`. This is a safe, read-only operation.",
            ActionType.SCREENSHOT: "This will capture a screenshot of your current screen. This may expose sensitive information visible on screen.",
        }

        base_explanation = explanations.get(action_type, f"This will perform action `{tool_name}` on `{target}`.")

        if risk_level == RiskLevel.HIGH:
            base_explanation += " ⚠️ HIGH RISK: This action could significantly impact your system."
        elif risk_level == RiskLevel.MEDIUM:
            base_explanation += " ⚠️ MEDIUM RISK: Review carefully before approving."
        else:
            base_explanation += " ✅ LOW RISK: This is generally safe."

        return base_explanation

    async def pre_tool_execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        workspace_path: str | None = None,
    ) -> Any:
        """Pre-tool execution interceptor hook.
        
        This method is called before any tool execution to check permissions.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            workspace_path: Current workspace path
            
        Returns:
            None if approved, raises PermissionDeniedException if denied
            
        Raises:
            PermissionDeniedException: If permission is denied
        """
        # Get workspace path
        if workspace_path is None:
            workspace_path = os.getcwd()

        # Classify the tool
        action_type, risk_level = self._classify_tool(tool_name)

        # Determine target based on tool type
        target = self._extract_target(tool_name, arguments, action_type)

        # Auto-approve read-only tools
        if risk_level == RiskLevel.LOW:
            return None

        # Load workspace rules if needed
        if workspace_path not in self._workspace_rules_cache:
            await self._load_workspace_rules(workspace_path)

        # Evaluate rules in order: Deny > Workspace Allow > Global Allow > Prompt User
        decision = await self._evaluate_rules(action_type, target, workspace_path)

        if decision == Decision.DENY:
            raise PermissionDeniedException(
                f"Permission denied for {tool_name} on {target}",
                request_id=None,
            )
        
        if decision in (Decision.ALLOW_ONCE, Decision.ALLOW_WORKSPACE, Decision.ALLOW_GLOBAL):
            return None

        # Need to prompt user
        return await self._prompt_user(
            tool_name=tool_name,
            arguments=arguments,
            action_type=action_type,
            target=target,
            risk_level=risk_level,
            workspace_path=workspace_path,
        )

    def _extract_target(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        action_type: ActionType,
    ) -> str:
        """Extract the target from tool arguments.
        
        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            action_type: Type of action
            
        Returns:
            Target string for rule matching
        """
        # For shell commands, get the command string
        if action_type == ActionType.SHELL:
            cmd = arguments.get("command", arguments.get("cmd", ""))
            if isinstance(cmd, list):
                return " ".join(str(c) for c in cmd)
            return str(cmd)

        # For file operations, get the file path
        if action_type in (ActionType.FILE_WRITE, ActionType.FILE_READ, ActionType.DIRECTORY_LIST):
            return str(arguments.get("path", arguments.get("file", arguments.get("target", ""))))

        # For GUI actions, get the element identifier
        if action_type == ActionType.GUI_CLICK:
            return str(arguments.get("element", arguments.get("target", arguments.get("selector", ""))))

        # For system settings, get the setting name
        if action_type == ActionType.SYSTEM_SETTING:
            return str(arguments.get("setting", arguments.get("name", arguments.get("key", ""))))

        # Fallback: use tool name
        return tool_name

    async def _evaluate_rules(
        self,
        action_type: ActionType,
        target: str,
        workspace_path: str,
    ) -> Decision | None:
        """Evaluate rules against the action.
        
        Rule evaluation order:
        1. Check for explicit DENY rules (global then workspace)
        2. Check for workspace ALLOW rules
        3. Check for global ALLOW rules
        4. Return None to prompt user
        
        Args:
            action_type: Type of action
            target: Target of the action
            workspace_path: Current workspace path
            
        Returns:
            Decision if matched, None to prompt user
        """
        # Get cached rules
        global_rules = self._global_rules_cache
        workspace_rules = self._workspace_rules_cache.get(workspace_path, [])

        # First pass: check for explicit DENY rules (workspace then global)
        for rule in workspace_rules + global_rules:
            if rule.matches(action_type, target) and rule.decision == Decision.DENY:
                return Decision.DENY

        # Second pass: check for workspace ALLOW rules
        for rule in workspace_rules:
            if rule.matches(action_type, target) and rule.decision != Decision.DENY:
                return rule.decision

        # Third pass: check for global ALLOW rules
        for rule in global_rules:
            if rule.matches(action_type, target) and rule.decision != Decision.DENY:
                return rule.decision

        # No matching rule found, prompt user
        return None

    async def _prompt_user(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        action_type: ActionType,
        target: str,
        risk_level: RiskLevel,
        workspace_path: str,
    ) -> None:
        """Prompt user for permission approval.
        
        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            action_type: Type of action
            target: Target of the action
            risk_level: Risk level
            workspace_path: Current workspace path
            
        Returns:
            None if approved, raises PermissionDeniedException if denied
            
        Raises:
            PermissionDeniedException: If user denies or times out
        """
        request_id = str(uuid.uuid4())
        
        # Generate diff if applicable (for file writes)
        diff = None
        if action_type == ActionType.FILE_WRITE:
            diff = arguments.get("diff", arguments.get("content", None))
            if diff and len(str(diff)) > 500:
                diff = str(diff)[:500] + "... (truncated)"

        # Generate explainer
        explainer = self._generate_explainer(
            action_type, target, tool_name, arguments, risk_level
        )

        # Create permission request
        request = PermissionRequest(
            request_id=request_id,
            action_type=action_type,
            target=target,
            risk_level=risk_level,
            diff=diff,
            explainer=explainer,
            workspace=workspace_path,
            tool_name=tool_name,
            arguments=arguments,
        )

        # Store pending request
        async with self._lock:
            self._pending_requests[request_id] = request

        # Send permission event via WebSocket
        if self._send_permission_event:
            event_data = {
                "event": "permission_required",
                "data": {
                    "request_id": request_id,
                    "action_type": action_type.value,
                    "target": target,
                    "diff": diff,
                    "risk_level": risk_level.value,
                    "explainer": explainer,
                    "workspace": workspace_path,
                },
            }
            self._send_permission_event(event_data)

        # Wait for user response (with timeout)
        try:
            await asyncio.wait_for(request.future, timeout=300.0)  # 5 minute timeout
        except asyncio.TimeoutError:
            async with self._lock:
                del self._pending_requests[request_id]
            raise PermissionDeniedException(
                f"Permission request timed out for {tool_name} on {target}",
                request_id=request_id,
            )
        except asyncio.CancelledError:
            async with self._lock:
                del self._pending_requests[request_id]
            raise PermissionDeniedException(
                f"Permission request cancelled for {tool_name} on {target}",
                request_id=request_id,
            )

        # Check decision
        decision = request.future.result()
        
        async with self._lock:
            del self._pending_requests[request_id]

        if decision == Decision.DENY:
            raise PermissionDeniedException(
                f"Permission denied by user for {tool_name} on {target}",
                request_id=request_id,
            )

        # Handle allow decisions
        if decision == Decision.ALLOW_GLOBAL:
            rule = Rule(
                pattern=target,
                action_type=action_type.value,
                decision=Decision.ALLOW_ONCE,  # Store as allow_once for safety
                scope="global",
                description=f"Auto-approved rule for {tool_name}",
            )
            await self._save_global_rule(rule)
        elif decision == Decision.ALLOW_WORKSPACE:
            rule = Rule(
                pattern=target,
                action_type=action_type.value,
                decision=Decision.ALLOW_ONCE,  # Store as allow_once for safety
                scope="workspace",
                description=f"Auto-approved rule for {tool_name}",
            )
            await self._save_workspace_rule(workspace_path, rule)

        # ALLOW_ONCE just proceeds without saving
        return None

    async def resolve_permission(
        self,
        request_id: str,
        decision: str,
    ) -> bool:
        """Resolve a pending permission request.
        
        Called by the POST /api/permissions/resolve endpoint.
        
        Args:
            request_id: The permission request ID
            decision: One of "allow_once", "allow_workspace", "allow_global", "deny"
            
        Returns:
            True if resolved successfully, False if request not found
            
        Raises:
            ValueError: If decision is invalid
        """
        try:
            decision_enum = Decision(decision)
        except ValueError:
            raise ValueError(
                f"Invalid decision: {decision}. Must be one of: {[d.value for d in Decision]}"
            )

        async with self._lock:
            request = self._pending_requests.get(request_id)
            if request is None:
                return False

        # Set the future result
        if not request.future.done():
            request.future.set_result(decision_enum)

        return True

    async def get_pending_requests(self) -> list[dict[str, Any]]:
        """Get all pending permission requests.
        
        Returns:
            List of pending request data dictionaries
        """
        async with self._lock:
            requests = []
            for req in self._pending_requests.values():
                requests.append({
                    "request_id": req.request_id,
                    "action_type": req.action_type.value,
                    "target": req.target,
                    "risk_level": req.risk_level.value,
                    "diff": req.diff,
                    "explainer": req.explainer,
                    "workspace": req.workspace,
                    "tool_name": req.tool_name,
                    "arguments": req.arguments,
                })
            return requests

    async def list_global_rules(self) -> list[dict[str, Any]]:
        """List all global rules.
        
        Returns:
            List of rule dictionaries
        """
        await self._load_global_rules()
        return [
            {
                "pattern": r.pattern,
                "action_type": r.action_type,
                "decision": r.decision.value,
                "scope": r.scope,
                "description": r.description,
            }
            for r in self._global_rules_cache
        ]

    async def list_workspace_rules(self, workspace_path: str) -> list[dict[str, Any]]:
        """List all workspace rules.
        
        Args:
            workspace_path: Path to the workspace
            
        Returns:
            List of rule dictionaries
        """
        await self._load_workspace_rules(workspace_path)
        rules = self._workspace_rules_cache.get(workspace_path, [])
        return [
            {
                "pattern": r.pattern,
                "action_type": r.action_type,
                "decision": r.decision.value,
                "scope": r.scope,
                "description": r.description,
            }
            for r in rules
        ]

    async def delete_global_rule(self, pattern: str, action_type: str | None = None) -> bool:
        """Delete a global rule.
        
        Args:
            pattern: Pattern of the rule to delete
            action_type: Optional action type to match
            
        Returns:
            True if rule was deleted, False if not found
        """
        async with self._lock:
            # Find and remove the rule
            original_len = len(self._global_rules_cache)
            self._global_rules_cache = [
                r for r in self._global_rules_cache
                if not (r.pattern == pattern and (action_type is None or r.action_type == action_type))
            ]
            
            if len(self._global_rules_cache) == original_len:
                return False

            # Rewrite the rules file
            data: dict[str, Any] = {"rules": [], "version": "1.0"}
            for rule in self._global_rules_cache:
                rule_data = {
                    "pattern": rule.pattern,
                    "action_type": rule.action_type,
                    "decision": rule.decision.value,
                    "scope": "global",
                }
                if rule.description:
                    rule_data["description"] = rule.description
                data["rules"].append(rule_data)

            if self._global_rules_path.exists():
                with open(self._global_rules_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            elif data["rules"]:
                self._global_rules_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._global_rules_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)

            return True

    async def delete_workspace_rule(
        self,
        workspace_path: str,
        pattern: str,
        action_type: str | None = None,
    ) -> bool:
        """Delete a workspace rule.
        
        Args:
            workspace_path: Path to the workspace
            pattern: Pattern of the rule to delete
            action_type: Optional action type to match
            
        Returns:
            True if rule was deleted, False if not found
        """
        async with self._lock:
            rules_path = get_workspace_rules_path(workspace_path)
            
            # Load current rules
            if workspace_path not in self._workspace_rules_cache:
                await self._load_workspace_rules(workspace_path)
            
            rules = self._workspace_rules_cache.get(workspace_path, [])
            original_len = len(rules)
            
            # Filter out the rule to delete
            self._workspace_rules_cache[workspace_path] = [
                r for r in rules
                if not (r.pattern == pattern and (action_type is None or r.action_type == action_type))
            ]
            
            if len(self._workspace_rules_cache[workspace_path]) == original_len:
                return False

            # Rewrite the rules file
            data: dict[str, Any] = {"rules": [], "version": "1.0"}
            for rule in self._workspace_rules_cache[workspace_path]:
                rule_data = {
                    "pattern": rule.pattern,
                    "action_type": rule.action_type,
                    "decision": rule.decision.value,
                    "scope": "workspace",
                }
                if rule.description:
                    rule_data["description"] = rule.description
                data["rules"].append(rule_data)

            if rules_path.exists():
                with open(rules_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            elif data["rules"]:
                rules_path.parent.mkdir(parents=True, exist_ok=True)
                with open(rules_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)

            return True


# Convenience function to create a security plugin instance
def create_security_plugin(
    send_permission_event: Callable[[dict[str, Any]], None] | None = None,
) -> SecurityPlugin:
    """Create a new SecurityPlugin instance.
    
    Args:
        send_permission_event: Callback to send permission events to WebSocket clients
        
    Returns:
        New SecurityPlugin instance
    """
    return SecurityPlugin(send_permission_event=send_permission_event)
