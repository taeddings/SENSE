"""
SENSE Bridge Safety Module
AST validation and command whitelist for secure command execution.

Part of Phase 1: Milestone 1.3 - Bridge & OS Control

Safety Features:
- Command whitelist with allowed base commands
- Forbidden pattern detection (destructive operations)
- AST-based Python code validation using RestrictedPython
- Path traversal detection
- Injection attempt detection
"""

import re
import ast
import shlex
from typing import List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum

# RestrictedPython for code sandboxing
try:
    from RestrictedPython import compile_restricted, safe_globals
    from RestrictedPython.Guards import safe_builtins
    RESTRICTED_PYTHON_AVAILABLE = True
except ImportError:
    RESTRICTED_PYTHON_AVAILABLE = False


class SafetyViolation(Enum):
    """Types of safety violations."""
    BLOCKED_COMMAND = "blocked_command"
    FORBIDDEN_PATTERN = "forbidden_pattern"
    PATH_TRAVERSAL = "path_traversal"
    INJECTION_ATTEMPT = "injection_attempt"
    UNSAFE_CODE = "unsafe_code"
    UNKNOWN_COMMAND = "unknown_command"


@dataclass
class SafetyCheckResult:
    """Result of a safety check."""
    safe: bool
    violation: Optional[SafetyViolation] = None
    reason: str = ""
    matched_pattern: str = ""


# Command whitelist from CLAUDE.md
COMMAND_WHITELIST: Set[str] = {
    # File operations (safe subset)
    "ls", "cat", "head", "tail", "less", "more",
    "cp", "mv", "mkdir", "touch", "find", "grep",
    "wc", "sort", "uniq", "diff", "file",
    # Navigation
    "pwd", "cd", "echo", "printf",
    # Development
    "python", "python3", "pip", "pip3",
    "git", "make", "cmake",
    # Package managers
    "pkg",        # Termux
    "apt", "apt-get", "dpkg",  # Debian/Ubuntu
    # Network (limited)
    "curl", "wget", "ping",
    # System info (read-only)
    "uname", "whoami", "id", "date", "uptime",
    "df", "du", "free", "top", "ps",
    # Text processing
    "awk", "sed", "cut", "tr",
    # Archives
    "tar", "gzip", "gunzip", "zip", "unzip",
    # Permissions (limited)
    "chmod", "chown",
}

# Forbidden patterns from CLAUDE.md plus additional dangerous patterns
FORBIDDEN_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # Destructive file operations
    (re.compile(r"rm\s+-rf\s+/"), "Recursive deletion of root"),
    (re.compile(r"rm\s+-rf\s+/\*"), "Recursive deletion of root contents"),
    (re.compile(r"rm\s+-rf\s+~"), "Recursive deletion of home directory"),
    (re.compile(r"rm\s+-rf\s+\$HOME"), "Recursive deletion of HOME"),
    (re.compile(r"rm\s+-rf\s+\.\s*$"), "Recursive deletion of current directory"),

    # Disk/device operations
    (re.compile(r"mkfs"), "Filesystem creation"),
    (re.compile(r"dd\s+if=.*/dev/"), "Direct device read"),
    (re.compile(r"dd\s+of=.*/dev/"), "Direct device write"),
    (re.compile(r">\s*/dev/"), "Redirect to device"),
    (re.compile(r">\s*/dev/sd"), "Write to disk device"),
    (re.compile(r">\s*/dev/null\s*2>&1\s*&"), "Background with suppressed output"),

    # Permission dangers
    (re.compile(r"chmod\s+777\s+/"), "World-writable root"),
    (re.compile(r"chmod\s+-R\s+777"), "Recursive world-writable"),
    (re.compile(r"chown\s+-R\s+.*\s+/\s*$"), "Recursive chown of root"),

    # Fork bombs and resource exhaustion
    (re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:"), "Fork bomb"),
    (re.compile(r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:"), "Fork bomb variant"),

    # Network dangers
    (re.compile(r"nc\s+-e"), "Netcat with execute"),
    (re.compile(r"ncat\s+-e"), "Ncat with execute"),
    (re.compile(r"bash\s+-i\s+>&\s*/dev/tcp/"), "Reverse shell"),

    # History/credential dangers
    (re.compile(r"history\s*-c"), "Clear command history"),
    (re.compile(r"cat\s+.*\.ssh/"), "Read SSH credentials"),
    (re.compile(r"cat\s+.*/etc/shadow"), "Read shadow file"),
    (re.compile(r"cat\s+.*/etc/passwd"), "Read passwd file"),

    # Sudo/privilege escalation
    (re.compile(r"sudo\s+su\s*-"), "Sudo to root"),
    (re.compile(r"sudo\s+-i"), "Sudo interactive"),
    (re.compile(r"sudo\s+bash"), "Sudo bash"),

    # Cron/scheduled tasks
    (re.compile(r"crontab\s+-r"), "Remove crontab"),
    (re.compile(r"crontab\s+-e"), "Edit crontab"),

    # System critical
    (re.compile(r"shutdown"), "System shutdown"),
    (re.compile(r"reboot"), "System reboot"),
    (re.compile(r"init\s+[0-6]"), "Change runlevel"),
    (re.compile(r"systemctl\s+stop"), "Stop services"),
    (re.compile(r"killall"), "Kill all processes"),
]

# Patterns indicating path traversal attempts
PATH_TRAVERSAL_PATTERNS: List[re.Pattern] = [
    re.compile(r"\.\./\.\./"),  # Multiple parent traversal
    re.compile(r"\.\.\\\.\.\\"),  # Windows-style traversal
    re.compile(r"%2e%2e/", re.IGNORECASE),  # URL-encoded traversal
    re.compile(r"\.\.%2f", re.IGNORECASE),  # Mixed encoding
]

# Patterns indicating injection attempts
INJECTION_PATTERNS: List[re.Pattern] = [
    re.compile(r"\$\(.*\)"),  # Command substitution
    re.compile(r"`.*`"),  # Backtick substitution
    re.compile(r";\s*rm\s"),  # Injection followed by rm
    re.compile(r"\|\s*bash"),  # Pipe to bash
    re.compile(r"\|\s*sh\s"),  # Pipe to sh
    re.compile(r"&&\s*curl.*\|.*bash"),  # Download and execute
]


def check_command_safety(command: str, strict_whitelist: bool = False) -> SafetyCheckResult:
    """
    Check if a command is safe to execute.

    Args:
        command: The shell command to check
        strict_whitelist: If True, only allow whitelisted commands

    Returns:
        SafetyCheckResult indicating if command is safe
    """
    if not command or not command.strip():
        return SafetyCheckResult(safe=True)

    # Check for forbidden patterns first (highest priority)
    for pattern, reason in FORBIDDEN_PATTERNS:
        if pattern.search(command):
            return SafetyCheckResult(
                safe=False,
                violation=SafetyViolation.FORBIDDEN_PATTERN,
                reason=reason,
                matched_pattern=pattern.pattern,
            )

    # Check for path traversal
    for pattern in PATH_TRAVERSAL_PATTERNS:
        if pattern.search(command):
            return SafetyCheckResult(
                safe=False,
                violation=SafetyViolation.PATH_TRAVERSAL,
                reason="Path traversal attempt detected",
                matched_pattern=pattern.pattern,
            )

    # Check for injection attempts
    for pattern in INJECTION_PATTERNS:
        if pattern.search(command):
            return SafetyCheckResult(
                safe=False,
                violation=SafetyViolation.INJECTION_ATTEMPT,
                reason="Command injection attempt detected",
                matched_pattern=pattern.pattern,
            )

    # Extract base command for whitelist check
    try:
        parts = shlex.split(command)
        if not parts:
            return SafetyCheckResult(safe=True)
        base_command = parts[0].split("/")[-1]  # Handle full paths
    except ValueError:
        # shlex.split failed, likely malformed command
        return SafetyCheckResult(
            safe=False,
            violation=SafetyViolation.INJECTION_ATTEMPT,
            reason="Malformed command (quote parsing failed)",
        )

    # Check whitelist
    if strict_whitelist and base_command not in COMMAND_WHITELIST:
        return SafetyCheckResult(
            safe=False,
            violation=SafetyViolation.UNKNOWN_COMMAND,
            reason=f"Command '{base_command}' not in whitelist",
        )

    return SafetyCheckResult(safe=True)


def check_python_safety(code: str) -> SafetyCheckResult:
    """
    Check if Python code is safe to execute using RestrictedPython.

    Args:
        code: Python code to check

    Returns:
        SafetyCheckResult indicating if code is safe
    """
    if not RESTRICTED_PYTHON_AVAILABLE:
        return SafetyCheckResult(
            safe=False,
            violation=SafetyViolation.UNSAFE_CODE,
            reason="RestrictedPython not available for code validation",
        )

    try:
        # Try to compile with RestrictedPython
        result = compile_restricted(code, "<string>", "exec")

        if result.errors:
            return SafetyCheckResult(
                safe=False,
                violation=SafetyViolation.UNSAFE_CODE,
                reason="; ".join(result.errors),
            )

        return SafetyCheckResult(safe=True)

    except SyntaxError as e:
        return SafetyCheckResult(
            safe=False,
            violation=SafetyViolation.UNSAFE_CODE,
            reason=f"Syntax error: {e}",
        )
    except Exception as e:
        return SafetyCheckResult(
            safe=False,
            violation=SafetyViolation.UNSAFE_CODE,
            reason=f"Validation error: {e}",
        )


def check_path_safety(path: str, allowed_roots: Optional[List[str]] = None) -> SafetyCheckResult:
    """
    Check if a file path is safe to access.

    Args:
        path: The file path to check
        allowed_roots: List of allowed root directories (None = no restriction)

    Returns:
        SafetyCheckResult indicating if path is safe
    """
    import os

    # Check for traversal patterns
    for pattern in PATH_TRAVERSAL_PATTERNS:
        if pattern.search(path):
            return SafetyCheckResult(
                safe=False,
                violation=SafetyViolation.PATH_TRAVERSAL,
                reason="Path traversal pattern detected",
            )

    # Normalize the path
    try:
        normalized = os.path.normpath(os.path.abspath(path))
    except Exception:
        return SafetyCheckResult(
            safe=False,
            violation=SafetyViolation.PATH_TRAVERSAL,
            reason="Invalid path format",
        )

    # Check against allowed roots
    if allowed_roots:
        allowed = False
        for root in allowed_roots:
            norm_root = os.path.normpath(os.path.abspath(root))
            if normalized.startswith(norm_root):
                allowed = True
                break

        if not allowed:
            return SafetyCheckResult(
                safe=False,
                violation=SafetyViolation.PATH_TRAVERSAL,
                reason=f"Path '{path}' outside allowed directories",
            )

    # Block sensitive paths
    sensitive_paths = [
        "/etc/shadow",
        "/etc/passwd",
        "/etc/sudoers",
        "/root/.ssh",
        "/home/*/.ssh",
    ]

    for sensitive in sensitive_paths:
        if sensitive.endswith("*"):
            # Wildcard match
            base = sensitive[:-1]
            if normalized.startswith(base):
                return SafetyCheckResult(
                    safe=False,
                    violation=SafetyViolation.PATH_TRAVERSAL,
                    reason=f"Access to sensitive path blocked",
                )
        elif normalized == sensitive or normalized.startswith(sensitive + "/"):
            return SafetyCheckResult(
                safe=False,
                violation=SafetyViolation.PATH_TRAVERSAL,
                reason=f"Access to sensitive path blocked",
            )

    return SafetyCheckResult(safe=True)


def sanitize_command(command: str) -> str:
    """
    Attempt to sanitize a command by removing dangerous elements.

    Note: This is a best-effort sanitization. When in doubt, reject the command.

    Args:
        command: The command to sanitize

    Returns:
        Sanitized command string
    """
    # Remove command substitutions
    command = re.sub(r"\$\([^)]*\)", "", command)
    command = re.sub(r"`[^`]*`", "", command)

    # Remove semicolon-separated additional commands
    command = command.split(";")[0].strip()

    # Remove pipe chains to shells
    if "|" in command:
        parts = command.split("|")
        safe_parts = []
        for part in parts:
            part = part.strip()
            if not any(shell in part for shell in ["bash", "sh", "zsh", "csh"]):
                safe_parts.append(part)
            else:
                break
        command = " | ".join(safe_parts)

    return command.strip()


def get_safe_builtins() -> dict:
    """
    Get safe Python builtins for restricted execution.

    Returns:
        Dictionary of safe builtin functions
    """
    if not RESTRICTED_PYTHON_AVAILABLE:
        return {}

    # Start with RestrictedPython's safe builtins
    builtins = dict(safe_builtins)

    # Add commonly needed safe functions
    safe_additions = {
        "len": len,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "map": map,
        "filter": filter,
        "sorted": sorted,
        "reversed": reversed,
        "sum": sum,
        "min": min,
        "max": max,
        "abs": abs,
        "round": round,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "type": type,
        "isinstance": isinstance,
        "hasattr": hasattr,
        "getattr": getattr,
    }

    builtins.update(safe_additions)
    return builtins
