"""
SENSE-v2 Experiential Grounding
Tier 3: Action outcome verification via internal state and tool results.

Part of Phase 1: Three-Tier Grounding System

Bridges to existing sense/tools/ modules for verification.
"""

import os
import json
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path

# Assuming imports from tools
try:
    from ..tools.filesystem import FileSystemVerifier  # Hypothetical class
    from ..tools.terminal import CommandVerifier
except ImportError:
    # Fallback to standard lib
    FileSystemVerifier = None
    CommandVerifier = None

@dataclass
class GroundingResult:
    verified: bool
    confidence: float = 1.0
    explanation: str = field(default="")
    source: str = "experiential"
    details: Dict[str, Any] = field(default_factory=dict)

class ExperientialGrounding:
    """
    Tier 3 Grounding: Verifies experiential claims based on internal state changes,
    file system outcomes, and command executions.
    """
    
    def __init__(self, use_tools: bool = True):
        self.use_tools = use_tools
        self.fs_verifier = FileSystemVerifier() if use_tools and FileSystemVerifier else None
        self.cmd_verifier = CommandVerifier() if use_tools and CommandVerifier else None
        self.state_history = []  # For tracking state changes

    def record_state(self, state: Dict[str, Any]):
        """Record current state for diff verification."""
        self.state_history.append(state)

    def verify_file_exists(self, path: str) -> GroundingResult:
        """
        Verify if a file exists post-action.
        Uses FileSystemTool if available, else os.path.
        """
        try:
            full_path = Path(path).resolve()
            if self.fs_verifier:
                exists = self.fs_verifier.check_exists(str(full_path))
            else:
                exists = full_path.exists()
            
            explanation = f"File existence check for '{path}': {'exists' if exists else 'does not exist'}"
            confidence = 1.0 if exists else 0.8  # Slightly lower if not exists, potential race condition
            
            return GroundingResult(
                verified=exists,
                confidence=confidence,
                explanation=explanation,
                source="experiential",
                details={"path": str(full_path), "exists": exists}
            )
        except Exception as e:
            return GroundingResult(
                verified=False,
                confidence=0.0,
                explanation=f"Error verifying file '{path}': {str(e)}",
                source="experiential"
            )

    def verify_command_succeeded(self, command: str, result: Dict[str, Any]) -> GroundingResult:
        """
        Verify if a command succeeded based on its result.
        result should have 'returncode', 'stdout', 'stderr'.
        """
        try:
            returncode = result.get('returncode', 1)
            stdout = result.get('stdout', '')
            stderr = result.get('stderr', '')
            
            succeeded = returncode == 0
            confidence = 1.0 if succeeded else min(0.9, 1.0 - len(stderr)/1000)  # Penalize based on error length
            
            explanation = f"Command '{command}' returncode: {returncode}, succeeded: {succeeded}"
            if stderr:
                explanation += f" Errors: {stderr[:200]}..."
            
            details = {
                "command": command,
                "returncode": returncode,
                "has_stdout": bool(stdout),
                "has_stderr": bool(stderr)
            }
            
            if self.cmd_verifier:
                # Additional verification using tool
                tool_result = self.cmd_verifier.validate_result(result)
                if tool_result:
                    succeeded = succeeded and tool_result.verified
                    confidence *= tool_result.confidence
                    explanation += f" Tool validation: {tool_result.explanation}"
            
            return GroundingResult(
                verified=succeeded,
                confidence=confidence,
                explanation=explanation,
                source="experiential",
                details=details
            )
        except Exception as e:
            return GroundingResult(
                verified=False,
                confidence=0.0,
                explanation=f"Error verifying command '{command}': {str(e)}",
                source="experiential"
            )

    def verify_state_change(self, expected_change: Dict[str, Any], current_state: Optional[Dict[str, Any]] = None) -> GroundingResult:
        """
        Verify if state changed as expected.
        Compares previous recorded state with current.
        """
        try:
            if len(self.state_history) < 1:
                return GroundingResult(
                    verified=False,
                    confidence=0.5,
                    explanation="No previous state recorded for comparison.",
                    source="experiential"
                )
            
            prev_state = self.state_history[-1]
            curr_state = current_state or {}  # Assume passed or get from context
            
            verified = True
            explanation_parts = []
            
            for key, expected in expected_change.items():
                prev_val = prev_state.get(key)
                curr_val = curr_state.get(key)
                
                if isinstance(expected, dict) and 'operator' in expected:
                    # Support simple operators like 'eq', 'ne', 'gt'
                    op = expected['operator']
                    val = expected['value']
                    
                    if op == 'eq':
                        match = curr_val == val
                    elif op == 'ne':
                        match = curr_val != val
                    elif op == 'changed':
                        match = prev_val != curr_val
                    else:
                        match = False
                    
                    verified = verified and match
                    explanation_parts.append(f"{key}: {curr_val} {op} {val} -> {match}")
                else:
                    # Simple equality
                    match = curr_val == expected
                    verified = verified and match
                    explanation_parts.append(f"{key}: expected {expected}, got {curr_val} -> {match}")
            
            explanation = "; ".join(explanation_parts)
            confidence = 1.0 if verified else 0.7  # Medium confidence if partial match
            
            # Update history
            if current_state:
                self.state_history.append(current_state)
            
            details = {
                "previous_state": prev_state,
                "current_state": curr_state,
                "expected_change": expected_change,
                "verified_keys": list(expected_change.keys())
            }
            
            return GroundingResult(
                verified=verified,
                confidence=confidence,
                explanation=explanation,
                source="experiential",
                details=details
            )
        except Exception as e:
            return GroundingResult(
                verified=False,
                confidence=0.0,
                explanation=f"Error verifying state change: {str(e)}",
                source="experiential"
            )

    def verify_action_outcome(self, action_type: str, action_params: Dict, outcome: Any) -> GroundingResult:
        """
        General method to verify outcome of various actions.
        Dispatches to specific verifiers based on action_type.
        """
        if action_type == "file_create":
            return self.verify_file_exists(action_params.get("path"))
        elif action_type == "command_exec":
            return self.verify_command_succeeded(
                action_params.get("command", ""), outcome
            )
        elif action_type == "state_update":
            return self.verify_state_change(action_params.get("expected_change", {}), outcome)
        else:
            return GroundingResult(
                verified=False,
                confidence=0.5,
                explanation=f"Unknown action type '{action_type}' for experiential verification.",
                source="experiential"
            )

# Example usage and tests (for ~120 lines)
if __name__ == "__main__":
    grounding = ExperientialGrounding(use_tools=False)
    
    # Test file exists
    result1 = grounding.verify_file_exists("/tmp/test.txt")
    print(f"File verify: {result1.verified}, {result1.explanation}")
    
    # Test command (mock)
    mock_result = {"returncode": 0, "stdout": "success", "stderr": ""}
    result2 = grounding.verify_command_succeeded("echo success", mock_result)
    print(f"Command verify: {result2.verified}, {result2.explanation}")
    
    # Test state change
    grounding.record_state({"count": 0})
    result3 = grounding.verify_state_change({"count": {"operator": "eq", "value": 1}}, {"count": 1})
    print(f"State verify: {result3.verified}, {result3.explanation}")
