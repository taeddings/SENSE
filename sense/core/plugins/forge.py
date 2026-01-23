"""
SENSE v2.3 Tool Forge - Dynamic Tool Creation

Crystallizes repeated successful behaviors into immutable, reusable tools.
Implements the Tool Forge Pipeline:
1. DETECT: Find repeated successful code patterns in memory
2. ABSTRACT: Refactor specific values into parameters
3. VERIFY: Run Tier 1 synthetic tests on the new tool
4. PERSIST: Save to local plugins/user_defined/
5. REGISTER: Hot-load into ToolRegistry

Part of Phase 2: Reasoning & Agency
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
from pathlib import Path
from datetime import datetime
from enum import Enum
import logging
import hashlib
import re
import ast
import importlib.util
import sys
import textwrap


class ForgeStatus(Enum):
    """Status of a forge operation."""
    DETECTED = "detected"
    ABSTRACTED = "abstracted"
    VERIFIED = "verified"
    INSTALLED = "installed"
    FAILED = "failed"


@dataclass
class CodePattern:
    """A detected code pattern from memory."""
    pattern_id: str
    code: str
    occurrences: int
    success_rate: float
    first_seen: datetime
    last_seen: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    extracted_params: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "code": self.code,
            "occurrences": self.occurrences,
            "success_rate": self.success_rate,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "context": self.context,
            "extracted_params": self.extracted_params,
        }


@dataclass
class CandidateSkill:
    """A candidate skill extracted from repeated patterns."""
    skill_id: str
    name: str
    description: str
    original_pattern: CodePattern
    parameterized_code: str
    parameters: List[Dict[str, Any]]
    test_cases: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "original_pattern": self.original_pattern.to_dict(),
            "parameterized_code": self.parameterized_code,
            "parameters": self.parameters,
            "test_cases": self.test_cases,
        }


@dataclass
class ProposedPlugin:
    """A proposed plugin ready for verification and installation."""
    plugin_id: str
    name: str
    version: str
    source_code: str
    candidate: CandidateSkill
    status: ForgeStatus = ForgeStatus.ABSTRACTED
    verification_results: Dict[str, Any] = field(default_factory=dict)
    file_path: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plugin_id": self.plugin_id,
            "name": self.name,
            "version": self.version,
            "source_code": self.source_code,
            "candidate": self.candidate.to_dict(),
            "status": self.status.value,
            "verification_results": self.verification_results,
            "file_path": str(self.file_path) if self.file_path else None,
        }


class PatternMatcher:
    """Analyzes code patterns to find similarities."""

    def __init__(self):
        self.logger = logging.getLogger("PatternMatcher")

    def normalize_code(self, code: str) -> str:
        """Normalize code for comparison (remove whitespace variations, etc.)."""
        # Remove comments
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        # Normalize whitespace
        code = re.sub(r'\s+', ' ', code)
        # Strip
        return code.strip()

    def extract_literals(self, code: str) -> List[Tuple[str, Any]]:
        """Extract literal values that could be parameterized."""
        literals = []

        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant):
                    # Get the literal value
                    literals.append(("constant", node.value))
                elif isinstance(node, ast.Name):
                    # Variable names
                    literals.append(("name", node.id))
        except SyntaxError:
            # Code isn't valid Python, try regex extraction
            # Find string literals
            strings = re.findall(r'["\']([^"\']+)["\']', code)
            for s in strings:
                literals.append(("string", s))
            # Find numbers
            numbers = re.findall(r'\b(\d+\.?\d*)\b', code)
            for n in numbers:
                literals.append(("number", float(n) if '.' in n else int(n)))

        return literals

    def compute_similarity(self, code1: str, code2: str) -> float:
        """Compute similarity score between two code snippets."""
        norm1 = self.normalize_code(code1)
        norm2 = self.normalize_code(code2)

        if not norm1 or not norm2:
            return 0.0

        # Simple token-based Jaccard similarity
        tokens1 = set(norm1.split())
        tokens2 = set(norm2.split())

        if not tokens1 or not tokens2:
            return 0.0

        intersection = tokens1 & tokens2
        union = tokens1 | tokens2

        return len(intersection) / len(union)

    def generate_pattern_id(self, code: str) -> str:
        """Generate a unique ID for a code pattern."""
        normalized = self.normalize_code(code)
        return hashlib.sha256(normalized.encode()).hexdigest()[:12]


class CodeAbstractor:
    """Converts concrete code into parameterized functions."""

    def __init__(self):
        self.logger = logging.getLogger("CodeAbstractor")

    def abstract_pattern(self, pattern: CodePattern) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Convert a concrete code pattern into a parameterized function.

        Returns:
            Tuple of (parameterized_code, parameters_list)
        """
        code = pattern.code
        parameters = []
        param_counter = 0

        # Extract and replace string literals
        def replace_string(match):
            nonlocal param_counter
            param_name = f"param_{param_counter}"
            param_counter += 1
            parameters.append({
                "name": param_name,
                "type": "str",
                "default": match.group(1),
                "description": f"String parameter (originally: {match.group(1)[:20]}...)"
            })
            return f"{{{param_name}}}"

        # Replace string literals with placeholders
        abstracted = re.sub(r'"([^"]+)"', replace_string, code)
        abstracted = re.sub(r"'([^']+)'", replace_string, abstracted)

        # Extract and replace numeric literals (but not in variable names)
        def replace_number(match):
            nonlocal param_counter
            value = match.group(0)
            # Skip if it looks like part of a variable name
            if re.match(r'^\d+$', value):
                param_name = f"num_param_{param_counter}"
                param_counter += 1
                parameters.append({
                    "name": param_name,
                    "type": "float" if '.' in value else "int",
                    "default": float(value) if '.' in value else int(value),
                    "description": f"Numeric parameter (originally: {value})"
                })
                return f"{{{param_name}}}"
            return value

        # Only replace standalone numbers
        abstracted = re.sub(r'\b(\d+\.?\d*)\b', replace_number, abstracted)

        return abstracted, parameters

    def generate_function_code(
        self,
        name: str,
        abstracted_code: str,
        parameters: List[Dict[str, Any]],
        description: str,
    ) -> str:
        """Generate a complete function from abstracted code."""
        # Build parameter signature
        param_signature = ", ".join([
            f"{p['name']}: {p['type']} = {repr(p['default'])}"
            for p in parameters
        ])

        # Build docstring
        param_docs = "\n        ".join([
            f"{p['name']}: {p['description']}"
            for p in parameters
        ])

        function_code = f'''
def {name}({param_signature}) -> Any:
    """
    {description}

    Args:
        {param_docs}

    Returns:
        Result of the operation
    """
    # Parameterized implementation
    result = None
    try:
        {textwrap.indent(abstracted_code, "        ").strip()}
    except Exception as e:
        raise RuntimeError(f"Execution failed: {{e}}")
    return result
'''
        return function_code.strip()


class SyntheticVerifier:
    """Verifies tools using Tier 1 synthetic grounding."""

    def __init__(self, sandbox_enabled: bool = True):
        self.logger = logging.getLogger("SyntheticVerifier")
        self.sandbox_enabled = sandbox_enabled

    def generate_test_cases(
        self,
        candidate: CandidateSkill,
        num_cases: int = 3,
    ) -> List[Dict[str, Any]]:
        """Generate synthetic test cases for a candidate skill."""
        test_cases = []

        for i in range(num_cases):
            test_case = {
                "id": f"test_{i}",
                "inputs": {},
                "expected_behavior": "no_exception",
            }

            # Generate inputs based on parameter types
            for param in candidate.parameters:
                if param["type"] == "str":
                    test_case["inputs"][param["name"]] = f"test_value_{i}"
                elif param["type"] == "int":
                    test_case["inputs"][param["name"]] = i + 1
                elif param["type"] == "float":
                    test_case["inputs"][param["name"]] = float(i + 1)
                else:
                    test_case["inputs"][param["name"]] = param.get("default")

            test_cases.append(test_case)

        return test_cases

    def verify_syntax(self, source_code: str) -> Tuple[bool, str]:
        """Verify that the source code has valid Python syntax."""
        try:
            ast.parse(source_code)
            return True, "Syntax valid"
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

    def verify_execution(
        self,
        source_code: str,
        test_cases: List[Dict[str, Any]],
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify that the code executes without errors.

        Uses a controlled execution environment with necessary builtins.
        """
        results = {
            "total_tests": len(test_cases),
            "passed": 0,
            "failed": 0,
            "errors": [],
        }

        # Create execution namespace with necessary builtins
        # Use full builtins but in a controlled namespace
        import builtins
        from typing import Any, Dict, List, Optional, AsyncIterator, Tuple, Union
        import asyncio
        from datetime import datetime as dt

        safe_builtins = {
            "__name__": "__main__",
            "__doc__": None,
            "__import__": __import__,
            # Typing support
            "Any": Any,
            "Dict": Dict,
            "List": List,
            "Optional": Optional,
            "AsyncIterator": AsyncIterator,
            "Tuple": Tuple,
            "Union": Union,
            # Async support
            "asyncio": asyncio,
            # Datetime support
            "datetime": dt,
            "print": print,
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
            "round": round,
            "sorted": sorted,
            "reversed": reversed,
            "isinstance": isinstance,
            "issubclass": issubclass,
            "hasattr": hasattr,
            "getattr": getattr,
            "setattr": setattr,
            "type": type,
            "object": object,
            "property": property,
            "staticmethod": staticmethod,
            "classmethod": classmethod,
            "super": super,
            "None": None,
            "True": True,
            "False": False,
            "RuntimeError": RuntimeError,
            "Exception": Exception,
            "ValueError": ValueError,
            "TypeError": TypeError,
            "KeyError": KeyError,
            "IndexError": IndexError,
            "AttributeError": AttributeError,
            "ImportError": ImportError,
            "NotImplementedError": NotImplementedError,
            "StopIteration": StopIteration,
            "OSError": OSError,
            "FileNotFoundError": FileNotFoundError,
        }
        exec_globals = {"__builtins__": safe_builtins}
        exec_locals = {}

        # Try to compile and execute
        try:
            exec(source_code, exec_globals, exec_locals)
        except Exception as e:
            results["errors"].append(f"Compilation failed: {e}")
            return False, results

        # Find the function
        func = None
        for name, obj in exec_locals.items():
            if callable(obj) and not name.startswith("_"):
                func = obj
                break

        if func is None:
            results["errors"].append("No callable function found")
            return False, results

        # Run test cases
        for test_case in test_cases:
            try:
                result = func(**test_case["inputs"])
                # Check expected behavior
                if test_case["expected_behavior"] == "no_exception":
                    results["passed"] += 1
                elif test_case.get("expected_output") is not None:
                    if result == test_case["expected_output"]:
                        results["passed"] += 1
                    else:
                        results["failed"] += 1
                        results["errors"].append(
                            f"Test {test_case['id']}: expected {test_case['expected_output']}, got {result}"
                        )
                else:
                    results["passed"] += 1
            except Exception as e:
                results["failed"] += 1
                results["errors"].append(f"Test {test_case['id']} failed: {e}")

        all_passed = results["failed"] == 0 and results["passed"] > 0
        return all_passed, results


class PluginGenerator:
    """Generates PluginABC-compatible plugin code."""

    PLUGIN_TEMPLATE = '''"""
Auto-generated SENSE plugin: {name}
Generated by ToolForge on {timestamp}

{description}
"""

from typing import Any, Dict, List, Optional, AsyncIterator
from datetime import datetime
import asyncio

# Import base class (adjust path as needed)
try:
    from core.plugins.interface import PluginABC, PluginCapability
except ImportError:
    # Fallback for standalone testing
    class PluginABC:
        def __init__(self, config=None):
            self.config = config or {{}}
    class PluginCapability:
        VIRTUAL = "virtual"


class {class_name}(PluginABC):
    """
    {description}

    Auto-generated from repeated successful code patterns.
    Pattern ID: {pattern_id}
    Occurrences: {occurrences}
    Success Rate: {success_rate:.1%}
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._last_result = None

    def get_manifest(self) -> Dict[str, Any]:
        """Return plugin capabilities and metadata."""
        return {{
            "name": "{name}",
            "version": "{version}",
            "type": "virtual",
            "capability": "virtual",
            "description": "{description}",
            "author": "ToolForge",
            "generated": "{timestamp}",
            "pattern_id": "{pattern_id}",
        }}

    async def stream_auxiliary_input(self) -> AsyncIterator[Any]:
        """No streaming input for this plugin."""
        return
        yield  # Make this a generator

    def emergency_stop(self) -> None:
        """No emergency stop needed for virtual plugin."""
        pass

    def safety_policy(self) -> Dict[str, Any]:
        """Return safety constraints."""
        return {{
            "max_execution_time": 30.0,
            "max_memory_mb": 100,
        }}

    def get_grounding_truth(self, query: str) -> Optional[float]:
        """Return ground truth if available."""
        return None

    def execute(self{param_signature}) -> Any:
        """
        Execute the crystallized skill.

        Args:
{param_docs}

        Returns:
            Result of the operation
        """
        result = None
        try:
{execution_code}
            self._last_result = result
        except Exception as e:
            raise RuntimeError(f"Execution failed: {{e}}")
        return result

    async def execute_async(self{param_signature}) -> Any:
        """Async wrapper for execute."""
        return self.execute({param_call})

    @property
    def last_result(self) -> Any:
        """Get the last execution result."""
        return self._last_result
'''

    def __init__(self):
        self.logger = logging.getLogger("PluginGenerator")

    def generate_plugin(self, candidate: CandidateSkill) -> str:
        """Generate a complete PluginABC-compatible plugin."""
        # Sanitize name for class
        class_name = self._to_class_name(candidate.name)

        # Build parameter signature
        if candidate.parameters:
            param_signature = ", " + ", ".join([
                f"{p['name']}: {p['type']} = {repr(p['default'])}"
                for p in candidate.parameters
            ])
            param_call = ", ".join([p['name'] for p in candidate.parameters])
            param_docs = "\n".join([
                f"            {p['name']}: {p['description']}"
                for p in candidate.parameters
            ])
        else:
            param_signature = ""
            param_call = ""
            param_docs = "            None"

        # Indent execution code
        execution_code = textwrap.indent(
            candidate.parameterized_code,
            "            "
        )

        # Format template
        plugin_code = self.PLUGIN_TEMPLATE.format(
            name=candidate.name,
            class_name=class_name,
            description=candidate.description,
            pattern_id=candidate.original_pattern.pattern_id,
            occurrences=candidate.original_pattern.occurrences,
            success_rate=candidate.original_pattern.success_rate,
            version="1.0.0",
            timestamp=datetime.now().isoformat(),
            param_signature=param_signature,
            param_call=param_call,
            param_docs=param_docs,
            execution_code=execution_code,
        )

        return plugin_code

    def _to_class_name(self, name: str) -> str:
        """Convert a name to a valid Python class name."""
        # Remove non-alphanumeric characters
        clean = re.sub(r'[^a-zA-Z0-9]', ' ', name)
        # Convert to PascalCase
        words = clean.split()
        return ''.join(word.capitalize() for word in words) + "Plugin"


class ToolForge:
    """
    Main Tool Forge class - Crystallizes repeated behaviors into tools.

    Pipeline:
    1. DETECT: scan_memory() finds repeated successful code patterns
    2. ABSTRACT: forge_tool() refactors into parameterized functions
    3. VERIFY: verify_tool() runs Tier 1 synthetic tests
    4. PERSIST: install_tool() saves to plugins/user_defined/
    5. REGISTER: hot-loads into the ToolRegistry
    """

    REPETITION_THRESHOLD: int = 3
    SIMILARITY_THRESHOLD: float = 0.7
    SUCCESS_RATE_THRESHOLD: float = 0.8

    def __init__(
        self,
        plugins_dir: Optional[Path] = None,
        tool_registry: Optional[Any] = None,
    ):
        self.logger = logging.getLogger("ToolForge")

        # Set plugins directory
        if plugins_dir is None:
            plugins_dir = Path(__file__).parent.parent.parent / "plugins" / "user_defined"
        self.plugins_dir = plugins_dir

        # Components
        self.pattern_matcher = PatternMatcher()
        self.code_abstractor = CodeAbstractor()
        self.verifier = SyntheticVerifier()
        self.generator = PluginGenerator()

        # Registry for hot-loading
        self.tool_registry = tool_registry

        # Tracking
        self._detected_patterns: Dict[str, CodePattern] = {}
        self._forged_tools: Dict[str, ProposedPlugin] = {}
        self._installed_plugins: List[str] = []

    def scan_memory(
        self,
        memory: Any,
        min_occurrences: Optional[int] = None,
    ) -> List[CandidateSkill]:
        """
        Scan memory for repeated successful code patterns.

        Args:
            memory: AgeMem instance or similar memory interface
            min_occurrences: Minimum occurrences to consider (default: REPETITION_THRESHOLD)

        Returns:
            List of CandidateSkill objects ready for forging
        """
        min_occurrences = min_occurrences or self.REPETITION_THRESHOLD
        candidates = []

        self.logger.info(f"Scanning memory for patterns (threshold: {min_occurrences})")

        # Extract code patterns from memory
        patterns = self._extract_patterns_from_memory(memory)

        # Group similar patterns
        pattern_groups = self._group_similar_patterns(patterns)

        # Filter by occurrence threshold and success rate
        for group_id, pattern_list in pattern_groups.items():
            if len(pattern_list) >= min_occurrences:
                # Calculate aggregate success rate
                success_rates = [p.get("success", True) for p in pattern_list]
                success_rate = sum(success_rates) / len(success_rates)

                if success_rate >= self.SUCCESS_RATE_THRESHOLD:
                    # Create CodePattern
                    representative = pattern_list[0]
                    pattern = CodePattern(
                        pattern_id=group_id,
                        code=representative["code"],
                        occurrences=len(pattern_list),
                        success_rate=success_rate,
                        first_seen=min(p.get("timestamp", datetime.now()) for p in pattern_list),
                        last_seen=max(p.get("timestamp", datetime.now()) for p in pattern_list),
                        context={"sources": [p.get("source") for p in pattern_list]},
                    )

                    self._detected_patterns[group_id] = pattern

                    # Create candidate skill
                    abstracted, params = self.code_abstractor.abstract_pattern(pattern)
                    candidate = CandidateSkill(
                        skill_id=f"skill_{group_id}",
                        name=self._generate_skill_name(pattern),
                        description=f"Auto-extracted skill from {len(pattern_list)} occurrences",
                        original_pattern=pattern,
                        parameterized_code=abstracted,
                        parameters=params,
                    )

                    candidates.append(candidate)
                    self.logger.info(
                        f"Found candidate: {candidate.name} "
                        f"({pattern.occurrences} occurrences, {success_rate:.0%} success)"
                    )

        return candidates

    def _extract_patterns_from_memory(self, memory: Any) -> List[Dict[str, Any]]:
        """Extract code patterns from memory system."""
        patterns = []

        # Try different memory interfaces
        if hasattr(memory, "execution_history"):
            # ReasoningOrchestrator style
            for result in memory.execution_history:
                if hasattr(result, "execution_result"):
                    patterns.append({
                        "code": str(result.execution_result),
                        "success": result.success if hasattr(result, "success") else True,
                        "timestamp": datetime.now(),
                        "source": result.task_id if hasattr(result, "task_id") else "unknown",
                    })

        elif hasattr(memory, "search"):
            # AgeMem style - search for code patterns
            try:
                import asyncio
                results = asyncio.get_event_loop().run_until_complete(
                    memory.search("code execution result", top_k=100)
                )
                for entry in results:
                    if "code" in str(entry).lower():
                        patterns.append({
                            "code": entry.get("content", str(entry)),
                            "success": entry.get("metadata", {}).get("success", True),
                            "timestamp": entry.get("created_at", datetime.now()),
                            "source": entry.get("key", "memory"),
                        })
            except Exception as e:
                self.logger.warning(f"Memory search failed: {e}")

        elif isinstance(memory, list):
            # Direct list of patterns
            for item in memory:
                if isinstance(item, dict) and "code" in item:
                    patterns.append(item)
                elif isinstance(item, str):
                    patterns.append({
                        "code": item,
                        "success": True,
                        "timestamp": datetime.now(),
                        "source": "direct",
                    })

        return patterns

    def _group_similar_patterns(
        self,
        patterns: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group similar code patterns together."""
        groups: Dict[str, List[Dict[str, Any]]] = {}

        for pattern in patterns:
            code = pattern.get("code", "")
            if not code:
                continue

            pattern_id = self.pattern_matcher.generate_pattern_id(code)

            # Check for similar existing groups
            matched_group = None
            for group_id, group_patterns in groups.items():
                if group_patterns:
                    similarity = self.pattern_matcher.compute_similarity(
                        code, group_patterns[0]["code"]
                    )
                    if similarity >= self.SIMILARITY_THRESHOLD:
                        matched_group = group_id
                        break

            if matched_group:
                groups[matched_group].append(pattern)
            else:
                groups[pattern_id] = [pattern]

        return groups

    def _generate_skill_name(self, pattern: CodePattern) -> str:
        """Generate a descriptive name for a skill."""
        code = pattern.code[:100]

        # Try to extract function/operation name
        if "def " in code:
            match = re.search(r'def\s+(\w+)', code)
            if match:
                return match.group(1)

        # Try to extract key operation
        operations = ["calculate", "process", "fetch", "parse", "convert", "check"]
        for op in operations:
            if op in code.lower():
                return f"{op}_operation"

        # Fallback to pattern ID
        return f"skill_{pattern.pattern_id[:8]}"

    def forge_tool(self, candidate: CandidateSkill) -> ProposedPlugin:
        """
        Convert a candidate skill into a proposed plugin.

        Args:
            candidate: CandidateSkill to forge

        Returns:
            ProposedPlugin ready for verification
        """
        self.logger.info(f"Forging tool from candidate: {candidate.name}")

        # Generate plugin code
        plugin_code = self.generator.generate_plugin(candidate)

        # Generate test cases
        test_cases = self.verifier.generate_test_cases(candidate)
        candidate.test_cases = test_cases

        # Create proposed plugin
        plugin = ProposedPlugin(
            plugin_id=f"plugin_{candidate.skill_id}",
            name=candidate.name,
            version="1.0.0",
            source_code=plugin_code,
            candidate=candidate,
            status=ForgeStatus.ABSTRACTED,
        )

        self._forged_tools[plugin.plugin_id] = plugin

        return plugin

    def verify_tool(self, plugin: ProposedPlugin) -> bool:
        """
        Verify a proposed plugin using Tier 1 synthetic tests.

        CRITICAL: Only returns True if 100% of tests pass.

        Args:
            plugin: ProposedPlugin to verify

        Returns:
            True if all tests pass, False otherwise
        """
        self.logger.info(f"Verifying tool: {plugin.name}")

        # Verify syntax
        syntax_ok, syntax_msg = self.verifier.verify_syntax(plugin.source_code)
        plugin.verification_results["syntax"] = {
            "passed": syntax_ok,
            "message": syntax_msg,
        }

        if not syntax_ok:
            plugin.status = ForgeStatus.FAILED
            self.logger.error(f"Syntax verification failed: {syntax_msg}")
            return False

        # Verify execution with test cases
        exec_ok, exec_results = self.verifier.verify_execution(
            plugin.source_code,
            plugin.candidate.test_cases,
        )
        plugin.verification_results["execution"] = exec_results

        if not exec_ok:
            plugin.status = ForgeStatus.FAILED
            self.logger.error(f"Execution verification failed: {exec_results['errors']}")
            return False

        plugin.status = ForgeStatus.VERIFIED
        self.logger.info(
            f"Tool verified: {plugin.name} "
            f"({exec_results['passed']}/{exec_results['total_tests']} tests passed)"
        )

        return True

    def install_tool(self, plugin: ProposedPlugin) -> str:
        """
        Install a verified plugin to the plugins directory and hot-load it.

        Args:
            plugin: Verified ProposedPlugin to install

        Returns:
            Path to the installed plugin file
        """
        if plugin.status != ForgeStatus.VERIFIED:
            raise ValueError(f"Cannot install unverified plugin: {plugin.name}")

        self.logger.info(f"Installing tool: {plugin.name}")

        # Ensure plugins directory exists
        self.plugins_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        filename = f"{plugin.name.lower().replace(' ', '_')}.py"
        file_path = self.plugins_dir / filename

        # Write plugin file
        file_path.write_text(plugin.source_code)
        plugin.file_path = file_path
        plugin.status = ForgeStatus.INSTALLED

        self.logger.info(f"Plugin written to: {file_path}")

        # Hot-load into registry
        self._hot_load_plugin(plugin)

        self._installed_plugins.append(str(file_path))

        return str(file_path)

    def _hot_load_plugin(self, plugin: ProposedPlugin) -> bool:
        """Hot-load a plugin into the registry."""
        if plugin.file_path is None:
            return False

        try:
            # Load the module
            spec = importlib.util.spec_from_file_location(
                plugin.name,
                plugin.file_path
            )
            if spec is None or spec.loader is None:
                return False

            module = importlib.util.module_from_spec(spec)
            sys.modules[plugin.name] = module
            spec.loader.exec_module(module)

            # Find the plugin class
            for name, obj in vars(module).items():
                if (
                    isinstance(obj, type)
                    and name.endswith("Plugin")
                    and name != "PluginABC"
                ):
                    # Register with tool registry if available
                    if self.tool_registry is not None:
                        if hasattr(self.tool_registry, "register"):
                            self.tool_registry.register(obj)
                        elif isinstance(self.tool_registry, dict):
                            self.tool_registry[plugin.name] = obj

                    self.logger.info(f"Hot-loaded plugin: {name}")
                    return True

        except Exception as e:
            self.logger.error(f"Failed to hot-load plugin: {e}")

        return False

    def check_for_crystallization(self, result: Any) -> bool:
        """
        Check if a task result contains patterns worth crystallizing.

        Called by ReasoningOrchestrator after successful task completion.
        """
        # Add to internal pattern tracking
        if hasattr(result, "execution_result"):
            code = str(result.execution_result)
            pattern_id = self.pattern_matcher.generate_pattern_id(code)

            if pattern_id in self._detected_patterns:
                self._detected_patterns[pattern_id].occurrences += 1
                self._detected_patterns[pattern_id].last_seen = datetime.now()

                # Check if threshold reached
                if self._detected_patterns[pattern_id].occurrences >= self.REPETITION_THRESHOLD:
                    self.logger.info(
                        f"Pattern {pattern_id} reached crystallization threshold "
                        f"({self._detected_patterns[pattern_id].occurrences} occurrences)"
                    )
                    return True
            else:
                self._detected_patterns[pattern_id] = CodePattern(
                    pattern_id=pattern_id,
                    code=code,
                    occurrences=1,
                    success_rate=1.0,
                    first_seen=datetime.now(),
                    last_seen=datetime.now(),
                )

        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get Tool Forge statistics."""
        return {
            "detected_patterns": len(self._detected_patterns),
            "forged_tools": len(self._forged_tools),
            "installed_plugins": len(self._installed_plugins),
            "plugins_dir": str(self.plugins_dir),
            "thresholds": {
                "repetition": self.REPETITION_THRESHOLD,
                "similarity": self.SIMILARITY_THRESHOLD,
                "success_rate": self.SUCCESS_RATE_THRESHOLD,
            },
        }


# Factory function
def create_tool_forge(**kwargs) -> ToolForge:
    """Create a configured ToolForge instance."""
    return ToolForge(**kwargs)
