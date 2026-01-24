"""
SENSE-v2 Synthetic Grounding
Tier 1: Deterministic verification via math, code, and logic.

Part of Phase 1: Three-Tier Grounding System

Dependencies: RestrictedPython (as approved in IMPLEMENTATION_STATE.md)
"""

import ast
import math
from typing import Any, Dict, Optional

from restrictedpython import compile_restricted, safe_globals, limited_builtins
from sense_v2.grounding import GroundingResult, GroundingSource  # Local import
from sense_v2.utils.security import RestrictedPython  # Assuming existing security utils if available

# Safe globals for RestrictedPython
SAFE_GLOBALS = safe_globals.copy()
SAFE_GLOBALS.update({
    'math': math,
    'len': len,
    'str': str,
    'int': int,
    'float': float,
    'bool': bool,
    'list': list,
    'dict': dict,
    'tuple': tuple,
    'set': set,
    # Add more as needed, but keep restricted
})

class SyntheticGrounding:
    """
    Tier 1: Synthetic grounding using deterministic computations.
    
    Verifies math, code execution, and basic logic without external dependencies.
    """
    
    def __init__(self, timeout: int = 5):
        self.timeout = timeout
        # Use RestrictedPython for safe eval
        self.safe_builtins = limited_builtins.copy()
        self.safe_builtins.update({
            'abs': abs,
            'max': max,
            'min': min,
            'sum': sum,
            'round': round,
        })
    
    def verify(self, claim: str, context: Dict[str, Any]) -> GroundingResult:
        """
        Verify synthetic claim (math, code, logic).
        
        Args:
            claim: Statement like "2 + 2 = 4" or code snippet.
            context: Dict with 'expression' for math, 'code' for execution, 'expected' for comparison.
            
        Returns:
            GroundingResult with high confidence if verified.
        """
        if 'expression' in context:
            return self._verify_math(context['expression'], context.get('claimed', None))
        
        if 'code' in context:
            return self._verify_code(context['code'], context.get('expected', None))
        
        if 'premises' in context and 'conclusion' in context:
            return self._verify_logic(context['premises'], context['conclusion'])
        
        # Default: Try to parse as math expression
        return self._verify_math(claim, None)
    
    def _verify_math(self, expression: str, claimed: Optional[float]) -> GroundingResult:
        """Verify mathematical expression."""
        try:
            # Safe eval using RestrictedPython
            code = compile_restricted(expression, '<string>', 'eval')
            result = eval(code, SAFE_GLOBALS, self.safe_builtins)
            
            if claimed is not None:
                verified = abs(result - claimed) < 1e-6  # Floating point tolerance
                confidence = 1.0 if verified else 0.0
                evidence = f"Computed: {result}, Claimed: {claimed}, Match: {verified}"
            else:
                verified = True
                confidence = 1.0
                evidence = f"Expression evaluated to: {result}"
            
            return GroundingResult(
                confidence=confidence, source=GroundingSource.SYNTHETIC,
                evidence=evidence, verified=verified
            )
        except Exception as e:
            # FLAG: Potential improvement – More specific error handling for syntax vs runtime
            return GroundingResult(
                confidence=0.0, source=GroundingSource.SYNTHETIC,
                evidence=f"Math verification failed: {str(e)}", verified=False
            )
    
    def _verify_code(self, code: str, expected: Optional[Any]) -> GroundingResult:
        """Verify code execution safely."""
        try:
            compiled = compile_restricted(code, '<string>', 'exec')
            exec(compiled, SAFE_GLOBALS, self.safe_builtins)
            
            # Assume code sets a variable 'result' or returns via print/capture
            # For simplicity, assume context provides 'result_var'
            result_var = context.get('result_var', 'result') if 'context' in locals() else 'result'
            result = SAFE_GLOBALS.get(result_var, None)
            
            if expected is not None:
                verified = result == expected
                confidence = 1.0 if verified else 0.0
                evidence = f"Code output: {result}, Expected: {expected}, Match: {verified}"
            else:
                verified = True
                confidence = 1.0
                evidence = f"Code executed successfully, output: {result}"
            
            return GroundingResult(
                confidence=confidence, source=GroundingSource.SYNTHETIC,
                evidence=evidence, verified=verified
            )
        except Exception as e:
            return GroundingResult(
                confidence=0.0, source=GroundingSource.SYNTHETIC,
                evidence=f"Code verification failed: {str(e)}", verified=False
            )
    
    def _verify_logic(self, premises: List[str], conclusion: str) -> GroundingResult:
        """Basic logic verification (e.g., modus ponens)."""
        # Simple rules-based; expand as needed
        premise_str = ' and '.join(premises).lower()
        conclusion_lower = conclusion.lower()
        
        # Example rules
        if "if p then q" in premise_str and "p" in premise_str and "q" in conclusion_lower:
            verified = True
            confidence = 0.9  # High but not perfect for simple logic
            evidence = "Logic verified via modus ponens."
        else:
            verified = False
            confidence = 0.0
            evidence = "Logic could not be verified with current rules."
        
        # FLAG: Potential improvement – Integrate sympy for formal logic
        return GroundingResult(
            confidence=confidence, source=GroundingSource.SYNTHETIC,
            evidence=evidence, verified=verified
        )
