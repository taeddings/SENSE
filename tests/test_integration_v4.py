import asyncio
import unittest
from unittest.mock import MagicMock, AsyncMock
from sense.core.reasoning_orchestrator import ReasoningOrchestrator
from sense.intelligence.integration import IntelligenceLayer, IntelligenceContext, IntelligenceResult
from sense.intelligence.uncertainty import UncertaintyScore, ConfidenceLevel, AmbiguityScore

class TestReasoningOrchestratorV4Integration(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Mock LLM Client
        self.mock_llm_client = MagicMock()
        self.mock_llm_client.chat.completions.create = AsyncMock(return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content="Mock LLM Response"))]
        ))

        # Mock Intelligence Layer
        self.mock_intelligence = MagicMock(spec=IntelligenceLayer)
        self.mock_intelligence.preprocess = AsyncMock(return_value=IntelligenceContext(
            original_task="test task",
            knowledge_context="Mock Knowledge",
            preference_hints=["Mock Hint"],
            ambiguity=AmbiguityScore(score=0.0, reasons=[])
        ))
        self.mock_intelligence.postprocess = AsyncMock(return_value=IntelligenceResult(
            response="Mock LLM Response",
            uncertainty=UncertaintyScore(confidence=0.9, level=ConfidenceLevel.HIGH)
        ))

        # Initialize Orchestrator with mocks
        self.orchestrator = ReasoningOrchestrator(llm_client=self.mock_llm_client)
        self.orchestrator.intelligence = self.mock_intelligence
        
        # Mock Tools
        self.orchestrator.tools = {"ddg_search": AsyncMock(return_value="Mock Search Result")}

    async def test_process_task_flow(self):
        """Verify the full v4 execution flow."""
        task = "search for python 3.14 release date"
        
        # Run task
        result = await self.orchestrator.process_task(task)
        
        # 1. Verify Intelligence Pre-process
        self.mock_intelligence.preprocess.assert_called_once()
        print("✅ Intelligence Pre-process called")

        # 2. Verify Reflex Arc (Search Trigger)
        # The orchestrator should have called ddg_search because "search" is in the task
        self.orchestrator.tools["ddg_search"].assert_called()
        print("✅ Reflex Arc Search triggered")

        # 3. Verify LLM Call (Council Prompt)
        self.mock_llm_client.chat.completions.create.assert_called()
        call_args = self.mock_llm_client.chat.completions.create.call_args
        prompt_content = call_args[1]['messages'][0]['content']
        
        self.assertIn("COUNCIL OF THOUGHT", prompt_content)
        self.assertIn("Mock Knowledge", prompt_content) # From Intelligence Layer
        self.assertIn("Mock Search Result", prompt_content) # From Reflex Arc (injected into context)
        print("✅ Council Prompt & Context Injection verified")

        # 4. Verify Intelligence Post-process
        self.mock_intelligence.postprocess.assert_called_once()
        print("✅ Intelligence Post-process called")
        
        print("🎉 Integration Test Passed!")

if __name__ == "__main__":
    unittest.main()
