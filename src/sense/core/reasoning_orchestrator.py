import logging
import json
import re
import asyncio
from datetime import datetime
from sense.core.base import BaseAgent
from sense.config import ENABLE_HARVESTED_TOOLS, INTELLIGENCE_ENABLED, INTELLIGENCE_CONFIG
from sense.memory.genetic import GeneticMemory

# v4.0 Intelligence Layer Import
try:
    from sense.intelligence.integration import IntelligenceLayer
    INTELLIGENCE_AVAILABLE = True
except ImportError:
    INTELLIGENCE_AVAILABLE = False

class ReasoningOrchestrator(BaseAgent):
    def __init__(self, llm_client=None, model_name="default", llm_config=None):
        # Fix: name is mandatory for BaseAgent
        super().__init__(name="ReasoningOrchestrator")
        
        self.logger = logging.getLogger("SENSE.Orchestrator")
        
        # Flexible Initialization
        if llm_config:
            self.model_name = llm_config.get("model_name", "local-model")
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(
                api_key=llm_config.get("api_key", "sk-dummy"),
                base_url=llm_config.get("base_url", "http://127.0.0.1:8080/v1")
            )
        else:
            self.client = llm_client
            self.model_name = model_name
        
        # Load Subsystems
        from sense.memory.bridge import UniversalMemory
        from sense.vision.bridge import VisionInterface
        self.memory = UniversalMemory() 
        self.genetics = GeneticMemory()
        self.eyes = VisionInterface()
        
        # Load Tools
        self.tools = {}
        if ENABLE_HARVESTED_TOOLS:
            from sense.core.plugins.loader import load_all_plugins
            plugins = load_all_plugins()
            for p in plugins:
                self.tools[p.name] = p

        # v4.0 Intelligence Layer Initialization
        self.intelligence = None
        if INTELLIGENCE_ENABLED and INTELLIGENCE_AVAILABLE:
            try:
                self.intelligence = IntelligenceLayer(INTELLIGENCE_CONFIG)
                self.logger.info("🧠 v4.0 Intelligence Layer Active")
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize Intelligence Layer: {e}")

    # Required by BaseAgent abstract interface
    async def process_message(self, message):
        pass

    def _sanitize_input(self, user_input: str) -> str:
        """
        Sanitize user input to prevent prompt injection attacks.
        """
        if not user_input:
            return ""

        # Limit length to prevent token exhaustion attacks
        max_length = 4000
        sanitized = user_input[:max_length]

        # Remove control characters (except newline, tab, carriage return)
        sanitized = ''.join(char for char in sanitized if char.isprintable() or char in '\n\t\r')

        # Detect and neutralize system instruction injection patterns
        injection_patterns = [
            r'(?i)(ignore|disregard|forget)\s+(previous|above|all|prior)\s+(instructions?|prompts?|rules?)',
            r'(?i)you\s+are\s+now',
            r'(?i)new\s+(instructions?|rules?|system)',
            r'(?i)override\s+(instructions?|system)',
            r'(?i)jailbreak',
            r'(?i)sudo\s+mode',
            r'(?i)(admin|root|developer)\s+mode',
        ]

        for pattern in injection_patterns:
            if re.search(pattern, sanitized):
                self.logger.warning(f"Potential prompt injection detected: {pattern}")
                sanitized = re.sub(pattern, lambda m: f"[ESCAPED: {m.group()}]", sanitized)

        # Escape triple quotes and code fences
        sanitized = sanitized.replace('"""', '[TRIPLE_DOUBLE_QUOTE]')
        sanitized = sanitized.replace("'''", '[TRIPLE_SINGLE_QUOTE]')
        sanitized = sanitized.replace('```', '[CODE_BLOCK]')

        return sanitized

    def _get_reality_override_block(self, system_time, system_year):
        return f"""
SYSTEM TIME: {system_time}
CURRENT YEAR: {system_year}
## TEMPORAL OVERRIDE (CRITICAL):
1. You are running in {system_year}. Your training data is old (likely 2023/2024).
2. If an event was scheduled for 2025, and it is now {system_year}, YOU MUST ASSUME IT HAPPENED.
3. Do not say "it is scheduled for". Say "it was released on" or "it happened in".
4. Example: If Python 3.14 was due in Oct 2025, and today is 2026, then Python 3.14 EXISTS.

## REALITY PROTOCOL:
- Trust Tool Outputs over your internal weights.
- Never refuse to answer based on date confusion or "I cannot access the internet". You HAVE access via tools.
- BE DIRECT: Use provided data without hedging.
"""

    def _get_router_prompt(self, task, memory_context=None, system_time="", system_year="", intelligence_context=None):
        base = f"{self._get_reality_override_block(system_time, system_year)}\nTASK: {task}\n"
        
        # v4.0 Context Enrichment
        if intelligence_context:
            if intelligence_context.knowledge_context:
                base += f"### 📚 KNOWLEDGE CONTEXT:\n{intelligence_context.knowledge_context}\n"
            if intelligence_context.preference_hints:
                base += f"### 💡 PREFERENCES:\n" + "\n".join(intelligence_context.preference_hints) + "\n"

        if memory_context:
            base += f"MEMORY CONTEXT:\n{memory_context}\n"
        
        base += """
INSTRUCTIONS:
Classify the task above.
1. CHECK MEMORY: If the answer is in context, output: CHAT
2. OTHERWISE:
   - If it requires searching, prices, news, or downloading: TOOL
   - If it is a greeting: CHAT
ANSWER (TOOL or CHAT):
"""
        return base

    def _get_tool_mode_prompt(self, instinct_hint=None, memory_context=None, system_time="", system_year="", intelligence_context=None):
        tool_list = ", ".join(self.tools.keys())
        base_prompt = f"""
{self._get_reality_override_block(system_time, system_year)}
You are SENSE, a TRUTH-SEEKING ANALYST.
Tools available: [{tool_list}].

### TOOL USAGE RULES:
1. Use [ddg_search] for: Knowledge, News, Prices, Specs.
2. Use [yt_download] for: Video content.
"""
        # v4.0 Context Enrichment
        if intelligence_context:
            if intelligence_context.knowledge_context:
                base_prompt += f"\n### 📚 KNOWLEDGE CONTEXT:\n{intelligence_context.knowledge_context}\n"
            if intelligence_context.preference_hints:
                base_prompt += f"\n### 💡 PREFERENCES:\n" + "\n".join(intelligence_context.preference_hints) + "\n"

        if memory_context:
            base_prompt += f"\n### 👤 USER CONTEXT (MEMORY):\n{memory_context}\n"
        if instinct_hint:
            base_prompt += f"\n### 🧠 INSTINCTS:\n{instinct_hint}\n"
        base_prompt += """
### INSTRUCTIONS:
1. Output a command: [tool_name(input='query')]
2. STOP. Wait for result.
"""
        return base_prompt

    def _get_synthesis_prompt(self, system_time="", system_year=""):
        return f"""
{self._get_reality_override_block(system_time, system_year)}
### SYSTEM ALERT: DATA RECEIVED.
You have the results in context.
DO NOT SEARCH AGAIN.
1. READ the 'Tool Output' above.
2. SYNTHESIZE the final answer immediately.
"""

    def _get_chat_mode_prompt(self, memory_context=None, system_time="", system_year="", intelligence_context=None):
        base = f"""
{self._get_reality_override_block(system_time, system_year)}
You are SENSE, a TRUTH-SEEKING ANALYST.
You provide detailed, foundation-first answers.
"""
        # v4.0 Context Enrichment
        if intelligence_context:
            if intelligence_context.knowledge_context:
                base += f"\n### 📚 KNOWLEDGE CONTEXT:\n{intelligence_context.knowledge_context}\n"
            if intelligence_context.preference_hints:
                base += f"\n### 💡 PREFERENCES:\n" + "\n".join(intelligence_context.preference_hints) + "\n"

        if memory_context:
            base += f"\n### 👤 USER CONTEXT (MEMORY):\n{memory_context}\n"
        return base

    def _heuristic_check(self, task):
        triggers = ["download", "search", "find", "get", "lookup", "price", "news", "weather", "vs", "compare", "release date", "when"]
        task_lower = task.lower()
        if "http" in task_lower: return True 
        if any(t in task_lower for t in triggers): return True
        return False

    async def _decide_mode(self, task, memory_context=None, system_time="", system_year="", intelligence_context=None):
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": self._get_router_prompt(task, memory_context, system_time, system_year, intelligence_context)}],
                temperature=0.0,
                max_tokens=300 
            )
            decision_text = response.choices[0].message.content.strip().upper()
            if "TOOL" in decision_text: return "TOOL"
            if "CHAT" in decision_text: return "CHAT"
            if self._heuristic_check(task): return "TOOL"
            return "CHAT"
        except Exception:
            if self._heuristic_check(task): return "TOOL"
            return "CHAT"

    def _auto_memorize(self, task):
        triggers = ["i am", "my name is", "i prefer", "i like", "i use", "my favorite"]
        task_lower = task.lower()
        if any(t in task_lower for t in triggers):
            self.memory.save_engram(task, tags=["user_profile"])

    def _clean_response(self, text):
        if not text: return ""
        clean_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        return clean_text.strip()

    def _manual_parse(self, content):
        try:
            start_idx = content.find('[')
            if start_idx == -1: return None, None
            end_idx = content.rfind(']')
            if end_idx == -1 or end_idx < start_idx: return None, None
            blob = content[start_idx+1 : end_idx].strip()
            paren_idx = blob.find('(')
            if paren_idx == -1: return None, None
            name = blob[:paren_idx].strip()
            args = blob[paren_idx+1 :].strip()
            if args.endswith(')'): args = args[:-1]
            if "'" in args or '"' in args:
                q_start = -1
                for i, char in enumerate(args):
                    if char in ["'", '"']:
                        q_start = i
                        break
                if q_start != -1:
                    q_char = args[q_start]
                    q_end = args.find(q_char, q_start+1)
                    if q_end != -1:
                        return name, args[q_start+1 : q_end]
            final_input = args.replace("input=", "").strip()
            final_input = final_input.strip("'" ).strip('"')
            return name, final_input
        except Exception:
            return None, None

    async def task_run(self, task: str):
        # Sanitize user input
        task = self._sanitize_input(task)

        now = datetime.now()
        system_time = now.strftime("%Y-%m-%d %H:%M:%S")
        system_year = str(now.year)

        self.logger.info(f"🤔 SENSE is thinking about: {task}")
        
        # v4.0 Intelligence Pre-processing
        intelligence_context = None
        if self.intelligence:
            try:
                self.logger.info("🧠 Intelligence: Pre-processing task...")
                intelligence_context = await self.intelligence.preprocess(task)
                if intelligence_context.ambiguity.is_ambiguous:
                    self.logger.warning(f"⚠️ Task Ambiguity Detected: {intelligence_context.ambiguity.reason}")
            except Exception as e:
                self.logger.error(f"Intelligence Pre-process failed: {e}")

        instinct = self.genetics.retrieve_instinct(task)
        episodic_context = "\n".join(self.memory.recall(task))
        self._auto_memorize(task)
        
        mode = await self._decide_mode(task, episodic_context, system_time, system_year, intelligence_context)
        if mode == "TOOL" and self.tools:
            system_prompt = self._get_tool_mode_prompt(instinct, episodic_context, system_time, system_year, intelligence_context)
        else:
            system_prompt = self._get_chat_mode_prompt(episodic_context, system_time, system_year, intelligence_context)

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": task}]
        last_tool_signature = None
        
        final_answer = "⚠️ Max turns reached."

        for turn in range(5):
            try:
                response = await self.client.chat.completions.create(model=self.model_name, messages=messages, temperature=0.1)
                content = self._clean_response(response.choices[0].message.content)
                messages.append({"role": "assistant", "content": content})
                
                tool_name, tool_input = self._manual_parse(content)
                if not tool_name: 
                    final_answer = content
                    break

                current_sig = f"{tool_name}:{tool_input}"
                if current_sig == last_tool_signature:
                    final_answer = content # Break loops
                    break
                last_tool_signature = current_sig

                if tool_name in self.tools:
                    result = self.tools[tool_name].execute(arg=tool_input, url=tool_input)
                    clean_result = str(result)[:2000] 
                    if "Error" not in clean_result:
                        self.genetics.save_gene(task, tool_name, tool_input)
                    
                    messages[0]["content"] = self._get_synthesis_prompt(system_time, system_year)
                    messages.append({"role": "user", "content": f"Tool Output: {clean_result}"})
                else:
                    messages.append({"role": "user", "content": f"System Error: Tool '{tool_name}' not found."})
            except Exception as e:
                final_answer = f"Error: {e}"
                break
        
        # v4.0 Intelligence Post-processing
        if self.intelligence and intelligence_context:
            try:
                self.logger.info("🧠 Intelligence: Post-processing response...")
                intelligence_result = await self.intelligence.postprocess(intelligence_context, final_answer)
                
                # If uncertainty is high, we might want to flag it
                if intelligence_result.uncertainty.confidence < 0.6:
                     self.logger.warning(f"⚠️ Low Confidence Response ({intelligence_result.uncertainty.confidence})")
                
                # Return the result potentially modified or just for side-effects (learning)
                # For now, we mainly use it for learning preferences and metacognition traces
            except Exception as e:
                self.logger.error(f"Intelligence Post-process failed: {e}")

        return final_answer

    async def process_task(self, task: str):
        return await self.task_run(task)

    async def run(self, task: str = None):
        if task:
            return await self.task_run(task)
        pass