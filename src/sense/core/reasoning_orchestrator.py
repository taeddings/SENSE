import logging
import json
import re
import asyncio
from sense.core.base import BaseAgent
from sense.config import ENABLE_HARVESTED_TOOLS
from sense.memory.genetic import GeneticMemory

class ReasoningOrchestrator(BaseAgent):
    def __init__(self, llm_client=None, model_name="default"):
        # Fix: name is mandatory for BaseAgent
        super().__init__(name="ReasoningOrchestrator")
        self.client = llm_client
        self.model_name = model_name
        self.logger = logging.getLogger("SENSE.Orchestrator")
        
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

    # Required by BaseAgent abstract interface
    async def process_message(self, message):
        pass

    def _get_router_prompt(self, task, memory_context=None):
        """
        ROUTER PROMPT: Now Memory-Aware.
        """
        base = f"TASK: {task}\n"
        if memory_context:
            base += f"MEMORY CONTEXT:\n{memory_context}\n"
            
        base += """
INSTRUCTIONS:
Classify the task above.
1. CHECK MEMORY: If the answer is explicitly in the MEMORY CONTEXT, output: CHAT
2. OTHERWISE:
   - If it requires searching, prices, news, documentation, or downloading: TOOL
   - If it is a greeting or internal logic: CHAT
ANSWER (TOOL or CHAT):
"""
        return base

    def _get_tool_mode_prompt(self, instinct_hint=None, memory_context=None):
        tool_list = ", ".join(self.tools.keys())
        base_prompt = f"""
You are SENSE, a TRUTH-SEEKING ANALYST.
You prioritize Grokipedia.com and raw technical sources.
Tools available: [{tool_list}].

### TOOL USAGE RULES:
1. Use [ddg_search] for: Knowledge, News, Prices, Specs.
   * PRIORITY: Look for [GROKIPEDIA] tags in results.
2. Use [yt_download] for: Video content.
"""
        if memory_context:
            base_prompt += f"\n### 👤 USER CONTEXT (MEMORY):\n{memory_context}\n"
            
        if instinct_hint:
            base_prompt += f"\n### 🧠 INSTINCTS:\n{instinct_hint}\n(Trust this instinct.)\n"
            
        base_prompt += """
### INSTRUCTIONS:
1. Output a command: [tool_name(input='actual_search_term')]
2. STOP. Wait for result.
"""
        return base_prompt

    def _get_synthesis_prompt(self):
        """State-Aware Loop Breaker"""
        return """
### SYSTEM ALERT: DATA RECEIVED.
You have the search results in your context history.
DO NOT SEARCH AGAIN.
1. READ the 'Tool Output' above.
2. SYNTHESIZE the answer immediately.
"""

    def _get_chat_mode_prompt(self, memory_context=None):
        base = """
You are SENSE, a TRUTH-SEEKING ANALYST.
You provide detailed, foundation-first answers.
"""
        if memory_context:
            base += f"\n### 👤 USER CONTEXT (MEMORY):\n{memory_context}\n(Use this context to answer PERSONAL questions.)\n"
        return base

    def _heuristic_check(self, task):
        # REMOVED 'what is' to prevent overriding memory recall
        triggers = ["download", "search", "find", "get", "lookup", "price", "news", "weather", "vs", "compare", "release date", "when"]
        task_lower = task.lower()
        if "http" in task_lower: return True 
        if any(t in task_lower for t in triggers): return True
        return False

    async def _decide_mode(self, task, memory_context=None):
        try:
            # Pass memory_context to the router so it knows what we already know
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": self._get_router_prompt(task, memory_context)}],
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
            self.logger.info("🧠 Personal context detected. Saving engram...")
            self.memory.save_engram(task, tags=["user_profile"])

    def _manual_parse(self, content):
        """Caveman Parser v2 (Stable)"""
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
            final_input = final_input.strip("'").strip('"')
            return name, final_input
        except Exception:
            return None, None

    async def task_run(self, task: str):
        self.logger.info(f"🤔 SENSE is thinking about: {task}")
        
        # 1. RETRIEVE MEMORIES
        instinct = self.genetics.retrieve_instinct(task)
        episodic_context = "\n".join(self.memory.recall(task))
        
        if instinct: self.logger.info(f"🧬 Instinct Triggered: {instinct[:50]}...")
        if episodic_context: self.logger.info(f"🧠 Memory Recalled: {episodic_context[:50]}...")

        # 2. AUTO-SAVE
        self._auto_memorize(task)
        
        # 3. ROUTING (Memory-Aware)
        mode = await self._decide_mode(task, memory_context=episodic_context)
        
        if mode == "TOOL" and self.tools:
            system_prompt = self._get_tool_mode_prompt(instinct_hint=instinct, memory_context=episodic_context)
            self.logger.info("⚙️  Mode: TOOL EXECUTION")
        else:
            system_prompt = self._get_chat_mode_prompt(memory_context=episodic_context)
            self.logger.info("💬 Mode: CONVERSATION")

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": task}]
        last_tool_signature = None
        
        for turn in range(5):
            try:
                response = await self.client.chat.completions.create(model=self.model_name, messages=messages, temperature=0.1)
                content = response.choices[0].message.content
                self.logger.info(f"🗣️  Turn {turn} Output:\n{content[:150]}...") 
                messages.append({"role": "assistant", "content": content})
                
                tool_name, tool_input = self._manual_parse(content)
                
                if not tool_name:
                    return f"✅ FINAL ANSWER:\n{content}"

                current_sig = f"{tool_name}:{tool_input}"
                if current_sig == last_tool_signature:
                    self.logger.warning(f"🛑 Duplicate command detected. Halting loop.")
                    return f"✅ FINAL ANSWER:\n(System halted loop.)"
                last_tool_signature = current_sig

                if tool_name in self.tools:
                    self.logger.info(f"🛠️  Executing: {tool_name} -> {tool_input}")
                    result = self.tools[tool_name].execute(arg=tool_input, url=tool_input)
                    clean_result = str(result)[:2000] 
                    self.logger.info(f"✅ Result: {clean_result[:100]}...")
                    
                    if "Error" not in clean_result:
                        self.genetics.save_gene(task, tool_name, tool_input)
                        self.logger.info("🧬 Gene Saved")

                    messages[0]["content"] = self._get_synthesis_prompt()
                    messages.append({"role": "user", "content": f"Tool Output: {clean_result}"})
                else:
                    messages.append({"role": "user", "content": f"System Error: Tool '{tool_name}' not found."})

            except Exception as e:
                self.logger.error(f"❌ Error: {e}")
                return f"Error: {e}"
        return "⚠️  Max turns reached."

    async def run(self, task: str = None):
        if task:
            return await self.task_run(task)
        # Standard Agent Run Loop
        pass
