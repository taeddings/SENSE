import logging
import json
import re
import asyncio
from sense.core.base import BaseAgent
from sense.config import ENABLE_HARVESTED_TOOLS

class ReasoningOrchestrator(BaseAgent):
    def __init__(self, llm_client=None, model_name="default"):
        # BaseAgent requires a name
        super().__init__(name="ReasoningOrchestrator")
        self.client = llm_client
        self.model_name = model_name
        self.logger = logging.getLogger("SENSE.Orchestrator")
        
        # Load Subsystems
        from sense.memory.bridge import UniversalMemory
        from sense.vision.bridge import VisionInterface
        self.memory = UniversalMemory()
        self.eyes = VisionInterface()
        
        # Load Tools
        self.tools = {}
        if ENABLE_HARVESTED_TOOLS:
            from sense.core.plugins.loader import load_all_plugins
            plugins = load_all_plugins()
            for p in plugins:
                self.tools[p.name] = p
        
        # FINAL STABLE REGEX: [tool_name( ... 'input' ... )]
        self.tool_regex = re.compile(r'''\[\s*(\w+)\s*\(.*?['"](.*?)['"].*?\)''', re.IGNORECASE | re.DOTALL)

    # Required by BaseAgent abstract interface
    async def process_message(self, message):
        pass

    def _get_router_prompt(self, task):
        return f"""
TASK: {task}

INSTRUCTIONS:
Classify the task above.
- If it requires downloading files, searching the web, or checking external data, output: TOOL
- If it is a greeting, general knowledge, or coding question that you can answer yourself, output: CHAT

ANSWER (TOOL or CHAT):
"""

    def _get_tool_mode_prompt(self):
        tool_list = ", ".join(self.tools.keys())
        return f"""
You are SENSE in EXECUTION MODE.
You have NO internal knowledge. You MUST use tools to answer.
Tools available: [{tool_list}].

### INSTRUCTIONS:
1. To use a tool, output a command in brackets: [tool_name(input='value')]
2. Wait for the Result.
3. Answer based ONLY on the result.

### EXAMPLES:
User: Download this video https://xyz
Assistant: [yt_download(input='https://xyz')]

User: Search for cats.
Assistant: [google_search(input='cats')]

### YOUR TURN:
"""

    def _get_chat_mode_prompt(self):
        return """

You are SENSE, a helpful AI Assistant.
Answer the user's question concisely and accurately using your internal knowledge.
Do not hallucinate tools.
"""

    def _heuristic_check(self, task):
        """Backup logic for small models"""
        triggers = ["download", "search", "find", "get the title", "use tool", "lookup", "what is the"]
        task_lower = task.lower()
        if "http" in task_lower: return True # URLs usually imply tools
        if any(t in task_lower for t in triggers): return True
        return False

    async def _decide_mode(self, task):
        """Step 1: The Router"""
        try:
            # 1. Ask the Brain
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": self._get_router_prompt(task)}],
                temperature=0.0,
                max_tokens=300 
            )
            decision_text = response.choices[0].message.content.strip().upper()
            
            if "TOOL" in decision_text: return "TOOL"
            if "CHAT" in decision_text: return "CHAT"
            
            # 2. Heuristic Backup
            self.logger.warning(f"⚠️ Ambiguous Router output. Using heuristics.")
            if self._heuristic_check(task): return "TOOL"
            return "CHAT"

        except Exception as e:
            self.logger.warning(f"Router Error ({e}). Using heuristics.")
            if self._heuristic_check(task): return "TOOL"
            return "CHAT"

    async def run(self, task: str):
        self.logger.info(f"🤔 SENSE is thinking about: {task}")
        
        # 1. ROUTING STEP
        mode = await self._decide_mode(task)
        
        # 2. SELECT SYSTEM PROMPT
        if mode == "TOOL" and self.tools:
            system_prompt = self._get_tool_mode_prompt()
            self.logger.info("⚙️  Mode: TOOL EXECUTION")
        else:
            system_prompt = self._get_chat_mode_prompt()
            self.logger.info("💬 Mode: CONVERSATION")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task}
        ]

        # 3. EXECUTION LOOP
        for turn in range(5):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.1
                )
                
                content = response.choices[0].message.content
                self.logger.info(f"🗣️  Turn {turn} Output:\n{content[:150]}...") 
                messages.append({"role": "assistant", "content": content})

                # DETECT TOOL USAGE
                tool_name = None
                tool_input = None
                
                # Safe Search
                bracket_match = self.tool_regex.search(content)
                
                if bracket_match:
                    try:
                        tool_name = bracket_match.group(1).strip()
                        tool_input = bracket_match.group(2).strip()
                    except IndexError:
                        self.logger.error("❌ Regex matched but groups failed.")
                        continue

                if tool_name and tool_name in self.tools:
                    self.logger.info(f"🛠️  Executing: {tool_name} -> {tool_input}")
                    try:
                        # Robust execution call
                        result = self.tools[tool_name].execute(tool_input, arg=tool_input, url=tool_input)
                    except Exception as e:
                        result = f"Error executing tool: {e}"
                    
                    clean_result = str(result)[:1000]
                    self.logger.info(f"✅ Result: {clean_result[:100]}...")
                    messages.append({"role": "user", "content": f"Tool Output: {clean_result}\n\n(Now provide the Final Answer.)"})
                
                elif tool_name:
                    messages.append({"role": "user", "content": f"Error: Tool '{tool_name}' not found."})
                
                else:
                    return f"✅ FINAL ANSWER:\n{content}"

            except Exception as e:
                self.logger.error(f"❌ Error: {e}")
                return f"Error: {e}"

        return "⚠️  Max turns reached."