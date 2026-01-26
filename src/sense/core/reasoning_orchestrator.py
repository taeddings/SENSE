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
        if "http" in task_lower: return True 
        if any(t in task_lower for t in triggers): return True
        return False

    async def _decide_mode(self, task):
        try:
            # 1. Router Call
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": self._get_router_prompt(task)}],
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

    def _manual_parse(self, content):
        """
        CAVEMAN PARSER: No Regex. Just string slicing.
        Returns: (tool_name, tool_input) or (None, None)
        """
        try:
            # 1. Find start bracket
            start_idx = content.find('[')
            if start_idx == -1: return None, None
            
            # 2. Find end bracket (look from the end to catch nested stuff if any)
            end_idx = content.rfind(']')
            if end_idx == -1 or end_idx < start_idx: return None, None
            
            # 3. Extract the blob: "yt_download(input='...')"
            blob = content[start_idx+1 : end_idx].strip()
            
            # 4. Split name and args by first parenthesis
            paren_idx = blob.find('(')
            if paren_idx == -1: return None, None
            
            name = blob[:paren_idx].strip()
            args = blob[paren_idx+1 :].strip()
            
            # 5. Remove trailing parenthesis ')' if present
            if args.endswith(')'):
                args = args[:-1]
                
            # 6. Clean Args (Remove input=, quotes)
            # We just want the value. 
            if "'" in args or '"' in args:
                # Find first quote
                q_start = -1
                for i, char in enumerate(args):
                    if char in ["'", '"']:
                        q_start = i
                        break
                if q_start != -1:
                    # Find matching closing quote
                    q_char = args[q_start]
                    q_end = args.find(q_char, q_start+1)
                    if q_end != -1:
                        final_input = args[q_start+1 : q_end]
                        return name, final_input

            # Fallback: just return raw args stripped
            final_input = args.replace("input=", "").strip()
            return name, final_input

        except Exception:
            return None, None

    async def run(self, task: str):
        self.logger.info(f"🤔 SENSE is thinking about: {task}")
        
        mode = await self._decide_mode(task)
        
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

        last_tool_signature = None

        for turn in range(5):
            try:
                # 1. GENERATE
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.1
                )
                
                content = response.choices[0].message.content
                self.logger.info(f"🗣️  Turn {turn} Output:\n{content[:150]}...") 
                messages.append({"role": "assistant", "content": content})

                # 2. PARSE (Caveman Style)
                tool_name, tool_input = self._manual_parse(content)
                
                # 3. EVALUATE LOGIC
                
                # CASE A: No tool found -> We are done.
                if not tool_name:
                    return f"✅ FINAL ANSWER:\n{content}"

                # CASE B: Duplicate Tool -> We are done (Loop protection).
                current_sig = f"{tool_name}:{tool_input}"
                if current_sig == last_tool_signature:
                    self.logger.warning(f"🛑 Duplicate command detected. Halting loop.")
                    return f"✅ FINAL ANSWER:\n{content}"
                last_tool_signature = current_sig

                # CASE C: Tool Execution
                if tool_name in self.tools:
                    self.logger.info(f"🛠️  Executing: {tool_name} -> {tool_input}")
                    
                    # Execute
                    try:
                        result = self.tools[tool_name].execute(arg=tool_input, url=tool_input)
                    except Exception as e:
                        result = f"Error: {e}"
                    
                    # Log & truncate
                    clean_result = str(result)[:2000] 
                    self.logger.info(f"✅ Result: {clean_result[:100]}...")
                    
                    messages.append({
                        "role": "user", 
                        "content": f"Tool Output: {clean_result}\n\n(Analyze this result. If complete, summarize. If not, use another tool.)"
                    })
                
                # CASE D: Tool Not Found
                else:
                    err_msg = f"System Error: Tool '{tool_name}' not found. Available tools: {list(self.tools.keys())}"
                    self.logger.warning(err_msg)
                    messages.append({"role": "user", "content": err_msg})

            except Exception as e:
                self.logger.error(f"❌ Error: {e}")
                return f"Error: {e}"

        return "⚠️  Max turns reached."