import logging
import json
import re
import asyncio
from datetime import datetime
from sense.core.base import BaseAgent
from sense.config import ENABLE_HARVESTED_TOOLS, INTELLIGENCE_ENABLED, INTELLIGENCE_CONFIG
from sense.memory.genetic import GeneticMemory
from sense.core.council import CouncilProtocol

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

    def _formulate_deep_query(self, task):
        """
        Generates a high-fidelity search query by extracting core entities
        and preserving technical context.
        """
        # 1. Clean basic noise
        task_clean = re.sub(r'[^\w\s\-\.]', '', task) # Keep dots/dashes for versions
        
        # 2. Identify Technical Entities (Regex Heuristics)
        # Matches: v1.0, 3.14, 2026-01-01, python_3, etc.
        tech_pattern = r'\b[a-zA-Z0-9]+[_\-.]\[a-zA-Z0-9\.]+\b|\b\d+(\.\d+)+\b'
        tech_terms = re.findall(tech_pattern, task_clean)
        
        # 3. Identify Proper Nouns (Capitalized words in mid-sentence)
        # Simple heuristic: words starting with Upper not at start of sentence
        # (Simplified for this context as task is usually short)
        proper_nouns = re.findall(r'\b[A-Z][a-z0-9]+\b', task) 
        
        # 4. Standard Keyword Extraction (Stoplist)
        stop_words = {
            "search", "find", "check", "verify", "what", "is", "the", "for", "please", 
            "can", "you", "release", "date", "of", "when", "was", "incorrect", "false", 
            "information", "regarding", "have", "must", "and", "save", "this", "to", 
            "your", "knowledge", 
            "base", "about", "tell", "me", "look", "up"
        }
        
        words = task_clean.split()
        keywords = [w for w in words if w.lower() not in stop_words]
        
        # 5. Prioritize Technical Terms & Proper Nouns
        # Merge lists, removing duplicates, keeping order
        final_terms = []
        
        # Add tech terms first (highest information density)
        for term in tech_terms:
            if term not in final_terms: final_terms.append(term)
            
        # Add proper nouns next
        for term in proper_nouns:
            if term not in final_terms: final_terms.append(term)
            
        # Add remaining keywords
        for term in keywords:
            if term not in final_terms: final_terms.append(term)
            
        # 6. Fallback & Construction
        if not final_terms: return task # Safe fail
        
        query = " ".join(final_terms)
        
        # 7. Context Injection (Simple Synonym Expansion)
        # If very short, maybe add domain context?
        if len(query.split()) < 2:
            if "python" in query.lower(): query += " programming language"
            if "linux" in query.lower(): query += " kernel"
            
        return query

    def _get_synthesis_prompt(self, system_time, system_year):
        # Even synthesis uses Council grounding
        return f"""
SYSTEM TIME: {system_time}
CURRENT YEAR: {system_year}
### SYSTEM ALERT: DATA RECEIVED.
You have the results in context.
DO NOT SEARCH AGAIN.
1. READ the 'Tool Output' above.
2. SYNTHESIZE the final answer immediately using the JUDGE persona.
"""

    def _heuristic_check(self, task):
        triggers = ["download", "search", "find", "get", "lookup", "price", "news", "weather", "vs", "compare", "release date", "when"]
        task_lower = task.lower()
        if "http" in task_lower: return True 
        if any(t in task_lower for t in triggers): return True
        return False

    async def _decide_mode(self, task, memory_context=None, intelligence_context=None):
        # Use Council Protocol for the system prompt
        system_prompt = CouncilProtocol.get_system_prompt(context=memory_context)
        system_prompt += "\nTASK: Decide if this requires a TOOL or direct CHAT response. Answer ONLY 'TOOL' or 'CHAT'."
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": task}],
                temperature=0.0,
                max_tokens=10 
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
            final_input = final_input.strip("'").strip('"')
            return name, final_input
        except Exception:
            return None, None

    async def _execute_tool(self, name, args_dict):
        try:
            # Get Tool
            tool_obj = self.tools.get(name)
            if not tool_obj: return f"System Error: Tool '{name}' not found."
            
            # --- THE FIX: UNWRAP ADAPTERS ---
            result = None
            
            # Case A: Adapter Class (has .execute)
            if hasattr(tool_obj, 'execute'):
                if asyncio.iscoroutinefunction(tool_obj.execute):
                    result = await tool_obj.execute(**args_dict)
                else:
                    result = tool_obj.execute(**args_dict)
                    if asyncio.iscoroutine(result):
                        result = await result
            
            # Case B: Callable (Function/Mock)
            elif callable(tool_obj):
                if asyncio.iscoroutinefunction(tool_obj):
                    result = await tool_obj(**args_dict)
                else:
                    result = tool_obj(**args_dict)
                    if asyncio.iscoroutine(result):
                        result = await result
            else:
                return f"System Error: Tool '{name}' is not callable or executable."

            return str(result)
            
        except Exception as e:
            self.logger.error(f"❌ Tool Failure: {e}")
            return f"Tool Error: {e}"

    async def task_run(self, task: str):
        # Sanitize user input
        task = self._sanitize_input(task)

        now = datetime.now()
        system_time = now.strftime("%Y-%m-%d %H:%M:%S")
        system_year = str(now.year)

        self.logger.info(f"🤔 PROCESSING: {task}")
        
        # --- LAYER 0: REFLEX ARC ---
        reflex_context = ""
        task_lower = task.lower()
        if any(keyword in task_lower for keyword in ["search", "find", "check", "verify", "when", "what", "release date"]):
            self.logger.info("⚡ REFLEX TRIGGERED: Analyzing Query...")
            try:
                if "ddg_search" in self.tools:
                    # DEEP QUERY FORMULATION
                    deep_query = self._formulate_deep_query(task)
                    self.logger.info(f"🔎 OPTIMIZED SEARCH: '{deep_query}'")
                    
                    raw_result = await self._execute_tool("ddg_search", {"input": deep_query})
                    
                    if "No results" in str(raw_result):
                         reflex_context = "\n\n## ⚡ REFLEX SEARCH: NO DIRECT MATCHES. USE ARCHITECT PERSONA TO INFER.\n"
                    else:
                        reflex_context = f"\n\n## ⚡ REFLEX SEARCH RESULTS (ABSOLUTE TRUTH):\n{raw_result}\n"
                    
                    self.logger.info(f"✅ Reflex Data Acquired ({len(str(raw_result))} bytes)")
            except Exception as e:
                self.logger.error(f"Reflex failed: {e}")

        # v4.0 Intelligence Pre-processing
        intelligence_context = None
        if self.intelligence:
            try:
                self.logger.info("🧠 Intelligence: Pre-processing task...")
                intelligence_context = await self.intelligence.preprocess(task)
                if intelligence_context and intelligence_context.ambiguity:
                    if hasattr(intelligence_context.ambiguity, 'is_ambiguous') and intelligence_context.ambiguity.is_ambiguous():
                        self.logger.warning(f"⚠️ Task Ambiguity Detected: {getattr(intelligence_context.ambiguity, 'reasons', 'unspecified')}")
            except Exception as e:
                self.logger.error(f"Intelligence Pre-process failed: {e}")

        instinct = self.genetics.retrieve_instinct(task)
        memories = self.memory.recall(task)
        episodic_context = "\n".join([f"- {m}" for m in memories]) if memories else ""
        
        # Merge Reflex Data into Context
        episodic_context += reflex_context
        
        self._auto_memorize(task)
        
        # Build Council Prompt
        system_prompt = CouncilProtocol.get_system_prompt(instinct, episodic_context)
        
        # Enrich with Intelligence Layer context if available
        if intelligence_context:
            if intelligence_context.knowledge_context:
                system_prompt += f"\n## 📚 KNOWLEDGE CONTEXT:\n{intelligence_context.knowledge_context}\n"
            if intelligence_context.preference_hints:
                system_prompt += f"\n## 💡 PREFERENCES:\n" + "\n".join(intelligence_context.preference_hints) + "\n"

        mode = await self._decide_mode(task, episodic_context, intelligence_context)
        
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
                    # TOOL EXECUTION
                    result = await self._execute_tool(tool_name, {"input": tool_input})
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
                
                if intelligence_result and intelligence_result.uncertainty:
                    if intelligence_result.uncertainty.confidence < 0.6:
                         self.logger.warning(f"⚠️ Low Confidence Response ({intelligence_result.uncertainty.confidence})")
            except Exception as e:
                self.logger.error(f"Intelligence Post-process failed: {e}")

        return final_answer

    async def process_task(self, task: str):
        return await self.task_run(task)

    async def run(self, task: str = None):
        if task:
            return await self.task_run(task)
        pass