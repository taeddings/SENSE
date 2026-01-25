import logging
import time
from sense.config import MEMORY_BACKEND
from sense.core.config import EngramConfig

class UniversalMemory:
    def __init__(self):
        self.backend = None
        self.mode = "NATIVE"
        self.logger = logging.getLogger("UniversalMemory")

        if MEMORY_BACKEND == "transplant_agent_zero":
            try:
                # Attempt to load the Heavy Transplant
                from sense.memory.transplant import memory as az_memory
                self.backend = az_memory
                self.mode = "AGENT_ZERO_FAISS"
                self.logger.info("✅ Agent Zero Memory (FAISS) loaded successfully.")
            except ImportError as e:
                self.logger.warning(f"⚠️ Heavy Memory Load Failed ({e}). Falling back to Native Engram.")
                self.mode = "NATIVE_FALLBACK"
            except Exception as e:
                self.logger.warning(f"⚠️ Heavy Memory Error ({e}). Falling back to Native Engram.")
                self.mode = "NATIVE_FALLBACK"

        if self.mode == "NATIVE" or self.mode == "NATIVE_FALLBACK":
            # Load SENSE's original lightweight memory
            from sense.engram.manager import EngramMemoryManager
            self.backend = EngramMemoryManager(EngramConfig())
            self.logger.info("✅ Native Engram Memory loaded.")
    
    def add(self, content: str):
        if self.mode == "AGENT_ZERO_FAISS":
            # Mapping to AZ method (Partial implementation, requires async context)
            # In a real scenario, we'd need to properly instantiate the Memory class with context
            self.logger.info(f"AZ Memory Add: {content[:20]}...")
            return True
        else:
            # Mapping to SENSE method
            entry = {
                "content": content,
                "age": 0,
                "relevance": 1.0,
                "timestamp": time.time()
            }
            return self.backend.store_memory(entry)

    def query(self, text: str):
        if self.mode == "AGENT_ZERO_FAISS":
            # Mapping to AZ method
            self.logger.info(f"AZ Memory Query: {text}")
            return []
        else:
            return self.backend.retrieve_memories(text)
