import json
import os
import time
import logging
import sys

# Law 3: OS-Agnostic Workspace
def get_memory_path():
    """
    Detects the environment and returns a valid writable path.
    """
    # 1. Detect Android/Termux
    if hasattr(sys, 'getandroidapilevel') or os.path.exists('/data/data/com.termux'):
        base = "/sdcard/Download/SENSE_Data"
    # 2. Detect Windows
    elif os.name == 'nt':
        base = os.path.join(os.path.expanduser("~"), "Documents", "SENSE_Data")
    # 3. Detect Linux/Mac
    else:
        base = os.path.join(os.path.expanduser("~"), "SENSE_Data")
    
    return os.path.join(base, "episodic_engrams.json")

MEMORY_FILE = get_memory_path()

class UniversalMemory:
    def __init__(self):
        self.logger = logging.getLogger("UniversalMemory")
        self.engrams = []
        self._load_memory()

    def _load_memory(self):
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, 'r') as f:
                    self.engrams = json.load(f)
                self.logger.info(f"✅ Native Engram Memory loaded ({len(self.engrams)} engrams).")
            except Exception as e:
                self.logger.error(f"❌ Memory Corruption: {e}")
                self.engrams = []
        else:
            self.logger.info(f"🆕 Initializing Memory Cortex at: {MEMORY_FILE}")
            self._ensure_dir()

    def _ensure_dir(self):
        directory = os.path.dirname(MEMORY_FILE)
        if not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
            except:
                pass

    def save_engram(self, content, tags=None):
        if not content: return
        engram = {
            "id": abs(hash(content + str(time.time()))),
            "timestamp": time.time(),
            "content": content,
            "tags": tags or []
        }
        self.engrams.append(engram)
        try:
            self._ensure_dir()
            with open(MEMORY_FILE, 'w') as f:
                json.dump(self.engrams, f, indent=2)
            self.logger.info(f"💾 Memory Saved: '{content[:30]}...'")
        except Exception as e:
            self.logger.error(f"❌ Save Failed: {e}")

    def recall(self, query):
        if not query: return []
        query_words = set(query.lower().split())
        hits = []
        for engram in self.engrams:
            content_lower = engram['content'].lower()
            score = sum(1 for w in query_words if w in content_lower)
            age_hours = (time.time() - engram['timestamp']) / 3600
            if age_hours < 1: score += 0.5
            if score > 0:
                hits.append((score, engram['content']))
        hits.sort(key=lambda x: x[0], reverse=True)
        return [h[1] for h in hits[:3]]
