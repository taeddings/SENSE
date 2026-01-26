import json
import os
import time
import logging
import sys
import math
import re

# Law 3: OS-Agnostic Workspace
def get_memory_path():
    if hasattr(sys, 'getandroidapilevel') or os.path.exists('/data/data/com.termux'):
        base = "/sdcard/Download/SENSE_Data"
    elif os.name == 'nt':
        base = os.path.join(os.path.expanduser("~"), "Documents", "SENSE_Data")
    else:
        base = os.path.join(os.path.expanduser("~"), "SENSE_Data")
    return os.path.join(base, "episodic_engrams.json")

MEMORY_FILE = get_memory_path()

# STOP WORDS LIST (Common words to ignore during resonance check)
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he",
    "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "were",
    "will", "with", "you", "your", "me", "my", "i", "this", "what", "how", "why"
}

class UniversalMemory:
    def __init__(self):
        self.logger = logging.getLogger("EpisodicCortex")
        self.engrams = []
        self._load_memory()

    def _load_memory(self):
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, 'r') as f:
                    self.engrams = json.load(f)
            except:
                self.engrams = []
        else:
            self._ensure_dir()

    def _ensure_dir(self):
        directory = os.path.dirname(MEMORY_FILE)
        if not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
            except:
                pass

    def _calculate_retention(self, engram):
        elapsed_hours = (time.time() - engram.get('last_accessed', time.time())) / 3600
        decay_rate = 0.005 / max(engram.get('strength', 1.0), 0.1)
        current_strength = engram.get('strength', 1.0) * math.exp(-decay_rate * elapsed_hours)
        return current_strength

    def _extract_keywords(self, text):
        """Extracts significant keywords by removing stop words."""
        if not text: return set()
        tokens = re.findall(r'\b\w+\b', text.lower())
        keywords = {t for t in tokens if t not in STOP_WORDS and len(t) > 2}
        return keywords

    def save_engram(self, content, tags=None):
        if not content: return
        
        for e in self.engrams:
            if content.lower() in e['content'].lower() or e['content'].lower() in content.lower():
                e['strength'] = e.get('strength', 1.0) + 0.5
                e['last_accessed'] = time.time()
                self._persist()
                return

        engram = {
            "id": abs(hash(content + str(time.time()))),
            "created_at": time.time(),
            "last_accessed": time.time(),
            "content": content,
            "tags": tags or [],
            "strength": 1.0
        }
        self.engrams.append(engram)
        self.logger.info(f"💾 Engram Created: '{content[:30]}...'")
        self._persist()

    def recall(self, query):
        if not query: return []
        
        # 1. Extract ONLY significant words from query
        query_keywords = self._extract_keywords(query)
        if not query_keywords: return []

        hits = []
        for engram in self.engrams:
            retention = self._calculate_retention(engram)
            engram_keywords = self._extract_keywords(engram['content'])
            
            # Intersection Match
            match_score = len(query_keywords.intersection(engram_keywords))
            
            if match_score > 0:
                engram['strength'] = engram.get('strength', 1.0) + 0.1
                engram['last_accessed'] = time.time()
                final_score = match_score * retention
                hits.append((final_score, engram['content']))
        
        self._persist()
        hits.sort(key=lambda x: x[0], reverse=True)
        return [h[1] for h in hits[:2]]

    def _persist(self):
        try:
            with open(MEMORY_FILE, 'w') as f:
                json.dump(self.engrams, f, indent=2)
        except Exception as e:
            self.logger.error(f"❌ Cortex Error: {e}")