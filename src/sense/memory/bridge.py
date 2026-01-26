import json
import os
import time
import logging
import sys
import math

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
        """
        The Ebbinghaus Forgetting Curve implementation.
        Strength = Initial_Strength * e^(-Decay * Time_Elapsed)
        """
        elapsed_hours = (time.time() - engram['last_accessed']) / 3600
        # Base decay is slow (0.001), faster if strength is low
        decay_rate = 0.005 / max(engram.get('strength', 1.0), 0.1)
        
        current_strength = engram.get('strength', 1.0) * math.exp(-decay_rate * elapsed_hours)
        return current_strength

    def save_engram(self, content, tags=None):
        """Creates a fresh Neural Engram."""
        if not content: return
        
        # Check for duplicate concepts to reinforce instead of duplicate
        for e in self.engrams:
            if content.lower() in e['content'].lower() or e['content'].lower() in content.lower():
                e['strength'] = e.get('strength', 1.0) + 0.5  # Reinforce existing
                e['last_accessed'] = time.time()
                self._persist()
                return

        engram = {
            "id": abs(hash(content + str(time.time()))),
            "created_at": time.time(),
            "last_accessed": time.time(),
            "content": content,
            "tags": tags or [],
            "strength": 1.0  # Initial neural strength
        }
        self.engrams.append(engram)
        self.logger.info(f"💾 Engram Created: '{content[:30]}...' (Strength: 1.0)")
        self._persist()

    def recall(self, query):
        """
        Retrieves memories based on Resonance (Keyword Match * Retention Strength).
        """
        if not query: return []
        query_words = set(query.lower().split())
        hits = []
        
        for engram in self.engrams:
            # 1. Neuroplasticity Check
            retention = self._calculate_retention(engram)
            
            # 2. Resonance Check
            content_lower = engram['content'].lower()
            match_score = sum(1 for w in query_words if w in content_lower)
            
            if match_score > 0:
                # Reinforce this memory because it was useful!
                engram['strength'] = engram.get('strength', 1.0) + 0.1
                engram['last_accessed'] = time.time()
                
                final_score = match_score * retention
                hits.append((final_score, engram['content']))
        
        self._persist() # Save the reinforcement updates
        
        hits.sort(key=lambda x: x[0], reverse=True)
        return [h[1] for h in hits[:3]]

    def _persist(self):
        try:
            with open(MEMORY_FILE, 'w') as f:
                json.dump(self.engrams, f, indent=2)
        except Exception as e:
            self.logger.error(f"❌ Cortex Error: {e}")