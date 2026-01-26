import json
import os
import difflib

class GeneticMemory:
    def __init__(self):
        # We store genes in the safe config area (src/sense/data/)
        self.memory_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "genes.json")
        self.genes = self._load_genes()

    def _load_genes(self):
        if not os.path.exists(os.path.dirname(self.memory_path)):
            os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
        
        if not os.path.exists(self.memory_path):
            return []
        
        try:
            with open(self.memory_path, 'r') as f:
                return json.load(f)
        except:
            return []

    def save_gene(self, task, tool_name, tool_input, outcome="success"):
        """
        Evolves the memory. If a strategy works, we save it.
        """
        # Simple structure for now
        gene = {
            "trigger": task.lower(),
            "tool": tool_name,
            "template": tool_input, # We might templatize this later
            "weight": 1.0
        }
        
        # Check if exists (Evolution)
        for existing in self.genes:
            # If we have this exact trigger, reinforce it
            if existing["trigger"] == gene["trigger"]:
                existing["weight"] += 0.1
                self._save_to_disk()
                return
        
        self.genes.append(gene)
        self._save_to_disk()

    def _save_to_disk(self):
        with open(self.memory_path, 'w') as f:
            json.dump(self.genes, f, indent=2)

    def retrieve_instinct(self, task):
        """
        Finds the closest matching gene to guide the agent.
        Uses simple string similarity for speed on mobile.
        """
        task = task.lower()
        best_match = None
        best_score = 0.0
        
        for gene in self.genes:
            # Ratcliff-Obershelp similarity
            score = difflib.SequenceMatcher(None, task, gene["trigger"]).ratio()
            if score > 0.6 and score > best_score: # 60% similarity threshold
                best_score = score
                best_match = gene
        
        if best_match:
            return f"MEMORY HINT: For tasks like '{best_match['trigger']}', you successfully used [{best_match['tool']}(input='{best_match['template']}')] in the past."
        return None
