import json
import os
import sys
import logging

# Law 3: OS-Agnostic Workspace
def get_genetic_path():
    if hasattr(sys, 'getandroidapilevel') or os.path.exists('/data/data/com.termux'):
        base = "/sdcard/Download/SENSE_Data"
    elif os.name == 'nt':
        base = os.path.join(os.path.expanduser("~"), "Documents", "SENSE_Data")
    else:
        base = os.path.join(os.path.expanduser("~"), "SENSE_Data")
    return os.path.join(base, "genetic_db.json")

DB_PATH = get_genetic_path()

class GeneticMemory:
    def __init__(self):
        self.logger = logging.getLogger("GeneticPool")
        self.genes = {} # {task_pattern: {tool: "x", arg: "y", fitness: 1.0}}
        self._load_genes()

    def _load_genes(self):
        if os.path.exists(DB_PATH):
            try:
                with open(DB_PATH, 'r') as f:
                    self.genes = json.load(f)
            except:
                self.genes = {}
        else:
            self._ensure_dir()

    def _ensure_dir(self):
        directory = os.path.dirname(DB_PATH)
        if not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
            except:
                pass

    def retrieve_instinct(self, task):
        """
        Scans the Gene Pool for a high-fitness solution to a similar task.
        """
        task_sig = self._vectorize(task)
        best_gene = None
        best_fitness = 0.0

        for gene_sig, gene_data in self.genes.items():
            similarity = self._similarity(task_sig, gene_sig)
            # Threshold: Must be 60% similar match
            if similarity > 0.6:
                # Evolutionary Selection: Multiply Similarity * Fitness
                score = similarity * gene_data.get('fitness', 1.0)
                if score > best_fitness:
                    best_fitness = score
                    best_gene = gene_data

        if best_gene and best_fitness > 0.8:
            return f"USE TOOL: [{best_gene['tool']}(input='{best_gene['arg']}')] (Fitness: {best_gene.get('fitness'):.2f})"
        return None

    def save_gene(self, task, tool_name, tool_arg):
        """
        Evolutionary Reinforcement:
        - If gene exists: Increase Fitness.
        - If new: Create with base Fitness.
        """
        sig = self._vectorize(task)
        
        if sig in self.genes:
            # Reinforce existing gene
            self.genes[sig]['fitness'] += 0.2
            # Cap fitness at 5.0
            self.genes[sig]['fitness'] = min(self.genes[sig]['fitness'], 5.0)
            self.logger.info(f"🧬 Gene Reinforced (Fitness: {self.genes[sig]['fitness']:.1f})")
        else:
            # Birth new gene
            self.genes[sig] = {
                "tool": tool_name, 
                "arg": tool_arg,
                "fitness": 1.0
            }
            self.logger.info("🧬 New Gene Mutated & Saved.")
        
        # Natural Selection: Prune weak genes occasionally
        if len(self.genes) > 100:
            self._prune_pool()
            
        self._persist()

    def _prune_pool(self):
        """Kills off the bottom 20% of genes by fitness."""
        sorted_genes = sorted(self.genes.items(), key=lambda x: x[1]['fitness'])
        cutoff = int(len(sorted_genes) * 0.2)
        for i in range(cutoff):
            del self.genes[sorted_genes[i][0]]
        self.logger.info(f"☠️ Natural Selection: Pruned {cutoff} weak genes.")

    def _vectorize(self, text):
        """Simple deterministic hash for gene signatures."""
        return " ".join(sorted(text.lower().split()))

    def _similarity(self, sig1, sig2):
        set1 = set(sig1.split())
        set2 = set(sig2.split())
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union else 0.0

    def _persist(self):
        try:
            with open(DB_PATH, 'w') as f:
                json.dump(self.genes, f, indent=2)
        except Exception as e:
            self.logger.error(f"❌ Gene Error: {e}")