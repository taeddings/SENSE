import json
import os
import sys

# Path to memory
if hasattr(sys, 'getandroidapilevel') or os.path.exists('/data/data/com.termux'):
    base = "/sdcard/Download/SENSE_Data"
elif os.name == 'nt':
    base = os.path.join(os.path.expanduser("~"), "Documents", "SENSE_Data")
else:
    base = os.path.join(os.path.expanduser("~"), "SENSE_Data")
MEMORY_FILE = os.path.join(base, "episodic_engrams.json")

def purge_memory():
    if not os.path.exists(MEMORY_FILE):
        print(f"No memory file found at {MEMORY_FILE}.")
        return

    try:
        with open(MEMORY_FILE, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading memory file: {e}")
        return

    # Filter out the "virus"
    new_data = [
        entry for entry in data 
        if "python 3.14" not in entry.get('content', '').lower()
    ]
    
    removed_count = len(data) - len(new_data)

    try:
        with open(MEMORY_FILE, 'w') as f:
            json.dump(new_data, f, indent=2)
        print(f"✅ BRAIN SURGERY COMPLETE: Removed {removed_count} hallucinated memories about Python 3.14.")
    except Exception as e:
        print(f"Error writing memory file: {e}")

if __name__ == "__main__":
    purge_memory()
