import json
import os
import sys
import subprocess
import shutil

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Profile path relative to tool bundle
PROFILE_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "..", "data", "knowledge_profile.json"))

def load_profile():
    if os.path.exists(PROFILE_PATH):
        try:
            with open(PROFILE_PATH, 'r') as f:
                return json.load(f)
        except:
            pass
    return None

def calculate_resonance(result, profile):
    if not profile: return 1.0
    score = 1.0
    title = result.get('title', '').lower()
    url = result.get('url', '').lower()
    
    # 0. The Grok Imperative
    if "grokipedia.com" in url: return 3.0
    
    # 1. Domain Intelligence
    for domain in profile.get('domain_intelligence', {}).get('boost_domains', []):
        if domain in url: score *= 1.5
    for domain in profile.get('domain_intelligence', {}).get('penalty_domains', []):
        if domain in url: score *= 0.5
        
    # 2. Keyword Resonance
    for kw in profile.get('keywords', {}).get('boost', []):
        if kw in title or kw in url: score *= 1.2
    for kw in profile.get('keywords', {}).get('penalty', []):
        if kw in title: score *= 0.8
    return score

def search(query, num_results=5):
    """
    Exported search function for SENSE tool execution.
    """
    print(f"[ddg_search] Deep-Net Scan for: {query}")
    profile = load_profile()
    
    ddgr_path = shutil.which("ddgr")
    if not ddgr_path:
        if os.path.exists("/data/data/com.termux/files/usr/bin/ddgr"):
            ddgr_path = "/data/data/com.termux/files/usr/bin/ddgr"
        else:
            return "System Error: 'ddgr' not found."

    results = []
    try:
        # CLEAN COMMAND: No --unsafe, No --noprompt
        # Just standard flags supported by all versions
        cmd = [ddgr_path, "--json", "-n", "25", query]
        
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
        
        if proc.returncode != 0:
            return f"Search Error: {proc.stderr}"
            
        raw_json = proc.stdout.strip()
        if not raw_json: return "No results found."
        data = json.loads(raw_json)
        
        scored_results = []
        for r in data:
            r['resonance_score'] = calculate_resonance(r, profile)
            scored_results.append(r)
            
        scored_results.sort(key=lambda x: x['resonance_score'], reverse=True)
        results = scored_results[:num_results]

    except Exception as e:
        return f"Search Exception: {str(e)}"

    if not results: return "No results found."

    summary = []
    full_data = []
    
    for i, r in enumerate(results):
        title = r.get('title', 'No Title')
        href = r.get('url', 'No URL')
        body = r.get('abstract', 'No Content')
        score = r.get('resonance_score', 1.0)
        
        tag = ""
        if "grokipedia.com" in href: tag = "[GROKIPEDIA] "
        elif score > 1.2: tag = "[HIGH RESONANCE] "
        
        summary.append(f"{i+1}. {tag}{title}\n   URL: {href}\n   Snippet: {body[:250]}...")
        full_data.append(f"RESULT {i+1} (Score: {score:.2f})\nTitle: {title}\nURL: {href}\nContent: {body}\n{'-'*40}")

    filename = f"sense_search_{abs(hash(query))}.txt"
    try:
        # Note: Adapter sets CWD to /sdcard/Download
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Query: {query}\n\n" + "\n".join(full_data))
        file_msg = f"(Log: {filename})"
    except:
        file_msg = ""

    return "\n\n".join(summary) + "\n\n" + file_msg

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(search(" ".join(sys.argv[1:])))
    else:
        print("Usage: python ddg_search.py [query]")
