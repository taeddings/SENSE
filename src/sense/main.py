import sys
import os
import logging
import argparse
import asyncio
import importlib.util
import socket
from urllib.parse import urlparse

# Add src to path
# __file__ is src/sense/main.py, so parent is src/sense, parent of that is src/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sense.core.reasoning_orchestrator import ReasoningOrchestrator

def is_server_active(url):
    """
    Performs a rapid 'Pulse Check' (TCP Connect) to see if the server is reachable.
    Timeout is set to 0.5 seconds to keep startup snappy.
    """
    try:
        parsed = urlparse(url)
        host = parsed.hostname
        # Handle cases where hostname might be None
        if not host: return False
        port = parsed.port or (443 if parsed.scheme == 'https' else 80)
        
        # Try to open a socket connection
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except:
        return False

def load_intelligent_config():
    """
    Loads config.local.py but performs a Health Check before applying it.
    Fallback: Localhost (Termux).
    """
    # 1. Default (The Safety Net)
    active_config = {
        "provider": "openai_compatible",
        "base_url": "http://127.0.0.1:8080/v1",
        "api_key": "local-model",
        "model_name": "local-mobile",
        "timeout": 60.0
    }

    # 2. Find Override File
    override_path = os.path.expanduser("~/project/SENSE/config.local.py")
    
    if os.path.exists(override_path):
        try:
            # Load the file dynamically
            spec = importlib.util.spec_from_file_location("custom_config", override_path)
            cfg_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cfg_module)
            
            if hasattr(cfg_module, "LLM_SETTINGS"):
                remote_settings = cfg_module.LLM_SETTINGS
                remote_url = remote_settings.get("base_url", "")

                # 3. THE INTELLIGENT HOOK (Pulse Check)
                print(f"📡 Pinging Remote Core at {remote_url}...")
                if is_server_active(remote_url):
                    active_config.update(remote_settings)
                    print(f"✅ CONNECTION SUCCESS: Switched to Remote Core (PC Mode).")
                else:
                    print(f"⚠️  CONNECTION FAILED: Remote Core unreachable.")
                    print(f"🔄 FALLBACK ACTIVE: Reverting to Local Core (Mobile Mode).")
        except Exception as e:
            print(f"❌ CONFIG ERROR: {e}")

    return active_config

def main():
    parser = argparse.ArgumentParser(description="SENSE: Self-Evolving Neural Stabilization Engine")
    parser.add_argument("task", nargs="*", help="The task or query for SENSE")
    parser.add_argument("--reset", action="store_true", help="Clear short-term context")
    args = parser.parse_args()

    # Configure basic logging for CLI
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("root")
    
    print("🤖 SENSE INITIALIZING...")

    # Load Config with Auto-Failover
    llm_config = load_intelligent_config()
    logger.info(f"🧠 LLM CONNECT: {llm_config['base_url']} [{llm_config['model_name']}]")

    try:
        agent = ReasoningOrchestrator(llm_config=llm_config)
        task_query = " ".join(args.task) if args.task else None
        
        if not task_query:
            print("❌ No task provided. Usage: sense 'Your task here'")
            return

        # Run the async task
        result = asyncio.run(agent.process_task(task_query))
        
        print("\n========================================")
        print("   SENSE EXECUTION RESULT")
        print("========================================")
        print(f"📝 {result}")
        print("========================================")

    except Exception as e:
        logger.error(f"CRITICAL FAILURE: {e}", exc_info=True)
        print(f"💥 SYSTEM CRASH: {e}")

if __name__ == "__main__":
    main()