import argparse
import asyncio
import logging
import sys

# 1. Import ONLY what exists
from sense.core.reasoning_orchestrator import ReasoningOrchestrator
from sense.llm.factory import LLMFactory

# Configure Logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

def parse_args():
    parser = argparse.ArgumentParser(description="SENSE: Universal Agent")
    parser.add_argument("task", nargs="*", help="The task to execute (optional, can be passed as positional args)")
    parser.add_argument("--task", dest="task_flag", type=str, required=False, help="Single task to execute (flag)")
    parser.add_argument("--provider", type=str, help="Override LLM Provider")
    parser.add_argument("--url", type=str, help="Override LLM URL")
    parser.add_argument("--model", type=str, help="Override Model Name")
    parser.add_argument("--key", type=str, help="Override API Key")
    return parser.parse_args()

async def async_main():
    args = parse_args()
    
    # Handle task input from positional args OR flag
    task_input = args.task_flag
    if not task_input and args.task:
        task_input = " ".join(args.task)
        
    print("🤖 SENSE INITIALIZING...")

    # 2. Initialize LLM Client
    llm_client = LLMFactory.create_client(
        cli_provider=args.provider,
        cli_url=args.url,
        cli_model=args.model,
        cli_key=args.key
    )
    
    model_name = LLMFactory.get_model_name(args.model)

    if not llm_client:
        print("❌ Fatal: Could not initialize LLM Brain.")
        return

    # 3. Inject Client into Orchestrator
    agent = ReasoningOrchestrator(llm_client=llm_client, model_name=model_name)

    # 4. Run Task or Standby
    if task_input:
        print(f"🚀 Executing Task: {task_input}")
        try:
            # The 'run' method returns the final string result
            result = await agent.run(task_input)
            
            print("\n" + "="*40)
            print("   SENSE EXECUTION RESULT")
            print("="*40)
            print(f"📝 {result}")
            print("="*40)
            
        except Exception as e:
            print(f"❌ Execution Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("🤖 SENSE Standby Mode (Ready for API/Dashboard)")

def main():
    """Synchronous entry point for console_scripts."""
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\n👋 SENSE Stopped by User")

if __name__ == "__main__":
    main()
