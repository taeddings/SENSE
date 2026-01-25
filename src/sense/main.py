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
    parser.add_argument("--task", type=str, required=False, help="Single task to execute")
    parser.add_argument("--provider", type=str, help="Override LLM Provider")
    parser.add_argument("--url", type=str, help="Override LLM URL")
    parser.add_argument("--model", type=str, help="Override Model Name")
    parser.add_argument("--key", type=str, help="Override API Key")
    return parser.parse_args()

async def main():
    args = parse_args()
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
    if args.task:
        print(f"🚀 Executing Task: {args.task}")
        try:
            # The 'run' method returns the final string result
            result = await agent.run(args.task)
            
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

if __name__ == "__main__":
    asyncio.run(main())