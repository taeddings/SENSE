"""SENSE v3.0 Comprehensive Dashboard - Main Entry Point"""

import os
import sys
import streamlit as st
import asyncio
import time
import threading
import logging
from io import StringIO

# Handle both script and module execution
if __name__ == "__main__" or not __package__:
    # Running as script - add parent to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from sense.core.reasoning_orchestrator import ReasoningOrchestrator
    from sense.core.evolution.curriculum import CurriculumAgent
    from sense.core.evolution.grpo import GRPOTrainer
    from sense.core.memory.ltm import AgeMem
    from sense.core.plugins.forge import ToolForge
    from sense.bridge import Bridge, EmergencyStop
    from sense.llm.model_backend import get_model
    from sense_v2.core.config import Config
else:
    # Running as module
    from .core.reasoning_orchestrator import ReasoningOrchestrator
    from .core.evolution.curriculum import CurriculumAgent
    from .core.evolution.grpo import GRPOTrainer
    from .core.memory.ltm import AgeMem
    from .core.plugins.forge import ToolForge
    from .bridge import Bridge, EmergencyStop
    from .llm.model_backend import get_model
    from sense_v2.core.config import Config

# Global state
if 'evolution_running' not in st.session_state:
    st.session_state.evolution_running = False
if 'evolution_thread' not in st.session_state:
    st.session_state.evolution_thread = None
if 'logs' not in st.session_state:
    st.session_state.logs = StringIO()
if 'config' not in st.session_state:
    st.session_state.config = {}
if 'stats' not in st.session_state:
    st.session_state.stats = {}

# Setup logging to capture in UI
log_handler = logging.StreamHandler(st.session_state.logs)
logging.getLogger().addHandler(log_handler)
logging.getLogger().setLevel(logging.INFO)

st.set_page_config(page_title="SENSE v3.0 Dashboard", page_icon="🧠", layout="wide")

st.title("🧠 SENSE v3.0 - Intelligence Amplification System")
st.markdown("**Model-Agnostic AI Evolution Platform** | Reflexion Loop + Grounding + Self-Evolution")

# Sidebar: Configuration & Controls
st.sidebar.header("⚙️ Configuration & Controls")

# Model Selection
model_options = ["transformers/gpt2", "openai/gpt-4", "anthropic/claude-3-sonnet-20240229", "ollama/llama3", "lmstudio/gpt-4"]
selected_model = st.sidebar.selectbox("LLM Backend", model_options, index=0)
st.session_state.config['model_name'] = selected_model

# Evolution Settings
st.sidebar.subheader("Evolution Settings")
population_size = st.sidebar.slider("Population Size", 4, 64, 16)
grpo_group_size = st.sidebar.slider("GRPO Group Size", 2, 16, 8)
curriculum_stages = st.sidebar.slider("Curriculum Stages", 5, 50, 10)
st.session_state.config.update({
    'population_size': population_size,
    'grpo_group_size': grpo_group_size,
    'curriculum_stages': curriculum_stages
})

# Emergency Controls
if st.sidebar.button("🚨 Emergency Stop"):
    EmergencyStop.stop()
    st.sidebar.error("Emergency Stop Activated")
if st.sidebar.button("🔄 Reset Emergency"):
    EmergencyStop.reset()
    st.sidebar.success("Emergency Reset")

# Main Tabs
tabs = st.tabs(["🏠 Home", "🤖 Orchestrator", "📚 Curriculum", "🧠 Memory", "🔧 ToolForge", "📊 Evolution", "⚡ Bridge", "📝 Logs", "📈 Stats"])

# Home Tab
with tabs[0]:
    st.header("Welcome to SENSE v3.0")
    st.markdown("""
    **Features:**
    - Reflexion Loop: Architect → Worker → Critic
    - Multi-LLM Support: OpenAI, Anthropic, Ollama, etc.
    - Grounding: Synthetic + Real-World + Experiential
    - Self-Evolution: Curriculum + GRPO Training
    - Safe Agency: Whitelisted OS Interactions
    - Full Deployment: Docker + API + UI

    **Quick Start:**
    1. Configure LLM in sidebar.
    2. Run tasks in Orchestrator tab.
    3. Start evolution in Evolution tab.
    """)
    st.info("All components are modular and pluggable. See docs for advanced usage.")

# Orchestrator Tab
with tabs[1]:
    task = st.text_input("Task", "Calculate 15 * 3")
    if st.button("Solve"):
        orch = ReasoningOrchestrator()
        result = asyncio.run(orch.solve_task(task))
        st.write(f"Success: {result.success}")
        st.write(f"Confidence: {result.verification.confidence}")
        st.json(result.to_dict())

with tabs[2]:
    config = {}
    agent = CurriculumAgent(config)
    if st.button("Generate Next Task"):
        task = asyncio.run(agent.generate_task())
        st.write(task)
        agent.advance_stage()

with tabs[3]:
    config = {}
    mem = AgeMem(config)
    if st.button("Add Memory"):
        mem.add_memory("test", "plan", "result", True)
        st.success("Added")
    if st.button("Retrieve"):
        similar = asyncio.run(mem.retrieve_similar("test"))
        st.json(similar)

with tabs[4]:
    st.header("🔧 Tool Forge - Dynamic Tool Creation")
    forge = ToolForge()

    st.subheader("Scan Memory for Patterns")
    if st.button("Scan Memory"):
        candidates = forge.scan_memory([])
        st.write(f"Found {len(candidates)} candidates")
        for candidate in candidates:
            with st.expander(f"Candidate: {candidate.name}"):
                st.write(f"**Occurrences:** {candidate.original_pattern.occurrences}")
                st.write(f"**Success Rate:** {candidate.original_pattern.success_rate:.1%}")
                st.code(candidate.parameterized_code, language="python")

    st.subheader("Forge Statistics")
    stats = forge.get_stats()
    col1, col2, col3 = st.columns(3)
    col1.metric("Detected Patterns", stats.get("detected_patterns", 0))
    col2.metric("Forged Tools", stats.get("forged_tools", 0))
    col3.metric("Installed Plugins", stats.get("installed_plugins", 0))

# Evolution Tab
with tabs[5]:
    st.header("📊 Evolution - GRPO Training")

    st.subheader("Population Management")
    try:
        grpo = GRPOTrainer(st.session_state.config)
        population = grpo.population_manager

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Population Size", len(population.population) if population.population else 0)
        with col2:
            if population.population:
                best = population.get_best_genome()
                st.metric("Best Fitness", f"{best.fitness:.2f}" if best else "N/A")

        # Evolution Controls
        st.subheader("Evolution Controls")
        generations = st.number_input("Generations to Evolve", min_value=1, max_value=100, value=5)

        if st.button("Run Evolution"):
            with st.spinner(f"Evolving {generations} generations..."):
                try:
                    for gen in range(generations):
                        population.evolve(generations=1)
                        progress = (gen + 1) / generations
                        st.progress(progress)
                    st.success(f"Evolution complete! {generations} generations evolved.")

                    # Show results
                    if population.population:
                        best = population.get_best_genome()
                        avg_fitness = sum(g.fitness for g in population.population) / len(population.population)
                        st.write(f"**Best Fitness:** {best.fitness:.2f}")
                        st.write(f"**Average Fitness:** {avg_fitness:.2f}")
                except Exception as e:
                    st.error(f"Evolution failed: {e}")

        # Population Visualization
        st.subheader("Population Overview")
        if population.population and len(population.population) > 0:
            fitness_values = [g.fitness for g in population.population]
            import pandas as pd
            df = pd.DataFrame({
                'Genome': [f"G{i}" for i in range(len(fitness_values))],
                'Fitness': fitness_values
            })
            st.bar_chart(df.set_index('Genome'))
        else:
            st.info("No population data available. Initialize population first.")

    except Exception as e:
        st.error(f"Failed to initialize evolution components: {e}")

# Bridge Tab
with tabs[6]:
    st.header("⚡ Bridge - Safe OS Interactions")

    bridge = Bridge()

    st.subheader("Command Whitelist")
    whitelist = bridge.whitelist
    st.write("Allowed commands:")
    for cmd in whitelist:
        st.code(cmd, language="bash")

    st.subheader("Execute Command")
    st.warning("⚠️ Only whitelisted commands will execute")

    command = st.text_input("Command", placeholder="ls -la")
    timeout = st.number_input("Timeout (seconds)", min_value=1, max_value=300, value=30)

    if st.button("Execute"):
        try:
            with st.spinner("Executing..."):
                result = bridge.execute(command, timeout=timeout)

                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    st.success(f"Return code: {result.get('returncode', 'N/A')}")

                    if result.get("stdout"):
                        st.subheader("Output (stdout)")
                        st.code(result["stdout"], language="bash")

                    if result.get("stderr"):
                        st.subheader("Errors (stderr)")
                        st.code(result["stderr"], language="bash")

        except Exception as e:
            st.error(f"Execution failed: {e}")

    st.subheader("Emergency Controls")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚨 Trigger Emergency Stop"):
            EmergencyStop.stop()
            st.error("Emergency stop activated!")
    with col2:
        if st.button("🔄 Reset Emergency Stop"):
            EmergencyStop.reset()
            st.success("Emergency stop reset")

    st.info(f"Emergency Stop Status: {'ACTIVE' if EmergencyStop.check() else 'INACTIVE'}")

# Logs Tab
with tabs[7]:
    st.header("📝 Logs - System Output")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Live Logs")
    with col2:
        if st.button("Clear Logs"):
            st.session_state.logs = StringIO()
            log_handler.stream = st.session_state.logs
            st.success("Logs cleared")

    # Display logs
    log_content = st.session_state.logs.getvalue()
    if log_content:
        st.text_area("Log Output", log_content, height=400, disabled=True)
    else:
        st.info("No logs available. Run operations to generate logs.")

    # Auto-refresh option
    auto_refresh = st.checkbox("Auto-refresh (every 5s)")
    if auto_refresh:
        time.sleep(5)
        st.rerun()

# Stats Tab
with tabs[8]:
    st.header("📈 Stats - System Statistics")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Execution Statistics")
    with col2:
        if st.button("🔄 Refresh"):
            st.rerun()

    # Get orchestrator stats
    try:
        orch = ReasoningOrchestrator()
        stats = orch.get_execution_stats()

        if stats.get("total_tasks", 0) > 0:
            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Tasks", stats["total_tasks"])
            col2.metric("Successful", stats["successful_tasks"])
            col3.metric("Success Rate", f"{stats['success_rate']:.1%}")
            col4.metric("Total Retries", stats["total_retries"])

            # Performance metrics
            st.subheader("Performance")
            col1, col2 = st.columns(2)
            col1.metric("Avg Execution Time", f"{stats['average_execution_time']:.2f}s")
            col2.metric("Avg Retries", f"{stats['average_retries']:.2f}")

            # Memory stats
            st.subheader("Memory Statistics")
            mem = AgeMem(st.session_state.config)
            col1, col2 = st.columns(2)
            col1.metric("STM Entries", len(mem.stm))
            col2.metric("LTM Entries", len(mem.ltm))

            # Tool Forge stats
            st.subheader("Tool Forge Statistics")
            forge = ToolForge()
            forge_stats = forge.get_stats()
            col1, col2, col3 = st.columns(3)
            col1.metric("Detected Patterns", forge_stats.get("detected_patterns", 0))
            col2.metric("Forged Tools", forge_stats.get("forged_tools", 0))
            col3.metric("Installed Plugins", forge_stats.get("installed_plugins", 0))

            # Evolution stats (if available)
            try:
                st.subheader("Evolution Statistics")
                grpo = GRPOTrainer(st.session_state.config)
                pop = grpo.population_manager
                if pop.population:
                    col1, col2 = st.columns(2)
                    col1.metric("Population Size", len(pop.population))
                    best = pop.get_best_genome()
                    col2.metric("Best Fitness", f"{best.fitness:.2f}" if best else "N/A")
            except Exception:
                st.info("Evolution stats not available")

        else:
            st.info("No execution statistics available. Run some tasks first.")

    except Exception as e:
        st.error(f"Failed to load stats: {e}")

st.sidebar.info("SENSE v3.0 - Intelligence Amplification")


def main():
    """Main entry point for console script."""
    # Streamlit runs the entire file, so we don't need to do anything here
    # This function is just for the entry point
    pass


if __name__ == "__main__":
    main()
