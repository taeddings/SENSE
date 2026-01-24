"""SENSE v3.0 Comprehensive Dashboard - Main Entry Point"""

import streamlit as st
import asyncio
import sys
import time
import threading
import logging
from io import StringIO

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
    forge = ToolForge()
    if st.button("Scan Memory"):
        candidates = forge.scan_memory([])
        st.write("Candidates:", len(candidates))

st.sidebar.info("SENSE v2.3 - Intelligence Amplification")
