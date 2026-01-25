# SENSE Project Context

## Project Overview

SENSE (Self-Evolving Network & Semantic Engine) is a model-agnostic, self-evolving AI agent framework designed to improve itself through recursive iterations. It transforms any LLM/SLM into a grounded, agentic system with persistent memory and autonomous capabilities.

**Key Features:**
*   **Self-Evolution:** Uses genetic algorithms (GRPO) to evolve system code and prompts.
*   **Reasoning Orchestrator:** Implements a Reflexion loop with Architect (planning), Worker (execution), and Critic (verification) phases.
*   **Tool Crystallization:** Automatically creates reusable tools from repeated successful code patterns via "Tool Forge".
*   **Three-Tier Grounding:** Verifies actions using Synthetic (deterministic), Real-world (external), and Experiential (outcome-based) checks.
*   **Autonomous Operation:** Can run continuously without human intervention.
*   **Safety:** Executes OS commands through a "Bridge" with whitelisting and emergency stop mechanisms.

**Main Technologies:**
*   Python 3.9+
*   PyTorch, NumPy
*   Streamlit (Dashboard)
*   FastAPI (API Server)
*   LLM Integrations (OpenAI, Anthropic, Ollama, etc.)

## Architecture

The project is structured as a modular Python package:

*   `src/sense/core`: Core components (Reasoning Orchestrator, Memory, Evolution, Grounding).
*   `src/sense/tools`: Tool discovery and auto-integration.
*   `src/sense/marketplace`: Plugin marketplace client.
*   `src/sense/alignment`: Human alignment and uncertainty detection.
*   `src/sense/knowledge`: Internet-scale knowledge retrieval (RAG, Web Search).
*   `src/sense/meta_learning`: Curriculum evolution and difficulty estimation.
*   `src/sense/bridge`: Safe OS execution layer.
*   `src/sense/interface`: Persona definitions.

## Building and Running

**Installation:**
```bash
pip install -e .
```

**Running the Agent:**
*   **Continuous Autonomous Mode:**
    ```bash
    sense --mode continuous
    ```
*   **Single Task Mode:**
    ```bash
    sense --mode single --task "Your task here"
    ```
*   **Evolution Mode:**
    ```bash
    sense --mode evolve --generations 10
    ```

**Interfaces:**
*   **Dashboard:**
    ```bash
    sense-dashboard
    ```
*   **API Server:**
    ```bash
    sense-api --port 8000
    ```

**Testing:**
```bash
pytest tests/ -v
```

## Development Conventions

*   **Code Style:** Follow standard Python PEP 8 conventions.
*   **Testing:** Maintain high test coverage. New features must include unit tests in the `tests/` directory.
*   **Safety:** All OS interactions must go through the `Bridge` component. Never use `subprocess` or `os.system` directly in agent logic.
*   **Documentation:** Keep `CLAUDE.md`, `IMPLEMENTATION_STATE.md`, and `README.md` updated with architecture changes and status.
*   **Configuration:** Use `config/sense.yaml` (or defaults) for system settings.
*   **Contribution:** Follow the guidelines in `CONTRIBUTING.md`.

## Current Status (as of 2026-01-24)

*   **Version:** v6.0 Complete
*   **Active Features:** Meta-Learning, Tool Ecosystem, Human Alignment, Internet Knowledge, Reflexion Loop, GRPO Evolution.
*   **Next Phase:** v7.0 (World Model & Memory).
