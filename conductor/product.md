# Product Guide: SENSE v2

## 1. Vision

SENSE v2 is a next-generation, self-evolving AI framework designed for complex, dynamic environments. It extends the original SENSE project by introducing a hierarchical, agent-based architecture that enables autonomous learning, adaptation, and execution of sophisticated tasks. The vision is to create a system that can not only learn from data but also actively manage its own operations, evolve its capabilities, and interact with its environment at a high level of abstraction.

## 2. Target Audience

*   **AI Researchers & Practitioners:** Those who require a robust platform for experimenting with agent-based architectures, evolutionary algorithms, and continuous learning systems.
*   **Software Engineers:** Developers building applications that need to adapt in real-time to changing data streams or user behaviors.
*   **Data Scientists:** Professionals who need to automate complex data analysis and model management pipelines, especially in environments with data drift.

## 3. Core Problems to Solve

*   **Autonomous Adaptation:** How can an AI system continuously adapt its models and strategies without human intervention when faced with evolving data (data drift)?
*   **Complex Task Execution:** How can an AI system reliably perform complex, multi-step tasks that involve interacting with the operating system, file system, and other external tools?
*   **Knowledge Persistence:** How can an agent effectively store, retrieve, and utilize knowledge over long periods to improve its performance and decision-making?
*   **Efficient Resource Management:** How can the system manage computational resources effectively, especially on modern, high-performance hardware?

## 4. Key Features & Capabilities

*   **Hierarchical Agent Architecture:**
    *   **Agent 0 (The School):** A co-evolutionary environment where a "Teacher" agent generates tasks and a "Student" agent learns and evolves its capabilities through a process of trial, error, and verification.
    *   **Agent Zero (The Workplace):** An orchestration layer where a "Master Agent" delegates complex, real-world tasks to a team of specialized sub-agents (e.g., Terminal, Filesystem, Browser).
*   **AgeMem - The Filing Cabinet:** A structured memory system that provides both short-term (STM) and long-term (LTM) storage for agents, enabling them to persist and recall information efficiently.
*   **Tool-Centric Design:** Agents interact with their environment through a well-defined set of schema-based Python tools, ensuring robust and predictable behavior. This includes tools for file I/O, running terminal commands, and web browsing.
*   **Self-Correction and Evolution:** The framework incorporates self-correction loops, allowing agents to learn from their mistakes. The evolutionary nature of Agent 0 ensures that the system's core models continuously improve.
*   **Hardware Optimization:** The architecture is designed with modern hardware in mind, specifically targeting Unified Memory Architectures (UMA) and AMD's RDNA 3.5 for high-performance computing.

## 5. Non-Functional Requirements

*   **Modularity:** The system is designed to be highly modular, allowing for the easy addition of new agents, tools, and memory components.
*   **Scalability:** The architecture should be able to scale to handle an increasing number of agents and more complex tasks.
*   **Reliability:** Through self-correction and robust error handling, the system aims for a high degree of operational reliability.
