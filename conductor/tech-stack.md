# Technology Stack: SENSE v2

This document outlines the technology stack for the SENSE v2 framework.

## 1. Programming Language

*   **Primary Language:** Python (3.6 or newer)
    *   **Reasoning:** Python is the de facto standard for AI and machine learning development, offering a rich ecosystem of libraries and a syntax that allows for rapid prototyping and development.

## 2. Core Libraries & Frameworks

*   **Machine Learning:**
    *   **TensorFlow (2.x):** Used for building and training neural network models, including the LSTMs and autoencoders that are central to the framework's learning and anomaly detection capabilities.
*   **Scientific Computing & Data Manipulation:**
    *   **NumPy:** The fundamental package for numerical computation in Python.
    *   **SciPy:** Used for scientific and technical computing.
    *   **Pandas:** Provides high-performance, easy-to-use data structures and data analysis tools.
*   **Natural Language Processing:**
    *   **Transformers:** Used for leveraging state-of-the-art NLP models.
*   **System Interaction:**
    *   **psutil:** A cross-platform library for retrieving information on running processes and system utilization (CPU, memory, disks, network), which is critical for the health monitoring components of SENSE.
    *   **Requests:** A simple, yet powerful, HTTP library for making requests to external systems.

## 3. Architecture

*   **Pattern:** Custom Hierarchical Agent-Based Architecture
    *   **Description:** SENSE v2 employs a unique, custom-built architecture centered around two main hierarchical levels:
        1.  **Agent 0 (The School):** A co-evolutionary system for model training and curriculum learning.
        2.  **Agent Zero (The Workplace):** A hierarchical task-delegation system for real-world operations, where a Master Agent orchestrates specialized sub-agents.
    *   **Reasoning:** This architecture is designed for maximum autonomy and adaptability, allowing the system to both learn and execute complex tasks in a structured and scalable manner. It separates the concerns of "learning" from "doing," enabling more specialized and efficient agent behaviors.
