# SENSE (Smartphone Evolutionary Neural System Engine) v3.4

> **"A Universal Cybernetic Kernel for Local AI"**

SENSE is a fully autonomous, local-first agent framework designed to run universally on **Android (Termux), Linux, macOS, and Windows**. It transforms any device into a self-evolving AI assistant that learns from experience, remembers its user, and seeks truth using a "Resonance Engine."

## 🧠 Core Architecture

* **Universal Core:** Automatically detects the OS (Android/Linux/Win) and adapts file paths and system calls accordingly.
* **The Brain (Orchestrator):** A state-aware reasoning engine that dynamically switches between "Hunter" (Search) and "Analyst" (Synthesis) modes.
* **The Instincts (Genetic Memory):** Reinforcement learning that remembers *how* to use tools.
* **The Cortex (Episodic Memory):** A persistent JSON memory system that stores user context in your documents folder.
* **The Eyes (Grok-Mode):** A "Deep-Net" search engine that prioritizes high-resonance sources like Grokipedia, GitHub, and Arxiv.

## 🛠️ Tech Stack

* **Language:** Python 3.x (Universal)
* **Backend:** OpenAI-Compatible Local Server (llama.cpp / Ollama / LM Studio)
* **Tools:** Dynamic "Harvesting" system (DuckDuckGo, YouTube-DL)

## 🚀 Installation

1.  **Clone the Repo:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/SENSE.git
    cd SENSE
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    # (On Termux only: pkg install termux-api)
    ```

3.  **Run SENSE:**
    ```bash
    # The system will auto-detect your OS and configure paths.
    sense "What is the Linux Kernel release date?"
    ```

## 🛡️ Privacy & Security

SENSE is designed for **Total Privacy**.
* **Android:** Data stored in `/sdcard/Download/SENSE_Data`.
* **PC/Mac:** Data stored in `~/Documents/SENSE_Data` (or equivalent).
* No data is ever sent to the cloud.
* Personal data files are Git-Ignored by default.

---
*Built by Todd Eddings.*
