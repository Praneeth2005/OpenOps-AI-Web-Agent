# OpenOpsAI Web-Agent: Empowered With deepseek R1:32b, OpenAI-o1 , Qwen 2.5-VL , claude-3.5 sonnet , gemini-2.0-flash-exp
A Project By Praneeth Kilari


![image](https://github.com/user-attachments/assets/ef0f556e-bd0c-4ffb-82e7-86aee6d227b9)


![image](https://github.com/user-attachments/assets/9a8d8905-a29a-4017-b8bf-75fad7d5f317)
Performaning A Task Using Locally Deployed deepseek r1:32b 
![image](https://github.com/user-attachments/assets/57316ba2-9847-4ded-b8fd-d9fa8fea9daa)
![image](https://github.com/user-attachments/assets/aecfe654-921e-4468-b9aa-ba8ed96260dd)
prompt is " Help me book a campsite on Hipcamp for a weekend in march 6th 2025. I prefer a site near Yellowstone with access to running water. My budget is $50 per night " 
using deepseek r1:32b (deployed locally)

**Seamlessly orchestrate your AI-powered tasks—empowering you to operate, automate, and innovate with OpenAI.**

OpenOps Agent is a web-based control center that leverages a headless browser and LLM (Large Language Model) capabilities. By simplifying complex tasks into natural language instructions, it allows you to offload repetitive, multi-step processes to an AI assistant.

## Table of Contents
- [Features](#features)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project Layout](#project-layout)
- [LLM Performance and Screenshots](#llm-performance-and-screenshots)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Features
- **Browser Automation:** Run tasks on a headless browser, or use your own browser instance.
- **LLM Integration:** Seamlessly connect to OpenAI or other LLM providers to parse, understand, and execute instructions.
- **Customizable Agents:** Switch between a default (org) agent or your custom agent logic.
- **Configurable UI:** Gradio-based interface with an attractive design for easy task orchestration.
- **Recording & Tracing:** Optionally record browser sessions and create execution traces for debugging.

---

## Getting Started

### Prerequisites
1. **Python 3.8+**  
   Ensure you have Python installed (3.8 or higher).
2. **Node.js (Optional)**  
   Required if you wish to run some additional scripts or advanced builds (depending on your workflow).

### Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/openops-agent.git
   cd openops-agent
   ```
2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure environment variables** (optional):  
   If you have an `.env` file with API keys or special settings, place it in the project root. For example:
   ```
   OPENAI_API_KEY=your-openai-api-key-here
   CHROME_PATH=/path/to/chrome-or-chromium
   ```
4. **Run the app**:
   ```bash
   python webui.py
   ```
   By default, the app will start at `http://127.0.0.1:7788`.

---

## Usage

1. **Open the Gradio UI**  
   Open your browser and navigate to the URL printed in your terminal (e.g., `http://127.0.0.1:7788`).

2. **Set Your Configuration**  
   - **AI Agent Configuration**: Choose between the built-in “org” agent or your custom agent.  
   - **Language Model Settings**: Select your LLM provider (e.g., OpenAI), specify your model name (`gpt-4`, etc.), temperature, and other LLM-related fields.
   - **Browser Environment**: Decide whether you want a headless browser or use your own browser instance. Configure window size, security settings, etc.
   - **Project Configuration**: Decide where to store traces and recordings, or set advanced project-wide settings.

3. **Describe Your Task**  
   In the **Run the AI Agent** section, type in your task. Example:
   ```
   I want to travel from Hyderabad to Visakhapatnam, book flight tickets on Jan 29th, 
   and my return is on Jan 31st. Book any flights.
   ```
   Add additional hints or constraints in the **Additional Hints** box.

4. **Run Agent**  
   - Click **Run Agent**.  
   - The AI will process your instructions, open the browser (headless or visible), and attempt to accomplish the requested steps.  
   - You can **Stop** the agent at any time if you need to halt execution.

5. **Monitor and Debug**  
   - Follow the **Live Browser View** to see your AI agent’s progress in real-time (or screenshots for headless mode).  
   - If you enabled **Recording** or **Tracing**, you can retrieve the generated videos and trace files under the **Recordings** and **Trace** sections.

---

## Configuration

OpenOps Agent supports a flexible configuration system. Edit default settings inside `default_config_settings.py`, or specify your own config file.

**Key Configuration Fields**  
- **llm_provider**: The chosen LLM provider (e.g., OpenAI, Anthropic, etc.).  
- **llm_model_name**: Model name, e.g., `text-davinci-003` or `gpt-4`.  
- **headless**: Run the browser with or without a GUI.  
- **disable_security**: Whether to disable some Chromium security checks.  
- **save_recording_path**: Where to store browser recordings.  
- **save_trace_path**: Where to store agent trace logs.  
- **use_own_browser**: Use your own system browser (experimental) instead of a headless one.  

Customize these settings either from the UI or by editing your config file, then **Load** or **Save** the configuration via the **Project Configuration** section in the UI.

---

## Project Layout

<details>
  <summary>Key Files and Folders</summary>

- **webui.py**  
  The Gradio-based UI entry point. Run this to launch OpenOps Agent.

- **src/**  
  Contains your custom agents, prompts, controllers, and utility functions.
  
- **browser_use/**  
  Houses the core browser automation logic and the default agent implementation.

- **requirements.txt**  
  Lists project dependencies.

- **.env** (optional)  
  Contains environment variables such as `OPENAI_API_KEY`, `CHROME_PATH`, etc., if not supplied directly in the UI.

</details>

---

Below is a more structured version of the **LLM Performance and Screenshots** section, with each model’s screenshots and a brief description:

---

## LLM Performance and Screenshots

To highlight how different language models perform within OpenOps Agent, we’ve included screenshots and notes below.

### Claude 3.5 Sonnect
Here’s a screenshot demonstrating Claude 3.5 Sonnect’s token usage and performance:

![image](https://github.com/user-attachments/assets/02e69b7b-63cb-4f2f-9b57-2f5787e81f75)

Additional view or details:

![image](https://github.com/user-attachments/assets/a0cf373b-e786-4839-9c79-3d0a63377f05)

**Observations/Notes**  
- Summaries of token usage or performance, e.g., “Handles long text well but uses moderate tokens.”  
- Any unique advantages or drawbacks observed during testing.

---

### Google Gemini 2.0 (Flash-Thinking-Exp-01-21)
Below are multiple screenshots showing token usage or other performance metrics:

![image](https://github.com/user-attachments/assets/ad230f4f-b505-4e27-ae19-fea1dadf08b0)
![image](https://github.com/user-attachments/assets/4f3f4e2b-52f1-493c-a1b8-03cff183f2db)
![image](https://github.com/user-attachments/assets/fed13711-3b75-46a8-9072-b1cae076878b)

**Observations/Notes**  
- “Fast generation speeds for shorter prompts.”  
- “Handles complex reasoning tasks effectively, though token usage can spike.”

---

### OpenAI ChatGPT 4o

**Observations/Notes**  
- “Excels in generating coherent long-form text with fewer token resets.”  
- “Higher-level reasoning is strong, with robust context handling.”



## Contributing

We welcome pull requests, issues, and feature requests! Feel free to open an issue to discuss major changes or improvements. For small fixes, simply create a PR referencing the relevant issue (if any).

1. **Fork** this repository.
2. **Create a branch**: `git checkout -b feature/new-idea`.
3. **Commit your changes**: `git commit -m 'Add new idea'`.
4. **Push to the branch**: `git push origin feature/new-idea`.
5. **Submit a pull request**.

---

## License

This project is licensed under the [MIT License](LICENSE). Feel free to modify or distribute as per the terms of the license.

---

## Acknowledgments

- [OpenAI](https://openai.com/) for providing powerful language models.  
- [Gradio](https://www.gradio.app/) for making it easy to build modern UIs in Python.  
- The entire open-source community for building the tools and libraries that made this project possible.

---

**Thank you for trying OpenOps Agent!**  
