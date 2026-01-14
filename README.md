LLM-AutoDiff: Auto-Differentiating Any LLM Workflow (Local Implementation)

A framework for Gradient-Driven Prompt Optimization (GDPO) running entirely on local hardware.
📄 Overview

LLM-AutoDiff is a custom implementation of the AdalFlow framework, adapted to run purely on local hardware (e.g., Google Colab T4 GPUs) without reliance on proprietary APIs like GPT-4.

This project treats LLM workflows (Agents, RAG) as Computation Graphs, where prompts are "trainable parameters" and natural language critiques act as "gradients." By backpropagating these textual gradients, we automate the optimization of prompts to improve system performance on complex tasks.


🚀 Key Innovations
1. Local-First Architecture

Unlike the original AdalFlow/Text-Grad papers which rely on OpenAI APIs, this fork implements a custom LocalLLMClient.

    Student Model: Qwen2.5-1.5B-Instruct (Fast inference, the "Target")

    Teacher/Optimizer: Qwen2.5-7B-Instruct (Stronger reasoning, the "Critic")

    Optimization: Uses bitsandbytes (4-bit quantization) and dynamic CPU/GPU offloading to fit both models on a single 16GB VRAM GPU.

2. The "PyTorch Metaphor" for LLMs

We approach prompt engineering as a differentiable programming problem:

    Forward Pass: The Student LLM attempts a task.

    Backward Pass: The Teacher LLM critiques the error (calculates the "gradient").

    Update: The Optimizer LLM refines the system prompt based on the critique.

3. Solves "Black Box" Limitations

    Pass-Through Gradients: Optimizes inputs to frozen tools (like Retrievers) by updating the upstream Query Generator.

    Time-Sequential Gradients: correctly attributes errors in cyclic agent loops to the specific step that failed.

📂 Project Structure
Bash

llm-autodiff/
├── src/
│   ├── client.py        # LocalLLMClient wrapper for HuggingFace/Transformers
│   ├── component.py     # AdalComponent definitions, Loss functions, and Evaluators
│   ├── agentct.py       # The Student Pipeline (e.g., ObjectCountTaskPipeline)
│   ├── spy_client.py    # Debugging wrapper to print Prompt/Response traces
│   ├── utils.py         # Data loaders (BBH, HotPotQA) and splitting logic
│   └── prompts.py       # System prompts for the Optimizer/Teacher
├── scripts/
│   ├── train_adalflow.py   # Main training loop (Generic)
│   └── train_object_count.py # Specific experiment for Object Counting
├── README.md
└── requirements.txt

🛠️ Installation
Prerequisites

    Python 3.10+

    NVIDIA GPU (T4 or better recommended)

    CUDA 11.8+

Setup

    Clone the repository:
    Bash

git clone https://github.com/kkoutsikos/llm-autodiff.git
cd llm-autodiff

Install Dependencies:
Bash

pip install adalflow transformers accelerate bitsandbytes datasets torch

HuggingFace Login (for Qwen models):
Python

    from huggingface_hub import login
    login("YOUR_HF_TOKEN")

🧪 Experiments & Usage
1. Object Counting (Algorithmic Precision)

    Goal: Optimize a small model (1.5B) to accurately count objects in complex lists (Big-Bench Hard).

    Workflow: One-LLM (Student -> Answer).

    Optimization: Fixes errors where the model counts legs instead of heads, or hallucinates items.

Run the training:
Bash

python scripts/train_object_count.py

    Note: This script enforces a strict 50/100/100 data split to test few-shot generalization.

2. RAG Pipeline (Planned)

    Goal: Optimize a Query Generator to improve retrieval relevance on HotPotQA.

    Workflow: Generator -> Retriever -> Answer.

    Status: In active development.

📊 Performance (Preliminary)
Task	Baseline Accuracy	Optimized Accuracy	Improvement
Object Counting	~10%	~75%	+65%
HotPotQA (RAG)	16.5%	32.25%	~2x

(Results based on initial validation runs using Qwen2.5-7B as the optimizer)
🧩 How It Works (The Loop)

    Initialization: The ObjectCountAdalComponent loads the Student and Teacher models.

    Forward: The Student answers a batch of questions.

    Eval: The answer is compared to Ground Truth (Exact Match).

    Critique: If Score < 1.0, the Teacher generates a text critique (e.g., "The model multiplied values instead of summing them").

    Optimize: The Optimizer reads the critique and proposes a new System Prompt.

    Validation: The new prompt is tested on a validation batch. If performance improves, the prompt is updated.

⚠️ Known Issues

    Teacher Hallucination: The 7B Teacher sometimes invents facts (e.g., "Chickens have 0 legs") to force the math to work. We are mitigating this with strict "Critic Constraints" in the prompt templates.

    Context Limits: Long optimization histories can overflow the context window. We currently truncate history to the last 1-2 steps.

📜 Citation

If you use this work, please credit the original AdalFlow paper and our implementation:

    LLM-AutoDiff: Auto-Differentiate Any LLM Workflow. arXiv:2501.16673