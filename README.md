# 🤖 Local Agentic LLM Fine-Tuning Pipeline (Qwen2.5)

An end-to-end local LLM fine-tuning and agentic workflow setup optimized for consumer-grade GPUs (NVIDIA RTX 5060 - 8GB VRAM).

## 🚀 Overview
This repository contains scripts for fine-tuning **Qwen2.5-1.5B** using 4-bit quantization and QLoRA, designed to build lightweight local autonomous AI agents.

## 🛠️ Tech Stack & Hardware
- **GPU:** NVIDIA GeForce RTX 5060 (8GB VRAM)
- **Frameworks:** PyTorch, Hugging Face `transformers`, `peft`, `bitsandbytes`, `trl`
- **Methodology:** QLoRA (4-bit quantization + LoRA adapters)

## 📁 Repository Structure
```text
local-llm-finetuning-qwen/
├── configs/
├── data/
├── src/
│   ├── inference.py
│   └── train.py
├── README.md
└── requirements.txt

