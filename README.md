# 🤖 KinderChatbot

**KinderChatbot** is an AI-powered chatbot tailored for kindergarten-teacher-style conversational use cases. It combines **Supervised Fine-Tuning (SFT)** and **Reward Modeling** to achive **Reinforcement Learning from Human Feedback (RLHF)**. Lightweight **LoRA adapters** enable efficient fine-tuning on the **attention layers** of pretrained model, while a custom reward model trained with human preferences intelligently selects the best reply.

---

## ✨ Features

- **Supervised Fine-Tuning (SFT):** Fine-tune a base language model using LoRA adapters for efficient and modular training.
- **Reward Model (RLHF):** Train a reward model with human preferences to score and rerank multiple generated responses.
- **Data-free Knowledge Distillation:** Generate synthetic dialogues as training data, and review by humans.
- **Modular Codebase:** Clean, extensible architecture for dataset generation, SFT training, reward modeling, and end-to-end inference.
- **Gradio Interface (bonus):** Interactive and user-friendly web UI deployed on HuggingFace Space for real-time chatting.

---

## 🗂️ Project Structure

```
.env
.gitignore
main.py
README.md
requirements.txt

Dataset/
└── generate_data.py

models/
├── reward/
└── sft/

utils/
├── reward.py
├── RLHF.py
└── sft.py
```

- `main.py`: Entry point for running the chatbot.
- `requirements.txt`: Python dependencies.
- `Dataset/`: Script for generating synthetic training data and directory for storing the generated data.
- `models/`: Directory for storing sft and reward models with their checkpoints.
- `utils/`: Core scripts for sft training, reward scoring training, and RLHF-based reranking.

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Model Weights

Model weights and tokenizer files are **not included** in this repository. Please download them from my HuggingFace: [LoRA fine-tuned SFT model](https://huggingface.co/Miao025/Qwen-KinderChatbot-LoRA), [Reward model](https://huggingface.co/Miao025/Qwen-KinderChatbot-Reward), and place them into the correct directories:

- LoRA fine-tuned SFT model → `models/sft/lora_adapter/`
- Reward model → `models/reward/reward_model/`

---

### 3. Try the Chatbot

Run the following command to start the chatbot interface:

```bash
python main.py
```
You can also interact with the chatbot [online](https://huggingface.co/spaces/Miao025/qwen-kinderchatbot).

---

## 🏃‍ Training
- **Supervised Fine-Tuning (SFT):** See `sft.py` for fine-tuning the base model (Qwen1.5-1.8B) with LoRA adapters.
- **Reward Model Training:** See `reward.py` for full training the base reward model (DistilbBERT).
- **Ranker:** See `RLHF.py` for calling fine-tuned models, using reward model to rank, and choose the top answer.
- **Data Generation** See `generate_data.py` to create and save the training data for sft and reward.

---
## 👩🏻 Contributor
- Miao

---
## 📃 References
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PEFT (Parameter-Efficient Fine-Tuning)](https://github.com/huggingface/peft)
- [LoRA](https://huggingface.co/docs/diffusers/training/lora)
- [DistilBERT](https://huggingface.co/onnxport/distilbert-base-uncased-onnx)
- [Gradio](https://pypi.org/project/gradio/2.9b50/)