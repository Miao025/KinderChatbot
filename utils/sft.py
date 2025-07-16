from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig
import torch

# Load tokenizer and model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-1.8B", trust_remote_code=True) # Note that for well-known brands, no need to specify "username/model-name" but the shorcut name
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-1.8B", trust_remote_code=True).to("cuda") # use gpu for training

# Configure LoRA
lora_config = LoraConfig(
    r=8, # lora rank
    lora_alpha=16, # lora scale factor
    target_modules=["q_proj", "k_proj", "v_proj"],  # layers lora will modify, here we set to attention layers
    lora_dropout=0.1 # randomly set 10% elements to zero 
)

# Wrap the model with LoRA adapters for fine-tuning
model = get_peft_model(model, lora_config)

# Load data from sft dataset
dataset = load_dataset("json", data_files={"train": "Dataset/sft_data.jsonl"})

# Combine question and answer into one text and tokenize it
def prepare_tokens(examples):
    """examples should be a dict like {"question": [...], "answer": [...]}, tokenized_inputs will be a dict like {"input_ids": [...], "labels": [...]}
    """
    texts = [q + " " + a for q, a in zip(examples["question"], examples["answer"])] # combine questions and answers into one sentence (Note that gpt2 is a decoder-only model, it predicts the next tokens)
    tokenized_inputs = tokenizer(
        texts,
        max_length=512, # cut the input tokens if it exceeds 512
        truncation=True,
        padding="max_length" # pad tokens that are less than 512
        )
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy() # set to predict the next token
    return tokenized_inputs

# Apply tokenization to entire dataset
tokenized_dataset = dataset.map(prepare_tokens, batched=True) # save the data into a batch(list) of examples

# Setup training arguments
training_args = TrainingArguments(
    output_dir="models/sft/lora_output",
    per_device_train_batch_size=5, # set how many examples to train on gpu at once
    num_train_epochs=1, # train the dataset 1 time
    fp16=True  # set this for faster training
)

# Setup Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    tokenizer=tokenizer
)

# Train LoRA adapter
trainer.train()

# Save only the LoRA adapters (small size) and tokenizer
model.save_pretrained("models/sft/lora_adapter")   
tokenizer.save_pretrained("models/sft/lora_adapter")