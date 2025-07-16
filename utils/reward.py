from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch

# Load tokenizer and model from Hugging Face
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2) # labels of 1 and 0, so num_labels is 2
model.to("cuda")

# Load data from reward dataset
dataset = load_dataset("json", data_files={"train": "Dataset/reward_data.jsonl"})

# Read questions and chosen/rejected answers and label them into pairwise data
def prepare_data(examples):
    """examples should be a dict like {"question": [...], "chosen": [...], "rejected": [...]}
    """
    prompts = []
    texts = []
    labels = []
    for q, c, r in zip(examples["question"], examples["chosen"], examples["rejected"]):
        # positive example
        prompts.append(q)
        texts.append(c)
        labels.append(1) # label chosen=1
        # negative example
        prompts.append(q)
        texts.append(r)
        labels.append(0) # label rejected=0
    return {"prompt": prompts, "text": texts, "labels": labels}
pairwise_data = dataset["train"].map(prepare_data, batched=True, remove_columns=dataset["train"].column_names)

# Tokenize pairwise data
def tokenize_function(examples):
    return tokenizer(examples["prompt"], examples["text"], max_length=128, truncation=True, padding="max_length")
tokenized_data = pairwise_data.map(tokenize_function, batched=True)

# Setup training arguments
training_args = TrainingArguments(
    output_dir="models/reward/reward_output",
    per_device_train_batch_size=16,
    num_train_epochs=4,
    fp16=True
)

# Setup trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data,
    tokenizer=tokenizer
)

# Train reward model
trainer.train()

# Save reward model
model.save_pretrained("models/reward/reward_model")
tokenizer.save_pretrained("models/reward/reward_model")