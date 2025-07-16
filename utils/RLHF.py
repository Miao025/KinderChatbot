from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch

# Load base model, LoRA adapter and reward model
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-1.8B")
tokenizer_sft = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-1.8B")
sft_model = PeftModel.from_pretrained(base_model, "models/sft/lora_adapter").to("cuda")
tokenizer_reward = AutoTokenizer.from_pretrained("models/reward/reward_model")
reward_model = AutoModelForSequenceClassification.from_pretrained("models/reward/reward_model").to("cuda")

# Generate a list of multiple (default to 5) responses using the fine-tuned model
def generate_responses(prompt, n=5):
    inputs = tokenizer_sft(prompt, return_tensors="pt", truncation=True).to("cuda") # "pt" means pytorch tensors so that the model can read
    outputs = []
    for i in range(n):
        generated_ids = sft_model.generate(
            **inputs, # the tokenized prompt
            max_length=256, # the max total length of generated text
            do_sample=True, # choose randomly instead of best next token to generate different answers
            top_p=0.9, # keep the smallest set of tokens whose cumulative probability adds up to ≥ 0.9 to avoid nonsense
            temperature=0.8 # control how sharp or flat the probability distribution is, the lower the less randomness
            )
        out = tokenizer_sft.decode(generated_ids[0], skip_special_tokens=True) # decode to human language, note to skip special tokens like padding
        if out.lower().startswith(prompt.lower()): # remove the prompt from the beginning of the answer if present
            out = out[len(prompt)+1:]
        
        outputs.append(out)
    return outputs

# Score each response using reward model
def score_response(prompt, response):
    inputs = tokenizer_reward(prompt, response, return_tensors="pt", truncation=True).to("cuda")
    with torch.no_grad():
        logits = reward_model(**inputs).logits # raw score before softmax
        score = torch.softmax(logits, dim=-1)[0,1].item()  # apply softmax to get the possibility of chosen and rejected, then get the chosen with label=1, then convert it into float
    return score

# Choose the best response
def return_best_response(prompt):
    candidates = generate_responses(prompt, n=5)
    scores = [(candidate, score_response(prompt, candidate)) for candidate in candidates]
    best_response = max(scores, key=lambda x: x[1])[0]
    return best_response