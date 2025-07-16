import os
import json
import google.generativeai as genai
from typing import Literal

def call_gemini(prompt: str) -> str:
    model = genai.GenerativeModel("gemini-2.0-flash")
    api_key = os.getenv("api_key")
    genai.configure(api_key=api_key)
    response = model.generate_content(prompt)
    return response.text


def generate_data(batch_size: int, target_pairs: int, dataset_type: Literal["sft", "reward"]) -> None:
    all_pairs = []

    while len(all_pairs) < target_pairs:
        # Build prompt
        if dataset_type == "sft":
            prompt = f"""
            Generate {batch_size} common questions that kindergarten children ask their teacher.
            For each question, provide a very simple and kind answer suitable for a 3-year-old.
            Output the results as a JSON array with objects like:
            {{"question": "...", "answer": "..."}}
        """
        elif dataset_type == "reward":
            prompt = f"""
            Generate {batch_size} common questions 3-year-old children ask their teacher.
            For each question, provide two answers:
            1. A very simple, kind, helpful, and age-appropriate answer (label it as "chosen").
            2. A very complicated, unhelpful, or rude answer (label it as "rejected").
            Output the results as a JSON array with objects structured like this:
            {{
            "question": "<question>",
            "chosen": "<kind answer>",
            "rejected": "<unkind answer>"
            }}
        """

        # Get response text
        response_text = call_gemini(prompt)
        
        # Parse JSON array from responses
        pairs = json.loads(response_text[7:-4])
        
        # Add to pair list
        all_pairs.extend(pairs)

    # Save pair list to Json file
    with open(f"{dataset_type}_data.jsonl", "w") as f:
        for pair in all_pairs[:(target_pairs+1)]:
            if dataset_type == "sft":
                obj = {
                    "question": pair["question"],
                    "answer": pair["answer"]
                }
            elif dataset_type == "reward":
                obj = {
                    "question": pair["question"],
                    "chosen": pair["chosen"],
                    "rejected": pair["rejected"]
                }                
            f.write(json.dumps(obj) + "\n")


# Run this to get your dataset, recommend at least: sft 500-2000 pairs, reward 300-1000 triplets.
if __name__=="__main__":
    generate_data(batch_size=..., target_pairs=..., dataset_type=...)