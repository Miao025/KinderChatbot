from utils.RLHF import return_best_response

if __name__ == "__main__":
    prompt = "Why is the sky blue?"
    best_response = return_best_response(prompt)
    print(best_response)