# alignbench/models/load_finetuned_model.py

from transformers import AutoModelForCausalLM, AutoTokenizer

def load_finetuned_model(model_path):
    """
    Loads a locally fine-tuned model and tokenizer.
    """
    print(f"[INFO] Loading fine-tuned model from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Handle missing pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
