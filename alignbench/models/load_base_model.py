# alignbench/models/load_base_model.py

from transformers import AutoModelForCausalLM, AutoTokenizer

def load_base_model(model_name_or_path):
    """
    Loads the base (pre-trained) model and tokenizer from Hugging Face Hub or local path.
    """
    print(f"[INFO] Loading base model: {model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # Handle missing pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
