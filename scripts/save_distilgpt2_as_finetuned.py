# scripts/save_distilgpt2_as_finetuned.py

from transformers import AutoModelForCausalLM, AutoTokenizer

def save_distilgpt2_model():
    model_name = "distilgpt2"
    save_path = "./finetuned_model"

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print(f"Saved {model_name} as local fine-tuned model at {save_path}")

if __name__ == "__main__":
    save_distilgpt2_model()
