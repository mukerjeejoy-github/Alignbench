# alignbench/datasets/load_in_domain.py

from datasets import load_dataset

def load_in_domain_dataset():
    """
    Loads the in-domain dataset (default: Wikitext-2).
    """
    print("[INFO] Loading in-domain dataset (wikitext-2)")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test[:1%]")  # small subset for quick tests
    return dataset
