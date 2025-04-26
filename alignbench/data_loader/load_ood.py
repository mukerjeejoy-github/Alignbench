# alignbench/datasets/load_ood.py

from datasets import load_dataset

def load_ood_dataset():
    """
    Loads an out-of-domain (OOD) dataset for drift evaluation (default: C4 subset).
    """
    print("[INFO] Loading OOD dataset (C4 subset)")
    dataset = load_dataset("c4", "en", split="validation[:1%]", trust_remote_code=True,)  # small subset for quick OOD testing
    return dataset
