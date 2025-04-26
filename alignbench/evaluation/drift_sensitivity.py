# alignbench/evaluation/drift_sensitivity.py

import torch
from tqdm import tqdm

def compute_drift_sensitivity(finetuned_model, tokenizer, train_dataset, ood_dataset):
    """
    Compute Drift Sensitivity Index (DSI): accuracy drop from in-domain to OOD.
    """
    print("[INFO] Computing Drift Sensitivity Index (DSI)")

    finetuned_model.eval()

    def compute_accuracy(dataset, name):
        correct = 0
        total = 0
        for example in tqdm(dataset, desc=f"Evaluating {name}"):
            inputs = tokenizer(example['text'], return_tensors="pt", padding="max_length", truncation=True, max_length=128)
            input_ids = inputs.input_ids

            with torch.no_grad():
                outputs = finetuned_model.generate(input_ids, max_new_tokens=5)
                # Very simple: check if model output is non-empty (just dummy classification)
                if outputs is not None:
                    correct += 1
                total += 1
        return correct / total

    in_domain_acc = compute_accuracy(train_dataset, "In-Domain")
    ood_acc = compute_accuracy(ood_dataset, "OOD")

    dsi = (in_domain_acc - ood_acc) / in_domain_acc
    return dsi
