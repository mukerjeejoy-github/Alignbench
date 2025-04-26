# alignbench/evaluation/fidelity_loss.py

import torch
from tqdm import tqdm

def compute_fidelity_loss(base_model, finetuned_model, tokenizer, dataset):
    """
    Compute Fidelity Loss (Cross-Entropy Divergence) between base and fine-tuned models.
    """
    print("[INFO] Computing Fidelity Loss (FL)")

    base_model.eval()
    finetuned_model.eval()

    loss_fn = torch.nn.KLDivLoss(reduction="batchmean")
    total_loss = 0
    num_batches = 0

    for example in tqdm(dataset, desc="Computing FL"):
        inputs = tokenizer(example['text'], return_tensors="pt", padding="max_length", truncation=True, max_length=128)
        input_ids = inputs.input_ids

        with torch.no_grad():
            base_outputs = base_model(input_ids=input_ids)
            finetuned_outputs = finetuned_model(input_ids=input_ids)

        base_logits = base_outputs.logits
        finetuned_logits = finetuned_outputs.logits

        # Normalize to log-probabilities
        base_log_probs = torch.nn.functional.log_softmax(base_logits, dim=-1)
        finetuned_probs = torch.nn.functional.softmax(finetuned_logits, dim=-1)

        batch_loss = loss_fn(base_log_probs, finetuned_probs)
        total_loss += batch_loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss
