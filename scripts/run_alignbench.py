# scripts/run_alignbench.py

import argparse
import yaml
from alignbench.models.load_base_model import load_base_model
from alignbench.models.load_finetuned_model import load_finetuned_model
from alignbench.data_loader.load_in_domain import load_in_domain_dataset
from alignbench.data_loader.load_ood import load_ood_dataset
from alignbench.data_loader.load_probes import load_probe_dataset
from alignbench.evaluation.fidelity_loss import compute_fidelity_loss
from alignbench.evaluation.drift_sensitivity import compute_drift_sensitivity
from alignbench.evaluation.alignment_robustness import compute_alignment_robustness
import os

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("\n===== Loading Base and Fine-tuned Models =====")
    base_model, base_tokenizer = load_base_model(config['model']['base_model'])
    finetuned_model, finetuned_tokenizer = load_finetuned_model(config['model']['finetuned_model'])

    print("\n===== Loading Datasets =====")
    in_domain_dataset = load_in_domain_dataset()
    ood_dataset = load_ood_dataset()
    probe_dataset = load_probe_dataset()

    print("\n===== Computing Fidelity Loss (FL) =====")
    fl = compute_fidelity_loss(base_model, finetuned_model, base_tokenizer, in_domain_dataset)
    print(f"Fidelity Loss (FL): {fl:.4f}")

    print("\n===== Computing Drift Sensitivity Index (DSI) =====")
    dsi = compute_drift_sensitivity(finetuned_model, finetuned_tokenizer, in_domain_dataset, ood_dataset)
    print(f"Drift Sensitivity Index (DSI): {dsi:.4f}")

    print("\n===== Computing Alignment Robustness Score (ARS) =====")
    ars = compute_alignment_robustness(finetuned_model, finetuned_tokenizer, probe_dataset)
    print(f"Alignment Robustness Score (ARS): {ars:.4f}")

    # Save results
    os.makedirs(config['output_dir'], exist_ok=True)
    results = {
        "Fidelity Loss (FL)": float(fl),
        "Drift Sensitivity Index (DSI)": float(dsi),
        "Alignment Robustness Score (ARS)": float(ars)
    }
    output_file = os.path.join(config['output_dir'], "benchmark_results.yaml")
    with open(output_file, 'w') as f:
        yaml.dump(results, f)
    print(f"\n===== Benchmark Results Saved to {output_file} =====")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AlignBench Benchmarking")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration YAML file.")
    args = parser.parse_args()
    main(args.config)
