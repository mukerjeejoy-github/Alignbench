# alignbench/utils/metrics_format.py

import yaml
import os

def save_metrics(metrics: dict, output_dir: str):
    """
    Save evaluation metrics to a YAML file in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "benchmark_results.yaml")
    with open(output_path, "w") as f:
        yaml.dump(metrics, f)
    print(f"[INFO] Metrics saved to {output_path}")
