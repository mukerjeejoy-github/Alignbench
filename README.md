# AlignBench: Benchmarking Fine-Tuned LLM Stability, Drift, and Alignment

AlignBench is a lightweight, production-grade benchmarking framework to systematically evaluate **fine-tuned Large Language Models (LLMs)** across three critical dimensions:

- **Fidelity Loss (FL)**: How much fine-tuning changes base model behavior
- **Drift Sensitivity Index (DSI)**: How sensitive the model becomes to domain shifts
- **Alignment Robustness Score (ARS)**: How well alignment is preserved post fine-tuning

Built for researchers, ML engineers, and AI practitioners.

---

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/AlignBench.git
cd AlignBench

# Create virtual environment
python3 -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run AlignBench
python -m scripts/run_alignbench.py --config configs/alignbench_config.yaml

```

# Project Structure

```bash
AlignBench/
├── alignbench/
│   ├── evaluation/          # Metric calculations (FL, DSI, ARS)
│   ├── datasets/             # Dataset loading (in-domain, OOD, probing)
│   ├── models/               # Model loading utilities
│   ├── utils/                # Logger, metrics formatting
├── configs/
├── scripts/
├── outputs/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── docs/
```

# Features
- Supports Hugging Face models (distilgpt2, falcon, etc.)
- Plug-and-play new datasets for OOD / probing
- Modular evaluation metrics
- YAML-driven configuration
- Research-grade output ready for papers & reports

# Citations

```bash
@misc{mukerjee2025alignbench,
  title={Beyond Fine-Tuning: Stability, Alignment, and Drift in Open LLMs},
  author={Joy Mukerjee},
  year={2025},
  url={https://github.com/yourusername/AlignBench}
}
```


# Clean Lightweight requirements.txt

```text
torch>=2.0
transformers>=4.38
datasets>=2.14
evaluate>=0.4
scikit-learn>=1.3
numpy>=1.24
pandas>=2.0
pyyaml>=6.0
tqdm>=4.66
```
### Note: Fine-tuned models are not stored inside this repository. 
Users should save or download their own fine-tuned models locally in `./finetuned_model/`.
