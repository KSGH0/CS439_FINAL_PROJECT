# Estimating the Human Capital Cost of AI Research
### CS439 Final Project — Karamvir Singh — Rutgers University

> **Research Question:** What is the estimated human capital cost of AI research across subfields from 2022 to 2026 — and can NLP predict research cost from abstract text alone?

## Results Summary
| Model | Accuracy | Weighted F1 |
|-------|----------|-------------|
| Naive Bayes (from scratch) | 0.5031 | 0.4932 |
| Logistic Regression (OvR, L2) | 0.4361 | 0.2649 |
| MLP (256-128-64, Adam) | **0.5254** | **0.5179** |
| Random baseline | 0.3333 | 0.3333 |

- **288,368** ArXiv papers (cs.AI, cs.LG, cs.CL, cs.NE, cs.IR, cs.CV), Dec 2022–Mar 2026
- Mean estimated cost per paper: **$40,239** | Monthly growth Dec 2022–Feb 2026: **+14.2%**
- K-Means cluster purity: **0.4363** (vs 0.3333 random baseline)

## Setup
```bash
pip install numpy pandas matplotlib seaborn scikit-learn requests torch
```
Python 3.12 for GPU (ROCm/CUDA auto-detected). Python 3.13 for CPU-only.

## Data
Place the 6 ArXiv JSONL files in `archive/`:
```
archive/cs_ai_papers.jsonl    archive/cs_ne_papers.jsonl
archive/cs_lg_papers.jsonl    archive/cs_ir_papers.jsonl
archive/cs_cl_papers.jsonl    archive/cs_cv_papers.jsonl
```

## Run
**GUI (recommended):**
```bash
python main.py
```

**Command line:**
```bash
python 01_cost.py       # Position-based cost per paper  (~5 min)
python 02_nlp.py        # TF-IDF + NB/LR/MLP classifiers (~20 min)
python 03_analysis.py   # Figures + analysis             (~5 min)
```

## Project Structure
```
├── main.py             # Entry point — GUI (settings, pipeline, query)
├── config.py           # All hyperparameters (single source of truth)
├── device.py           # Hardware auto-detection with CPU fallback
├── 01_cost.py          # Loads 288K papers, position-based cost model
├── 02_nlp.py           # TF-IDF + NB / LR / MLP classifiers
├── 03_analysis.py      # PCA, K-Means, temporal analysis, figures
├── archive/            # Raw ArXiv JSONL data (6 files)
├── inputs/             # Intermediate pipeline data
├── outputs/            # Figures + model_results.csv
└── cs439_final_project.tex  # Final paper (NeurIPS format)
```

## Repository
[https://github.com/KSGH0/CS439_FINAL_PROJECT](https://github.com/KSGH0/CS439_FINAL_PROJECT)
```

## Key Design Decisions
- **Position-based salary**: first author = PhD, last = PI, middle = weighted mix — no external data needed
- **Fully configurable**: all salary tiers and hyperparameters adjustable in `main.py`

## Reproducibility
All results are deterministic given `SEED = 2026` in `config.py`. Full pipeline runs in ~30 minutes.
