# Adaptive Top-k Retrieval for Precision Oncology RAG

> A research extension introducing **confidence-calibrated adaptive retrieval** for Retrieval-Augmented Generation (RAG) in precision oncology.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-yellow)
![Research](https://img.shields.io/badge/Research-RAG-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

# Overview

Retrieval-Augmented Generation (RAG) systems generally retrieve a **fixed number of documents (Top-k)** for every query, regardless of query complexity or retrieval confidence.

Although simple, this strategy often retrieves unnecessary contexts for straightforward clinical queries while potentially providing insufficient evidence for more challenging cases.

This repository presents a **Confidence-Calibrated Adaptive Top-k Retrieval Framework** that dynamically determines the retrieval depth using calibrated confidence estimated from similarity-score statistics.

The implementation extends the retrieval pipeline of the original **Precision Oncology RAG** framework and provides a complete experimental pipeline including:

- Confidence prediction
- Confidence calibration
- Adaptive Top-k retrieval
- Threshold optimization
- Feature importance analysis
- Ablation study
- Statistical significance testing

---

# Motivation

Conventional retrieval pipelines have several limitations:

- Fixed Top-k retrieval for every query
- No retrieval confidence estimation
- No adaptive retrieval policy
- No confidence calibration
- Limited statistical evaluation

This work addresses these limitations through a confidence-aware retrieval framework capable of dynamically selecting the appropriate retrieval depth.

---

# Research Contributions

This project introduces the following research extensions:

- Confidence-aware retrieval framework
- Retrieval Confidence Score (RCS)
- Confidence Retrieval Calibration Score (CRCS)
- Adaptive Top-k retrieval (Top-3, Top-5, Top-10)
- Confidence calibration using Isotonic Regression
- Automatic threshold optimization
- Feature importance analysis
- Bootstrap confidence intervals
- Wilcoxon signed-rank significance testing
- Comprehensive ablation study

---

# Methodology

The proposed pipeline follows the workflow below.

```
                    Query
                      │
                      ▼
             Similarity Search
                      │
                      ▼
          Top-k Similarity Scores
                      │
                      ▼
        Similarity Feature Extraction
                      │
                      ▼
        Confidence Prediction Model
                      │
                      ▼
       Confidence Calibration (CRCS)
                      │
                      ▼
         Adaptive Top-k Selection
                      │
                      ▼
          Retrieval Evaluation
                      │
                      ▼
        Threshold Optimization
                      │
                      ▼
          Statistical Analysis
```

---

# Similarity Features

The confidence model is trained using retrieval similarity statistics.

Extracted features include:

- Top-1 similarity
- Top-2 similarity
- Top-3 similarity
- Gap(Top1, Top2)
- Gap(Top1, Top3)
- Gap(Top2, Top3)
- Mean similarity
- Standard deviation
- Minimum similarity
- Maximum similarity
- Similarity range
- Coefficient of variation

---

# Adaptive Retrieval Policy

Retrieval depth is selected according to calibrated confidence.

| Confidence Level | Retrieved Documents |
|------------------|--------------------:|
| High Confidence | Top-3 |
| Medium Confidence | Top-5 |
| Low Confidence | Top-10 |

Thresholds are optimized using validation data.

---

# Experimental Results

| Method | Recall@1 | Recall@5 | Recall@10 | MRR | Average k |
|---------|---------:|---------:|----------:|----:|----------:|
| Fixed Top-5 | 0.4272 | 0.7384 | 0.8618 | 0.5605 | 5.00 |
| Rule-based Dynamic-k | 0.4284 | 0.7396 | 0.8625 | 0.5617 | 6.23 |
| Confidence Classifier | 0.4284 | 0.7396 | 0.8625 | 0.5617 | 6.20 |
| Confidence Calibration (CRCS) | 0.4284 | 0.7396 | 0.8625 | 0.5617 | 6.20 |
| **Final Adaptive Retrieval** | **0.4288** | **0.7380** | **0.8626** | **0.5576** | **6.21** |

---

# Threshold Optimization

Threshold optimization evaluates different confidence thresholds to balance retrieval effectiveness and retrieval cost.

Observed trends:

- Higher thresholds increase average retrieval depth.
- Lower thresholds reduce retrieval cost.
- Optimal thresholds maintain retrieval quality while dynamically adapting the retrieval depth.

---

# Feature Importance

The confidence model identifies the following features as the most informative.

| Feature | Importance |
|---------|-----------:|
| Minimum Similarity | 0.1509 |
| Coefficient of Variation | 0.1254 |
| Standard Deviation | 0.1131 |
| Similarity Range | 0.1056 |
| Mean Similarity | 0.0882 |

---

# Evaluation Metrics

Performance is evaluated using:

- Recall@1
- Recall@5
- Recall@10
- Mean Reciprocal Rank (MRR)

Statistical analysis includes:

- Bootstrap confidence intervals
- Wilcoxon signed-rank test
- Cohen's d effect size

---

# Repository Structure

```
Adaptive-TopK-Retrieval-for-Precision-Oncology-RAG/

│
├── notebooks/
│   └── Adaptive_TopK_RAG_Research.ipynb
│
├── research_extension/
│   ├── analysis/
│   ├── data/
│   ├── evaluators/
│   ├── extractors/
│   ├── indexes/
│   ├── loaders/
│   ├── models/
│   ├── optimization/
│   ├── pipeline/
│   └── utils/
│
├── figures/
│
├── results/
│   ├── FinalResults.csv
│   ├── Table1_MainResults.csv
│   ├── Table2_ThresholdOptimization.csv
│   ├── Table3_FeatureImportance.csv
│   ├── Table4_Ablation.csv
│   ├── Table5_StatisticalSignificance.csv
│   ├── adaptive_results_calibrated.csv
│   ├── merged_results.csv
│   └── similarity_analysis.csv
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

# Reproducing the Experiments

## Clone the repository

```bash
git clone https://github.com/Arnav-Singh-5080/Adaptive-TopK-Retrieval-for-Precision-Oncology-RAG.git
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run the notebook

Open

```
notebooks/Adaptive_TopK_RAG_Research.ipynb
```

and execute the cells sequentially.

---

# Results Directory

The repository includes all experimental outputs.

| File | Description |
|------|-------------|
| Table1_MainResults.csv | Overall retrieval performance |
| Table2_ThresholdOptimization.csv | Threshold optimization experiments |
| Table3_FeatureImportance.csv | Feature importance scores |
| Table4_Ablation.csv | Ablation study |
| Table5_StatisticalSignificance.csv | Statistical analysis |
| FinalResults.csv | Final evaluation metrics |
| adaptive_results_calibrated.csv | Adaptive retrieval outputs |
| merged_results.csv | Baseline vs adaptive comparison |
| similarity_analysis.csv | Similarity feature statistics |

---

# Technologies Used

- Python
- FAISS
- Sentence Transformers
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- SciPy

---

# Reproducibility

This repository includes:

- Source code
- Experimental notebook
- Evaluation scripts
- Dataset files
- FAISS index
- Confidence model
- Statistical analysis pipeline
- Experimental results

The complete experimental workflow can be reproduced using the provided notebook.

---

# Future Work

Possible future directions include:

- Learning adaptive thresholds automatically
- Reinforcement learning for retrieval policies
- Cross-dataset evaluation
- Hybrid sparse–dense retrieval
- End-to-end RAG generation optimization
- Clinical confidence calibration

---

# Acknowledgements

This work extends the open-source **Precision Oncology RAG** framework developed in the **rag-llm-cancer-paper** repository.

The original project provided the retrieval pipeline, benchmark datasets, and evaluation framework that served as the foundation for this research extension.

---

# License

This project is released under the MIT License.

---

# Author

**Arnav Singh**

Research Intern at
Indian Institute Of Information Technology, Design & Manufacturing, Kurnool

**GitHub:** https://github.com/Arnav-Singh-5080

**LinkedIn:** https://www.linkedin.com/in/arnav-singh-a87847351/

---

> **Note:** The primary contribution of this repository is the development and evaluation of a confidence-calibrated adaptive retrieval framework for precision oncology RAG. The reported experiments demonstrate that adaptive retrieval can maintain competitive retrieval performance while dynamically selecting retrieval depth based on calibrated confidence estimates.
