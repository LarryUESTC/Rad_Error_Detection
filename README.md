# Radiology Report Error Detection with Large Language Models

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)  
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/downloads/)  

This repository contains the official implementation for the paper:  
**"Real-World Validation of Large Language Model for Radiology Report Error Detection Across Multimodal and Multisite Clinical Settings"**

We present a large-scale, real-world evaluation of **LLMs** for automated detection of clinically significant errors in free-text radiology reports across **X-ray, CT, and MRI** modalities. Our model achieves **0.96 accuracy**, outperforms radiologists of varying experience levels, and processes reports in **<6 seconds**—demonstrating strong potential for clinical deployment in quality assurance workflows.

---

## 📌 Key Features

- ✅ **Multimodal error detection**: Supports X-ray, CT, and MRI reports.
- ✅ **Fine-grained error taxonomy**: Detects 5 error types grouped into:
  - **Type 1 (Interpretive)**: Omission (1a), Inconsistency (1b)
  - **Type 2 (Factual)**: Laterality confusion (2a), Semantic errors (2b), Other technical errors (2c)
- ✅ **Cross-center generalization**: Validated on an independent multi-institutional dataset (1,000 reports).
- ✅ **Clinically interpretable reasoning**: Generates human-readable justifications for detected errors (mean Likert score: **4.57/5**).
- ✅ **High-throughput & real-time**: Processes one report in **~5.8 seconds** on standard hardware.

---

## 📁 Repository Structure

```
.
├── data/                   # (Placeholder) Data loading & preprocessing scripts
├── models/                 # Model inference pipeline for Qwen 3.0
├── eval/                   # Evaluation scripts (accuracy, F1, cross-center validation)
├── human_ai_benchmark/     # Code for human vs. LLM comparison
├── reasoning/              # Modules for generating & evaluating error rationales
├── utils/                  # Helper functions (metrics, logging, etc.)
├── requirements.txt        # Python dependencies
├── run_inference.py        # Main script to run error detection on reports
├── run_evaluation.py       # Reproduce main results & figures
└── README.md
```

> 🔒 **Note**: Due to privacy and regulatory constraints, **real clinical data cannot be shared publicly**. Researchers may contact the corresponding author for collaborative data access (see paper for details).

---


## 📄 License

This project is licensed under the **Apache License 2.0** – see [LICENSE](LICENSE) for details.

