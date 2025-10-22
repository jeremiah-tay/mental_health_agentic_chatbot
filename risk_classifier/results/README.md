# Results

This folder contains all evaluation outputs from model testing — including CSV summaries and visual plots from both **Phase 1 (Model Selection)** and **Phase 2 (Hyperparameter Tuning & Final Evaluation)**.

---

## 🧩 Experiment Overview

### **Phase 1 – Model Selection**
- Used **5-fold cross-validation** across multiple model and feature setups.  
- Goal: Identify top-performing base models using **macro-F1** and **macro-recall** metrics.  
- Output: `phase1_model_selection.csv` and corresponding ROC / confusion matrix plots.

### **Phase 2 – Nested Cross-Validation (5×3)**
- Used **5 outer folds × 3 inner folds** for hyperparameter tuning and evaluation.  
- Inner folds → find best hyperparameters (learning rate, LoRA rank, dropout).  
- Outer folds → evaluate final tuned model performance.  
- Output: `phase2_summary_metrics.csv`, `phase2_best_hyperparams_per_combo.csv`, and final plots.

---

## File Flow
Phase 1: Model Selection
* phase1_model_selection.csv
↓
Phase 2: Nested CV & Hyperparameter Tuning
├── phase2_best_hyperparams_per_combo.csv
├── phase2_nestedcv_top2_fastgrid.csv
└── phase2_summary_metrics.csv
↓
Final Evaluation
↓
final_top2_nestedcv.csv


## CSV Files

| File | Description |
|------|--------------|
| `phase1_model_selection.csv` | Results of 5-fold CV across all model + feature combinations. |
| `phase2_best_hyperparams_per_combo.csv` | Best hyperparameters found for each shortlisted model. |
| `phase2_nestedcv_top2_fastgrid.csv` | Grid search results for top 2 model candidates during Phase 2. |
| `phase2_summary_metrics.csv` | Summary of nested CV results with macro-F1, macro-recall, precision, and ROC-AUC. |
| `final_top2_nestedcv.csv` | Final averaged metrics for the two selected models (used in `SafetyCheck.py`). |

---

## Plots

| File | Description |
|------|--------------|
| `phase1_roberta__lora-finetune_roc_auc.png` | ROC curve for RoBERTa (LoRA) during Phase 1 model selection. |
| `phase1_roberta__lora-finetune_cm.png` | Confusion matrix for RoBERTa (LoRA) – Phase 1. |
| `phase1_mentalroberta__lora-finetune_roc_auc.png` | ROC curve for MentalRoBERTa (LoRA) – Phase 1. |
| `phase1_mentalroberta__lora-finetune_cm.png` | Confusion matrix for MentalRoBERTa (LoRA) – Phase 1. |
| `phase2_roberta__lora-finetune_cm.png` | Confusion matrix for RoBERTa (LoRA) after Phase 2 tuning. |
| `phase2_mentalroberta__lora-finetune_cm.png` | Confusion matrix for MentalRoBERTa (LoRA) after Phase 2 tuning. |

---

## Notes

- **Phase 1** = Model selection (5-fold CV)  
- **Phase 2** = Nested CV (5×3) for hyperparameter tuning and final evaluation  
- Metrics used: **macro-F1**, **macro-recall**, **macro-precision**, **macro-accuracy** and **ROC-AUC**  
- Final models used in `SafetyCheck.py`:  
  - **Primary:** RoBERTa (LoRA fine-tuned)  
  - **Fallback:** MentalRoBERTa (LoRA fine-tuned)
