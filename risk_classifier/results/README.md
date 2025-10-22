# Results

This folder stores all CSV outputs from the model testing, finetuning and evaluation process.

##

| File | Description |
|------|--------------|
| `phase1_model_selection.csv` | Cross-validation results for all base model + feature combinations (used to identify top-performing setups). |
| `phase2_best_hyperparams_per_combo.csv` | Stores the best hyperparameters (e.g. learning rate, LoRA rank, dropout) found for each top model in nested CV. |
| `phase2_nestedcv_top2_fastgrid.csv` | Detailed grid search results for the top 2 models with corresponding best hyperparameters during Phase 2. |
| `phase2_summary_metrics.csv` | Summary of Phase 2 performance across all folds — includes macro-F1, macro-recall, precision, and ROC-AUC. |
| `final_top2_nestedcv.csv` | Final averaged metrics for the top 2 selected models after nested cross-validation. |

---

## Notes
- All metrics are computed using 5-fold nested cross-validation.  
- The final selected models were **RoBERTa (LoRA)** and **MentalRoBERTa (LoRA)** based on macro-F1 and recall performance.  
- These results were used to determine the final primary and fallback models in `SafetyCheck.py`.

