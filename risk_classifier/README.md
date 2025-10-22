# Risk Classifier

This folder contains the **SafetyCheck** module used in the mental health chatbot.  
It classifies user text as **“at risk” (1)** or **“not at risk” (0)** using fine-tuned transformer models.

---

## Files

| File | Description |
|------|--------------|
| `safetycheck.py` | Main class that loads the models and runs predictions. |
| `test_safetycheck.py` | Quick script to test if SafetyCheck works correctly. |
| `download_models.py` | Downloads the trained models from Google Drive and extracts them into `saved_models/`. |
| `requirements.txt` | Lists all Python packages needed to run this module. |
| `safetycheckmodels.py` | Model testing, finetuning and training. |
| `results/` | Saved CSVs and plots from testing. |

---

## Basic Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
2. Download models:
   ```bash
   python download_models.py
3. Run quick test:
   ```bash
   python test_safetycheck.py

# Expected output
text: I feel hopeless and want to give up.
prediction: at risk
Returned: 1


text: Today was a beautiful day with my family.
prediction: not at risk
Returned: 0

# Notes
* Models are downloaded from Google Drive and saved in saved_models/.
* Runs completely offline after download.
* Outputs are deterministic (no random variation between runs).
