# Risk Classifier

This folder contains the **SafetyCheck** module used in the mental health chatbot.  
It classifies user text as **“at risk” (1)** or **“not at risk” (0)** using fine-tuned transformer models.

---

## Files

| File | Description |
|------|--------------|
| `safetycheck.py` | Main class that loads the models and runs predictions. |
| `download_models.py` | Downloads the trained models from Google Drive and extracts them into `saved_models/`. |
| `requirements.txt` | Lists all Python packages needed to run this module. |
| `safetycheckmodels.py` | Model testing, finetuning and training. |
| `phase3_safetycheckvariant.py` | Testing different variants of riskclassifier. |
| `results/` | Saved CSVs and plots from testing. |

---

## Basic Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
2. Download models:
   ```bash
   python download_models.py

# Expected output
```
text: I feel like nothing matters anymore.
/n probabilities: [not at risk=0.6078, at risk=0.3922]
prediction: not at risk
returned value: 0

text: im so tired i want to sleep forever
probabilities: [not at risk=0.3816, at risk=0.6184]
prediction: at risk
returned value: 1
```

# Notes
* Models are downloaded from Google Drive and saved in `saved_models/`.
* Runs completely offline after download.
* Classification threshold is 0.5 by default

