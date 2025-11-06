# ============================================================
# SafetyCheck — Offline Ensemble Risk Classifier (V4)
# ============================================================

import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ------------------------------------------------------------
# Enforce OFFLINE MODE for reproducibility
# ------------------------------------------------------------
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

torch.manual_seed(42)
np.random.seed(42)
torch.use_deterministic_algorithms(True, warn_only=True)
torch.cuda.empty_cache()


# ============================================================
# SAFETYCHECK CLASS (ENSEMBLE)
# ============================================================
class SafetyCheck:
    """
    SafetyCheck(text) → returns both:
        • prediction (0 = not at risk, 1 = at risk)
        • probabilities for each class

    Ensemble of:
      • finetuned_mentalRoBERTa
      • finetuned_mentalBERT
    """

    def __init__(self, base_dir="saved_models"):
        """Load both fine-tuned models from the local saved_models directory."""
        self.base_dir = base_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Local model paths
        self.roberta_path = os.path.join(base_dir, "mentalroberta")
        self.bert_path = os.path.join(base_dir, "mentalbert")

        print(f"[SafetyCheck] Using base directory: {self.base_dir}")
        self.roberta_model, self.roberta_tok = self._load_model(self.roberta_path)
        self.bert_model, self.bert_tok = self._load_model(self.bert_path)

    # ------------------------------------------------------------
    def _load_model(self, path):
        """Load model and tokenizer fully offline."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model folder not found: {path}")

        tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True)
        model.to(self.device).eval()
        print(f"Loaded model from {path}")
        return model, tok

    # ------------------------------------------------------------
    def _predict_probs(self, text, model, tokenizer):
        """Return softmax probabilities for a single model."""
        enc = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        return probs

    # ------------------------------------------------------------
    def __call__(self, text, threshold=0.5, verbose=True):
        """Run ensemble prediction and return dict with pred + probs."""
        # Get probabilities from both models
        probs_roberta = self._predict_probs(text, self.roberta_model, self.roberta_tok)
        probs_bert = self._predict_probs(text, self.bert_model, self.bert_tok)

        # Average the "at risk" probabilities
        avg_prob_1 = np.mean([probs_roberta[1], probs_bert[1]])
        avg_prob_0 = 1 - avg_prob_1
        pred = int(avg_prob_1 >= threshold)

        if verbose:
            print(f"\ntext: {text}")
            print(f"roberta → [0={probs_roberta[0]:.4f}, 1={probs_roberta[1]:.4f}]")
            print(f"bert    → [0={probs_bert[0]:.4f}, 1={probs_bert[1]:.4f}]")
            print(f"avg     → [not at risk={avg_prob_0:.4f}, at risk={avg_prob_1:.4f}]")
            print(f"prediction: {'at risk' if pred else 'not at risk'}")

        return {
            "prediction": pred,
            "probs": {
                "not_at_risk": float(avg_prob_0),
                "at_risk": float(avg_prob_1)
            }
        }
