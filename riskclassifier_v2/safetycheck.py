import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from importlib import import_module


class SafetyCheck:
    """
    SafetyCheck(text, threshold=0.5)
    ------------------------------------------------
    Ensemble of:
      • mental/mental-roberta-base
      • mental/mental-bert-base-uncased

    Methods
    --------
    __call__(text, threshold=0.5, verbose=False)
        Runs inference and returns both prediction and probabilities.
    """

    def __init__(self, base_dir: str = None):
        """
        Initializes the SafetyCheck ensemble.
        Automatically checks for model folders and downloads them if missing.
        """
        if base_dir is None:
            base_dir = os.path.join(os.path.dirname(__file__), "saved_models")

        self.base_dir = base_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model paths
        self.roberta_path = os.path.join(base_dir, "mental_mental-roberta-base")
        self.bert_path = os.path.join(base_dir, "mental_mental-bert-base-uncased")

        print(f"[SafetyCheck] Using base directory: {self.base_dir}")
        self._ensure_models_exist()

        # Load both models
        self.roberta_model, self.roberta_tok = self._load_model(self.roberta_path)
        self.bert_model, self.bert_tok = self._load_model(self.bert_path)

    # ------------------------------------------------------------
    def _ensure_models_exist(self):
        """
        Ensures that the fine-tuned models exist locally.
        If not found, automatically downloads them using download_models.py.
        """
        if not (os.path.exists(self.roberta_path) and os.path.exists(self.bert_path)):
            print("Model folders not found. Running download_models.py...")
            downloader = import_module("riskclassifier_v2.download_models")
            downloader.download_and_extract()
        else:
            print("All model folders found locally.")

    # ------------------------------------------------------------
    def _load_model(self, path: str):
        """Loads a fine-tuned model and its tokenizer from the given path."""
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        model.to(self.device).eval()
        print(f"Loaded model from {path}")
        return model, tokenizer

    # ------------------------------------------------------------
    def _predict_probs(self, text: str, model, tokenizer):
        """Generates class probabilities for a single text input."""
        encoded = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        return probs

    # ------------------------------------------------------------
    def __call__(self, text: str, threshold: float = 0.5, verbose: bool = False):
        """
        Performs ensemble prediction for the input text.
        Averages 'at risk' probabilities from both models and applies thresholding.

        Returns:
            dict → {
                "prediction": int,
                "probs": {
                    "not_at_risk": float,
                    "at_risk": float
                }
            }
        """
        probs_roberta = self._predict_probs(text, self.roberta_model, self.roberta_tok)
        probs_bert = self._predict_probs(text, self.bert_model, self.bert_tok)

        avg_prob_1 = np.mean([probs_roberta[1], probs_bert[1]])
        avg_prob_0 = 1 - avg_prob_1
        pred = int(avg_prob_1 >= threshold)

        if verbose:
            print(f"[Input] {text}")
            print(f"Roberta → [0={probs_roberta[0]:.4f}, 1={probs_roberta[1]:.4f}]")
            print(f"BERT    → [0={probs_bert[0]:.4f}, 1={probs_bert[1]:.4f}]")
            print(f"Avg     → [not at risk={avg_prob_0:.4f}, at risk={avg_prob_1:.4f}]")
            print(f"Prediction: {'at risk' if pred else 'not at risk'}")

        return {
            "prediction": pred,
            "probs": {
                "not_at_risk": float(avg_prob_0),
                "at_risk": float(avg_prob_1)
            }
        }
