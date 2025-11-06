# -*- coding: utf-8 -*-
import os
import random
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from peft import PeftModel

# ============================================================
# OFFLINE MODE + REPRODUCIBILITY
# ============================================================
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.use_deterministic_algorithms(True, warn_only=True)
torch.cuda.empty_cache()

# ============================================================
# SAFETYCHECK ENSEMBLE
# ============================================================
class SafetyCheck:
    """
    SafetyCheck(text) → returns (pred, prob)

    Ensemble of:
      - MentalRoBERTa (fine-tuned)
      - MentalBERT (fine-tuned)
    """

    def __init__(self, base_dir="saved_models"):
        """Loads both models locally and prepares ensemble."""
        self.base_dir = base_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # model paths
        self.roberta_path = os.path.join(base_dir, "mental_mental-roberta-base")
        self.bert_path = os.path.join(base_dir, "mental_mental-bert-base-uncased")

        print(f"using base directory: {self.base_dir}")
        print("loading model 1 (MentalRoBERTa)...")
        self.model1, self.tok1 = self._load_model(self.roberta_path)

        print("loading model 2 (MentalBERT)...")
        self.model2, self.tok2 = self._load_model(self.bert_path)

        # load thresholds
        self.thr1 = self._load_threshold(self.roberta_path)
        self.thr2 = self._load_threshold(self.bert_path)
        print(f"loaded thresholds → roberta={self.thr1:.3f}, bert={self.thr2:.3f}\n")

    # ------------------------------------------------------------
    def _load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"model folder not found: {model_path}")

        print(f"   loading from: {model_path}")
        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

        model_type = getattr(config, "model_type", "roberta")
        base_model_id = {
            "roberta": "roberta-base",
            "bert": "bert-base-uncased"
        }.get(model_type, "roberta-base")

        base = AutoModelForSequenceClassification.from_pretrained(
            base_model_id,
            num_labels=getattr(config, "num_labels", 2),
            local_files_only=False,
            ignore_mismatched_sizes=True
        )

        adapter_cfg = os.path.join(model_path, "adapter_config.json")
        adapter_weights = os.path.join(model_path, "adapter_model.safetensors")
        if os.path.exists(adapter_cfg) and os.path.exists(adapter_weights):
            print("   applying LoRA adapter weights...")
            model = PeftModel.from_pretrained(base, model_path, local_files_only=True, is_trainable=False)
        else:
            print("   no adapter found, using base model only")
            model = base

        model.to(self.device)
        model.eval()
        print("   model loaded successfully\n")
        return model, tokenizer

    # ------------------------------------------------------------
    def _load_threshold(self, model_path):
        meta_path = os.path.join(model_path, "best_meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    data = json.load(f)
                    return float(data.get("chosen_threshold", 0.5))
            except Exception:
                pass
        return 0.5

    # ------------------------------------------------------------
    def _predict_single(self, text, model, tokenizer):
        """Get raw probability from one model."""
        enc = tokenizer(text, truncation=True, padding=True, max_length=256, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        return probs[1]  # prob of "at risk"

    # ------------------------------------------------------------
    def __call__(self, text):
        """Run ensemble inference and return (pred, prob)."""
        prob1 = self._predict_single(text, self.model1, self.tok1)
        prob2 = self._predict_single(text, self.model2, self.tok2)
        avg_prob = (prob1 + prob2) / 2.0

        pred = int(avg_prob >= 0.5)  # fixed ensemble threshold
        print(f"text: {text}")
        print(f"probabilities: [MentalRoBERTa={prob1:.4f}, MentalBERT={prob2:.4f}, ensemble_avg={avg_prob:.4f}]")
        print(f"prediction: {'at risk' if pred == 1 else 'not at risk'}\n")

        return pred, avg_prob
