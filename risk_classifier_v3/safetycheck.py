import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

#enforce offline mode
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

#reproducibility
torch.manual_seed(42)
np.random.seed(42)
torch.use_deterministic_algorithms(True, warn_only=True)
torch.cuda.empty_cache()

# ============================================================
# SAFETYCHECK CLASS (FINE-TUNED ROBERTA MODEL)
# ============================================================
class SafetyCheck:
    """
    SafetyCheck(text) → returns:
        • prediction (0 = not at risk, 1 = at risk)
        • probability (float, probability of at risk class, rounded to 5 decimal places)
    Uses fine-tuned RoBERTa model trained on suicide risk detection dataset
    """
    def __init__(self, model_path=None, base_dir=None):
        """Load fine-tuned RoBERTa model from the local directory."""
        #handle legacy base_dir parameter for backward compatibility
        if base_dir is not None and model_path is None:
            model_path = os.path.join(base_dir, "roberta-base")
        
        #if no path provided, use the path relative to this script
        if model_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, "saved_models", "roberta-base")
        
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"[SafetyCheck] loading model from: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"model folder not found: {self.model_path}")
        
        #load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path, local_files_only=True)
        self.model.to(self.device).eval()
        
        print(f"[SafetyCheck] model loaded successfully on {self.device}")
    
    # ------------------------------------------------------------
    def _predict_probs(self, text):
        """return softmax probabilities."""
        enc = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
        return probs
    
    # ------------------------------------------------------------
    def __call__(self, text, threshold=0.5, verbose=True):
        """run prediction and return prediction label and probability."""
        #get probabilities
        probs = self._predict_probs(text)
        
        prob_not_at_risk = probs[0]
        prob_at_risk = probs[1]
        
        pred = int(prob_at_risk >= threshold)
        
        if verbose:
            print(f"\ntext: {text}")
            print(f"probabilities → [not at risk={prob_not_at_risk:.4f}, at risk={prob_at_risk:.4f}]")
            print(f"prediction: {'at risk' if pred else 'not at risk'}")
        
        return pred, round(float(prob_at_risk), 5)
