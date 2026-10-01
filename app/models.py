import logging

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger("capstone")
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


class PyTorchBackend:
    """FP32 baseline. ONNX-FP32 and ONNX-INT8 backends (Day 45) will implement
    this exact same predict(text) -> (label, confidence) interface."""

    name = "pytorch-fp32"

    def __init__(self):
        logger.info(f"[{self.name}] loading {MODEL_NAME}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        self.model.eval()
        logger.info(f"[{self.name}] ready.")

    def predict(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred_idx = int(torch.argmax(probs))
        label = self.model.config.id2label[pred_idx]
        confidence = float(probs[pred_idx])
        return label, confidence


class ModelManager:
    """Owns every backend. Today: pytorch-fp32 only. Day 45 registers two
    more here — the rest of the app never needs to know which backend
    actually ran, only that it matches this interface."""

    def __init__(self):
        self.backends = {}

    def load_all(self):
        pt = PyTorchBackend()
        self.backends[pt.name] = pt
        logger.info(f"ModelManager ready. Backends: {list(self.backends.keys())}")

    def predict(self, text: str, mode: str):
        if mode not in self.backends:
            raise ValueError(f"Unknown mode '{mode}'. Available: {list(self.backends.keys())}")
        return self.backends[mode].predict(text)

    def available_modes(self):
        return list(self.backends.keys())