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

        onnx_fp32 = ONNXBackend("onnx-fp32", "models/model_fp32.onnx", pt.tokenizer)
        self.backends[onnx_fp32.name] = onnx_fp32
        onnx_int8 = ONNXBackend("onnx-int8", "models/model_int8.onnx", pt.tokenizer)
        self.backends[onnx_int8.name] = onnx_int8
        logger.info(f"ModelManager ready. Backends: {list(self.backends.keys())}")
        
    def predict(self, text: str, mode: str):
        if mode not in self.backends:
            raise ValueError(f"Unknown mode '{mode}'. Available: {list(self.backends.keys())}")
        return self.backends[mode].predict(text)

    def available_modes(self):
        return list(self.backends.keys())
    
import json
import os

import numpy as np
import onnxruntime as ort


class ONNXBackend:
    """Shared by both onnx-fp32 and onnx-int8 — same logic, different file.
    Deliberately pure numpy — no torch dependency in this code path at all."""

    def __init__(self, name: str, onnx_path: str, tokenizer):
        self.name = name
        logger.info(f"[{self.name}] loading {onnx_path}...")
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"{onnx_path} not found — run `uv run scripts/export_models.py` first.")
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.tokenizer = tokenizer
        with open(os.path.join(os.path.dirname(onnx_path), "id2label.json")) as f:
            self.id2label = {int(k): v for k, v in json.load(f).items()}
        logger.info(f"[{self.name}] ready.")

    def predict(self, text: str):
        inputs = self.tokenizer(text, return_tensors="np", truncation=True, max_length=128)
        feed = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
        logits = self.session.run(None, feed)[0][0]
        exp = np.exp(logits - np.max(logits))
        probs = exp / exp.sum()
        pred_idx = int(np.argmax(probs))
        return self.id2label[pred_idx], float(probs[pred_idx])