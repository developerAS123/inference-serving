"""
One-time setup: exports the classifier to ONNX (dynamic shapes) and quantizes
it to INT8. Run this before starting the API with onnx-fp32/onnx-int8 modes.

Usage: uv run scripts/export_models.py
"""
import json
import os

import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
OUT_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

inputs = tokenizer("Exporting this model for serving.", return_tensors="pt")

print("Exporting to ONNX (dynamic batch + sequence length)...")
fp32_path = os.path.join(OUT_DIR, "model_fp32.onnx")
torch.onnx.export(
    model,
    (inputs["input_ids"], inputs["attention_mask"]),
    fp32_path,
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "logits": {0: "batch_size"},
    },
    opset_version=17,
    dynamo=False,
)
onnx.checker.check_model(onnx.load(fp32_path))
print(f"Saved + validated: {fp32_path}")

print("Quantizing to INT8...")
int8_path = os.path.join(OUT_DIR, "model_int8.onnx")
quantize_dynamic(model_input=fp32_path, model_output=int8_path, weight_type=QuantType.QInt8)
print(f"Saved: {int8_path}")

with open(os.path.join(OUT_DIR, "id2label.json"), "w") as f:
    json.dump(model.config.id2label, f)

print("\nDone. models/ now has model_fp32.onnx, model_int8.onnx, id2label.json")
