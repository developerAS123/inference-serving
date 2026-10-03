import random
from locust import HttpUser, task, between

SAMPLE_TEXTS = [
    "This capstone project is coming together nicely.",
    "The service is slow and keeps returning errors.",
]

class CapstoneUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def predict_onnx_int8(self):
        self.client.post("/predict", json={"text": random.choice(SAMPLE_TEXTS), "mode": "onnx-int8"})