# Inference Optimization & Serving Service — Architecture

## Goal
A FastAPI service serving a sentiment classifier through three interchangeable
optimization backends (PyTorch-FP32, ONNX-FP32, ONNX-INT8), selectable per
request, with latency logging, a load-testing tool, and a benchmark dashboard
comparing all three — fully containerized.

## Architecture

    [Client] --HTTP--> [FastAPI app] --routes by `mode`--> [Model Manager]
                                                                |-- PyTorch-FP32 (transformers)
                                                                |-- ONNX-FP32 (onnxruntime)
                                                                |-- ONNX-INT8 (onnxruntime, quantized)
                             |
                             v
                    [Structured logs: mode, latency, timestamp]
                             |
                             v
           [Load-test script] --generates--> [Benchmark dashboard: table + charts]

Everything above runs inside one Docker container (Day 48). The load-test
script and dashboard run as separate, on-demand tooling against the running
container — not bundled inside it.

## Endpoints (built Day 44-47)
- `POST /predict {text, mode}` -> {label, confidence, latency_ms, mode}
- `GET /health` -> readiness check, confirms all 3 backends loaded
- `GET /stats` -> aggregated P50/P99 latency per mode, from logged requests
- `GET /dashboard` -> HTML page visualizing /stats

## Model Manager design
- All 3 backends loaded ONCE at startup (Day 36's lifespan pattern), never per-request
- Each backend exposes the same interface: `predict(text) -> (label, confidence)`
- `mode` in the request body selects which backend handles that call
- PyTorch-FP32: `AutoModelForSequenceClassification` directly
- ONNX-FP32 / ONNX-INT8: `onnxruntime.InferenceSession` on files produced by
  `scripts/export_models.py` (reuses Day 16 export + Day 19 quantization code)

## Explicit scoping decision
No vLLM in the core build — this is a classification service, not LLM
generation; vLLM's optimizations don't apply here, and adding a second server
process would add deployment risk without serving the project's actual goal.
[Optional stretch, only after Day 49's core is solid: a 4th mode proxying to a
separately-run vLLM server for a generative endpoint, clearly separated from
the core deliverable.]

## Week plan
- Day 43 (today): this document, environment set up
- Day 44: Model Manager + /predict, all 3 backends working correctly
- Day 45: /stats endpoint, per-mode latency tracking
- Day 46: load-testing script (async concurrent requests, Day 40's pattern),
  results table + chart
- Day 47: /dashboard HTML page, structured logging to file
- Day 48: Dockerize, polish README, attempt a public demo deploy
- Day 49: final testing, capstone announcement post

## Success criteria
- All 3 modes return correct, consistent predictions for the same input
- Load test produces a real, credible latency/throughput comparison across modes
- Entire service runs via a single `docker run` command
- README lets a stranger get it running in under 5 minutes