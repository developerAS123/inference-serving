FROM python:3.12-slim

# Hugging # Non-root user (UID 1000): good container hygiene, and compatible with hosts that require it
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# CPU-only PyTorch. This service runs on CPU, and the default PyPI wheel drags
# in gigabytes of CUDA libraries that would never be used.
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user app/ ./app/
COPY --chown=user scripts/export_models.py ./scripts/export_models.py

# Export + quantize at BUILD time: the image is self-contained and the
# container starts without redoing this one-time work on every launch
RUN python scripts/export_models.py

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]