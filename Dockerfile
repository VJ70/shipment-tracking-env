FROM python:3.11-slim

# Non-root user for HF Spaces compatibility
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=appuser:appuser . .

# HF Spaces requires port 7860
EXPOSE 7860

USER appuser

# Start the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]