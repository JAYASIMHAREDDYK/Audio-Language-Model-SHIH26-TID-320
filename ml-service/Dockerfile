# ==============================================================================
# VOX ML Service — Production Dockerfile for Render / Cloud Deployment
# ==============================================================================

FROM python:3.11-slim

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system audio libraries and build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency definition
COPY requirements.txt .

# Install CPU-optimized PyTorch first for fast, lightweight cloud build
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install remaining ML and API dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and configurations
COPY . .

# Default port for Render (Render dynamically sets $PORT)
ENV PORT=8000

EXPOSE 8000

# Start FastAPI server via Uvicorn
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
