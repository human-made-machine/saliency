# Dockerfile for Vertex AI GPU training
# Base image: TensorFlow 2.15 GPU with Python 3.10
FROM us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-15.py310:latest

WORKDIR /app

# Copy requirements and install additional dependencies
COPY pyproject.toml .

# Install uv for faster dependency resolution
RUN pip install uv

# Install project dependencies (excluding tensorflow as it's in base image)
RUN uv pip install --system gdown>=5.0.0 h5py>=3.10.0 imageio>=2.33.0 \
    matplotlib>=3.8.0 numpy>=1.26.0 requests>=2.31.0 scipy>=1.11.0

# Copy source code
COPY *.py ./
COPY weights/ ./weights/ 2>/dev/null || true

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default entrypoint for training
ENTRYPOINT ["python", "main.py"]
CMD ["train"]
