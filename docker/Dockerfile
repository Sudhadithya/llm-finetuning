# Use PyTorch with CUDA as base image
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Set non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire project repository
COPY . .

# Expose API port
EXPOSE 8000

# Set default command for starting the API
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
