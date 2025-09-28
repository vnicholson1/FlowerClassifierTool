# Use official Python image
FROM python:3.10-slim

# Install system dependencies for OpenCV and SIFT
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        gfortran \
        && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy only Python files, the features and templates folder
COPY *.py ./
COPY templates ./templates/
COPY data ./data/
COPY bovw_kmeans.pkl ./
COPY training_features.json ./

# Expose Flask port
EXPOSE 4000

# Run the Flask app
CMD ["python", "app.py"]