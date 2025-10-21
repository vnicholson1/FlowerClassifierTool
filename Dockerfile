# Use Python 3.11 slim for ARM (Debian-based)
FROM --platform=linux/arm/v7 python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose Flask port
EXPOSE 4000

# Run the Flask app
CMD ["python3", "app.py"]
