# Use official lightweight Python image
FROM python:3.10.8-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create app directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
# Ensure requirements.txt exists before copying
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

CMD [ "bash" ]
