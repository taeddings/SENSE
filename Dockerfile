FROM python:3.12-slim

# Install system deps
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY SENSE/ .

# Expose ports for dashboard/server
EXPOSE 8501 8000

# Default to main loop
CMD ["python", "sense/main.py"]