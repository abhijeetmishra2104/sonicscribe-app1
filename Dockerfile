FROM python:3.10-slim

# Prevents Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git curl libsndfile1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Create uploads folder
RUN mkdir -p uploads

# Expose Flask/Gunicorn port
EXPOSE 8080

# Start app using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
