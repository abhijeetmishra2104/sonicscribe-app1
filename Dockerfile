FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5001

# --timeout: a single /analyze-note request makes three Gemini calls; the 30s
# default SIGKILLs the worker mid-request and gunicorn answers with its own
# HTML error page, which the JSON error handlers in app.py never get to see.
# Keep it above app.py's GEMINI_TIMEOUT so our error wins the race.
CMD ["gunicorn", "--bind", "0.0.0.0:5001", \
     "--timeout", "180", \
     "--graceful-timeout", "30", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--log-level", "info", \
     "app:app"]