# Base Python Image
FROM python:3.11-slim

# Working directory
WORKDIR /app

# Prevent bytecode & buffer stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

RUN useradd -m -r appuser && chown -R appuser:appuser /app
USER appuser

# Expose FastAPI Web Port
EXPOSE 8000

# Healthcheck endpoint
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/status')" || exit 1

# Launch FastAPI Application
ENTRYPOINT ["python", "server.py"]
