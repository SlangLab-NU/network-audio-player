FROM python:3.11-slim

WORKDIR /app
COPY . /app

# Install CPU-only torch first so silero-vad doesn't pull the 4 GB CUDA build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Start the server (replace 'app:app' if your callable is named differently)
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]