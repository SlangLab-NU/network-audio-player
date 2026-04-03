FROM python:3.11-slim

WORKDIR /app
COPY . /app

# Install CPU-only torch + torchaudio first so silero-vad can't pull CUDA builds
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install silero-vad — torch/torchaudio already present so pip won't re-pull CUDA builds
RUN pip install --no-cache-dir silero-vad

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Start the server (replace 'app:app' if your callable is named differently)
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]