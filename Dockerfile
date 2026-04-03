FROM python:3.11-slim

WORKDIR /app
COPY . /app

# Install CPU-only torch (no torchaudio — we use stdlib wave for audio I/O)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install silero-vad without torchaudio dep (our code bypasses read_audio)
RUN pip install --no-cache-dir silero-vad --no-deps

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Start the server (replace 'app:app' if your callable is named differently)
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]