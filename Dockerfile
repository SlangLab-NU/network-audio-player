FROM python:3.11-slim

WORKDIR /app
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir gunicorn && \
    if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Start the server (replace 'app:app' if your callable is named differently)
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]