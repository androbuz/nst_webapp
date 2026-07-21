FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Pull data from DVC (using HF Secrets as Build Args)
ARG DVC_ACCESS_KEY_ID
ARG DVC_SECRET_ACCESS_KEY

RUN dvc remote modify storage access_key_id $DVC_ACCESS_KEY_ID && \
    dvc remote modify storage secret_access_key $DVC_SECRET_ACCESS_KEY && \
    dvc pull

# Expose the port HF Spaces expects
EXPOSE 7860

# Run using our new entry point script
CMD ["python", "run.py"]
