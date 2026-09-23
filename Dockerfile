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

# RUN dvc remote add -d storage s3://nst-dvc-models --force && \
#     dvc remote modify storage endpointurl https://s3.eu-central-003.backblazeb2.com && \
#     dvc remote modify storage access_key_id "$DVC_ACCESS_KEY_ID" && \
#     dvc remote modify storage secret_access_key "$DVC_SECRET_ACCESS_KEY" && \
#     dvc pull

# Pull models from DVC using Hugging Face secrets
RUN --mount=type=secret,id=DVC_ACCESS_KEY_ID,mode=0444,required=true \
    --mount=type=secret,id=DVC_SECRET_ACCESS_KEY,mode=0444,required=true \
    DVC_ACCESS_KEY_ID="$(cat /run/secrets/DVC_ACCESS_KEY_ID)" && \
    DVC_SECRET_ACCESS_KEY="$(cat /run/secrets/DVC_SECRET_ACCESS_KEY)" && \
    dvc remote add -d storage s3://nst-dvc-models --force && \
    dvc remote modify storage endpointurl https://s3.eu-central-003.backblazeb2.com && \
    dvc remote modify storage access_key_id "$DVC_ACCESS_KEY_ID" && \
    dvc remote modify storage secret_access_key "$DVC_SECRET_ACCESS_KEY" && \
    dvc pull

# Expose the port HF Spaces expects
EXPOSE 7860

# Run using our new entry point script
CMD ["python", "run.py"]
