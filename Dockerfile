FROM python:3.10-slim

# Optional: prevent interactive prompts during installs
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy your code
COPY simdistserve/ /app/simdistserve/

# # Install Python dependencies
RUN pip install --no-cache-dir -r /app/simdistserve/requirements.txt
ENV PYTHONPATH="/app/simdistserve:${PYTHONPATH}"

ENTRYPOINT ["python", "/app/simdistserve/main.py"]