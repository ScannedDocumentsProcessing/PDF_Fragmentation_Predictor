# Base image
FROM python:3.10

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Work directory
WORKDIR /app

# Copy requirements file
COPY ./requirements.txt .

# Copy sources
COPY src src

COPY model/pdf_fragmentation_classifier.bentomodel /tmp/pdf_fragmentation_classifier.bentomodel

# Environment variables
ENV ENVIRONMENT=${ENVIRONMENT}
ENV LOG_LEVEL=${LOG_LEVEL}
ENV ENGINE_URL=${ENGINE_URL}
ENV MAX_TASKS=${MAX_TASKS}
ENV ENGINE_ANNOUNCE_RETRIES=${ENGINE_ANNOUNCE_RETRIES}
ENV ENGINE_ANNOUNCE_RETRY_DELAY=${ENGINE_ANNOUNCE_RETRY_DELAY}

# Exposed ports
EXPOSE 80

# Switch to src directory
WORKDIR "/app/src"

# Command to run on start
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]