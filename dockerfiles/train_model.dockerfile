# Base image
FROM python:3.10-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Fairly certain that this requires that you log into
# Weights and Biases locally first
ARG WANDB_API_KEY
ENV WANDB_API_KEY=$WANDB_API_KEY

COPY ../requirements.txt requirements.txt
COPY ../pyproject.toml pyproject.toml
COPY ../src/ src/


WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir
ENV GOOGLE_APPLICATION_CREDENTIALS='gcpkey.json'

#ENTRYPOINT ["python", "-u", "src/train_model.py"]
ENTRYPOINT ["make", "train"]