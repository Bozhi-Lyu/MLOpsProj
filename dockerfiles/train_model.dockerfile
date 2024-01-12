# Base image
FROM python:3.10-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Fairly certain that this requires that you log into
# Weights and Biases locally first
ARG WANDB_API_KEY
ENV WANDB_API_KEY=$WANDB_API_KEY

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY data/ data/
COPY thebestofthebest-411009-4622a4d8f58f.json thebestofthebest-411009-4622a4d8f58f.json

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir
RUN dvc remote modify --local thebestofthebest-411009-4622a4d8f58f.json

#ENTRYPOINT ["python", "-u", "src/train_model.py"]
ENTRYPOINT ["make", "train"]