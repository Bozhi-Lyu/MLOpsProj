# Base image
FROM python:3.10-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/* 

# Fairly certain that this requires that you log into
# Weights and Biases locally first


COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml

# Copy your Git repository into the Docker image
COPY . /app
COPY models/ /app/models
# Change the working directory to the directory that contains your Git repository
WORKDIR /app

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
