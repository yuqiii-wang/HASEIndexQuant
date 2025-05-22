# ---- Builder Stage ----
FROM python:3.12-alpine as builder

WORKDIR /app

# Install poetry
RUN pip install poetry

# Copy only files necessary for dependency installation and proto generation
COPY ./src/pyproject.toml poetry.lock ./
COPY ./src/protos/market_data.proto ./src/protos/market_data.proto
# Create empty __init__.py if not copied, so protoc output dir is a package
RUN mkdir -p ./src/protos && touch ./src/protos/__init__.py

# Install dependencies (including dev for grpcio-tools)
# --no-root: don't install the project itself yet
RUN poetry install --no-root --with dev

# Generate protobuf files
# Ensure the output directory exists and is a package
RUN poetry run python -m grpc_tools.protoc \
    -I./src/protos \
    --python_out=./src/protos \
    ./src/protos/market_data.proto
    # Note: outputting to ./market_data_consumer makes the import
    # from market_data_consumer.protos import market_data_pb2 work

# Now copy the rest of the application code
COPY market_data_consumer ./market_data_consumer

FROM python:3.12-alpine

WORKDIR /app

RUN pip install poetry

COPY ./src/pyproject.toml ./src/poetry.lock ./
# Install project and deps to generate requirements.txt

COPY ./src/dual_udp_consumer.py .
# No external libraries needed for this basic client

ENV PYTHONUNBUFFERED=1

# CLIENT_LISTEN_PORT will be set by K8s (containerPort)
# ENV CLIENT_LISTEN_PORT="9999"

CMD ["python", "dual_udp_consumer.py"]