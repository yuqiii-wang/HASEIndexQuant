poetry install

poetry run python -m grpc_tools.protoc \
    -I./protos \
    --python_out=./protos \
    --pyi_out=./protos \ # Optional: for type stubs
    ./protos/market_data.proto