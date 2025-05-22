FROM python:3.12-alpine

WORKDIR /app

COPY ./src/dual_udp_server.py .
# No external libraries needed for this basic client

ENV PYTHONUNBUFFERED=1

# CLIENT_LISTEN_PORT will be set by K8s (containerPort)
# ENV CLIENT_LISTEN_PORT="9999"

CMD ["python", "dual_udp_server.py"]