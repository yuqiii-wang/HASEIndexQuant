export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7890


minikube start
minikube addons enable metrics-server

minikube dashboard

helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update
helm install postgresqldb bitnami/postgresql \
  -f k8s/postgresqldb-values.yaml \
  -n postgresqldb \
  --create-namespace

helm repo add nats https://nats-io.github.io/k8s/helm/charts/
helm repo update
helm install nats nats/nats \
  -f k8s/nats-values.yaml \
  -n nats \
  --create-namespace

helm repo add timescaledb https://charts.timescale.com/
helm repo update
kubectl create secret generic timescaledb-superuser-secret \
  --from-literal=password='YOUR_STRONG_SUPERUSER_PASSWORD' \
  -n timescaledb
helm install tsdb-cluster timescaledb/timescaledb-multinode \
  -f ./k8s/timescaledb-values.yaml \
  -n timescaledb \
  --create-namespace

helm install promscale timescaledb/promscale \
  --namespace monitoring \
  -f ./k8s/promscale-values.yaml

helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm install kube-prom-stack prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  -f ./k8s/kube-prometheus-grafana-values.yaml \
  --create-namespace

helm repo add grafana https://grafana.github.io/helm-charts
helm repo update
helm install loki-stack grafana/loki-stack \
  --namespace monitoring \
  -f ./k8s/loki-stack-values.yaml \
  --create-namespace

###

# Get Grafana 'admin' user password by running:
kubectl --namespace monitoring get secrets kube-prom-stack-grafana -o jsonpath="{.data.admin-password}" | base64 -d ; echo

# Access Grafana local instance:
export POD_NAME=$(kubectl --namespace monitoring get pod -l "app.kubernetes.io/name=grafana,app.kubernetes.io/instance=kube-prom-stack" -oname)
kubectl --namespace monitoring port-forward $POD_NAME 3000

### Local code and docker

docker build -f k8s/containers/dual_udp_consumer.dockerfile -t dual_udp_consumer .
docker build -f k8s/containers/dual_udp_server.dockerfile -t dual_udp_server .

kubectl apply -f k8s/udp-consumer-service.yaml
kubectl apply -f k8s/udp-consumer-deployment.yaml
kubectl apply -f k8s/udp-server-deployment.yaml

poetry run python -m grpc_tools.protoc \
    -I./protos \
    --python_out=./protos \
    --pyi_out=./protos \ # Optional: for type stubs
    ./protos/market_data.proto