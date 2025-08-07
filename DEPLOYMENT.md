# QEM-Bench Deployment Guide

## Overview

This guide covers deployment of QEM-Bench with the new Quantum-Inspired Task Planning module across various environments, from local development to enterprise-scale production deployments.

## Quick Deployment

### Local Development

```bash
# Clone repository
git clone https://github.com/danieleschmidt/quantum-inspired-task-planner.git
cd quantum-inspired-task-planner

# Install dependencies
pip install -e ".[dev]"

# Run basic validation
python -c "from qem_bench.planning import QuantumInspiredPlanner; print('✅ Installation successful')"

# Run example
python examples/planning_example.py
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install
COPY . /app
WORKDIR /app

RUN pip install -e ".[full]"

# Expose ports
EXPOSE 8000

# Run application
CMD ["python", "-m", "qem_bench.planning.server"]
```

Build and run:
```bash
docker build -t qem-bench:latest .
docker run -p 8000:8000 qem-bench:latest
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: qem-bench-planning
spec:
  replicas: 3
  selector:
    matchLabels:
      app: qem-bench-planning
  template:
    metadata:
      labels:
        app: qem-bench-planning
    spec:
      containers:
      - name: qem-bench
        image: qem-bench:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "8Gi"
            cpu: "4"
        env:
        - name: QEM_BACKEND
          value: "gpu"
        - name: QEM_MEMORY_LIMIT
          value: "6"
---
apiVersion: v1
kind: Service
metadata:
  name: qem-bench-planning-service
spec:
  selector:
    app: qem-bench-planning
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Production Deployment

### Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   API Gateway   │    │   Monitoring    │
│   (HAProxy/     │    │   (Kong/Istio)  │    │   (Prometheus)  │
│    Nginx)       │    │                 │    │                 │
└─────────┬───────┘    └─────────┬───────┘    └─────────────────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
    ┌────────────────────────────┴────────────────────────────┐
    │                Application Layer                        │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
    │  │ QEM Planning│  │ QEM Planning│  │ QEM Planning│     │
    │  │   Node 1    │  │   Node 2    │  │   Node 3    │     │
    │  │             │  │             │  │             │     │
    │  └─────────────┘  └─────────────┘  └─────────────┘     │
    └────────────────────────────────┬────────────────────────┘
                                     │
    ┌────────────────────────────────┴────────────────────────────┐
    │                   Data Layer                                │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
    │  │   Redis     │  │ PostgreSQL  │  │   Distributed       │  │
    │  │   Cache     │  │  Metadata   │  │   File System       │  │
    │  │             │  │             │  │   (Checkpoints)     │  │
    │  └─────────────┘  └─────────────┘  └─────────────────────┘  │
    └─────────────────────────────────────────────────────────────┘
```

### Infrastructure Requirements

#### Minimum Requirements
- **CPU**: 4 cores (8 recommended)
- **Memory**: 8GB RAM (16GB+ recommended)
- **Storage**: 50GB SSD (100GB+ recommended)
- **Network**: 1Gbps (10Gbps for distributed)

#### Recommended Production
- **CPU**: 16+ cores with AVX2 support
- **Memory**: 64GB+ RAM
- **GPU**: NVIDIA V100/A100 for acceleration
- **Storage**: NVMe SSD with 10,000+ IOPS
- **Network**: 25Gbps+ with low latency

#### Enterprise Scale
- **CPU**: 32+ cores per node
- **Memory**: 128GB+ RAM per node
- **GPU**: Multiple V100/A100 per node
- **Storage**: Distributed storage (Ceph/GlusterFS)
- **Network**: 100Gbps InfiniBand

### Configuration

#### Production Configuration File

```yaml
# config/production.yaml
planning:
  max_iterations: 2000
  convergence_threshold: 1e-6
  quantum_annealing_schedule: "exponential"
  superposition_width: 0.15
  entanglement_strength: 0.6
  interference_factor: 0.4
  use_gpu: true
  enable_monitoring: true

performance:
  backend: "gpu"
  max_workers: 16
  memory_limit_gb: 48
  gpu_memory_fraction: 0.8
  enable_jit: true
  enable_vectorization: true
  enable_parallelization: true
  cache_size_mb: 4096
  batch_size: 64
  optimization_level: 2

monitoring:
  enable_metrics: true
  metrics_interval: 30
  enable_profiling: true
  log_level: "INFO"
  audit_logging: true

security:
  enable_input_sanitization: true
  enable_access_control: true
  enable_audit_logging: true
  rate_limiting: true
  max_requests_per_minute: 1000

globalization:
  default_locale: "en-US"
  supported_locales: ["en-US", "en-GB", "es-ES", "fr-FR", "de-DE", "ja-JP", "zh-CN"]
  compliance_frameworks: ["GDPR", "CCPA"]
  data_retention_days: 730
  enable_encryption: true
  enable_anonymization: true

multiregion:
  primary_region: "us-east-1"
  secondary_regions: ["us-west-2", "eu-west-1", "ap-southeast-1"]
  enable_failover: true
  enable_load_balancing: true
  enable_data_replication: true
  health_check_interval: 30
  failover_timeout: 60
```

#### Environment Variables

```bash
# Core Configuration
export QEM_CONFIG_FILE="/etc/qem-bench/production.yaml"
export QEM_LOG_LEVEL="INFO"
export QEM_ENABLE_GPU="true"

# Performance Tuning
export QEM_MEMORY_LIMIT_GB="48"
export QEM_MAX_WORKERS="16"
export QEM_BATCH_SIZE="64"
export QEM_CACHE_SIZE_MB="4096"

# Security
export QEM_ENABLE_AUTH="true"
export QEM_JWT_SECRET="your-secure-jwt-secret"
export QEM_RATE_LIMIT_RPM="1000"

# Database
export QEM_DB_HOST="postgres-cluster.internal"
export QEM_DB_PORT="5432"
export QEM_DB_NAME="qem_bench"
export QEM_DB_USER="qem_user"
export QEM_DB_PASSWORD="secure-password"

# Redis Cache
export QEM_REDIS_HOST="redis-cluster.internal"
export QEM_REDIS_PORT="6379"
export QEM_REDIS_PASSWORD="redis-password"

# Monitoring
export QEM_PROMETHEUS_GATEWAY="prometheus-gateway:9091"
export QEM_GRAFANA_URL="https://grafana.company.com"
export QEM_ALERT_WEBHOOK="https://alerts.company.com/webhook"

# Globalization
export QEM_DEFAULT_LOCALE="en-US"
export QEM_COMPLIANCE_MODE="GDPR,CCPA"
export QEM_DATA_REGION="EU"
```

## Multi-Region Deployment

### AWS Deployment

```yaml
# terraform/aws/main.tf
provider "aws" {
  region = var.primary_region
}

# EKS Cluster
resource "aws_eks_cluster" "qem_bench" {
  name     = "qem-bench-${var.environment}"
  role_arn = aws_iam_role.cluster.arn
  version  = "1.27"

  vpc_config {
    subnet_ids = aws_subnet.private[*].id
  }

  depends_on = [
    aws_iam_role_policy_attachment.cluster_AmazonEKSClusterPolicy,
  ]
}

# GPU Node Group
resource "aws_eks_node_group" "gpu_nodes" {
  cluster_name    = aws_eks_cluster.qem_bench.name
  node_group_name = "gpu-nodes"
  node_role_arn   = aws_iam_role.node.arn
  subnet_ids      = aws_subnet.private[*].id

  instance_types = ["p3.2xlarge", "p3.8xlarge"]
  
  scaling_config {
    desired_size = 2
    max_size     = 10
    min_size     = 1
  }

  remote_access {
    ec2_ssh_key = var.key_name
  }
}

# RDS for Metadata
resource "aws_rds_cluster" "qem_metadata" {
  cluster_identifier      = "qem-metadata-${var.environment}"
  engine                  = "aurora-postgresql"
  engine_version          = "13.7"
  database_name           = "qem_bench"
  master_username         = "qem_user"
  master_password         = var.db_password
  backup_retention_period = 7
  preferred_backup_window = "07:00-09:00"
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.default.name
}

# ElastiCache for Redis
resource "aws_elasticache_replication_group" "qem_cache" {
  description          = "QEM-Bench Cache"
  replication_group_id = "qem-cache-${var.environment}"
  port                 = 6379
  parameter_group_name = "default.redis7"
  node_type           = "cache.r6g.xlarge"
  num_cache_clusters  = 3
  
  subnet_group_name = aws_elasticache_subnet_group.default.name
  security_group_ids = [aws_security_group.redis.id]
}
```

### Google Cloud Deployment

```yaml
# terraform/gcp/main.tf
provider "google" {
  project = var.project_id
  region  = var.region
}

# GKE Cluster with GPU support
resource "google_container_cluster" "qem_bench" {
  name     = "qem-bench-${var.environment}"
  location = var.region

  initial_node_count = 1
  
  node_config {
    machine_type = "e2-standard-4"
    guest_accelerator {
      type  = "nvidia-tesla-t4"
      count = 1
    }
  }

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }
}

# Cloud SQL for PostgreSQL
resource "google_sql_database_instance" "qem_metadata" {
  name             = "qem-metadata-${var.environment}"
  database_version = "POSTGRES_13"
  region           = var.region

  settings {
    tier = "db-standard-2"
    
    backup_configuration {
      enabled    = true
      start_time = "07:00"
    }
    
    ip_configuration {
      ipv4_enabled = true
      authorized_networks {
        value = "0.0.0.0/0"
        name  = "all"
      }
    }
  }
}

# Memorystore for Redis
resource "google_redis_instance" "qem_cache" {
  name           = "qem-cache-${var.environment}"
  tier           = "STANDARD_HA"
  memory_size_gb = 16
  region         = var.region
  
  redis_version = "REDIS_6_X"
}
```

### Azure Deployment

```yaml
# terraform/azure/main.tf
provider "azurerm" {
  features {}
}

# AKS Cluster
resource "azurerm_kubernetes_cluster" "qem_bench" {
  name                = "qem-bench-${var.environment}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  dns_prefix          = "qem-bench-${var.environment}"

  default_node_pool {
    name           = "system"
    node_count     = 3
    vm_size        = "Standard_D4s_v3"
    vnet_subnet_id = azurerm_subnet.internal.id
  }

  identity {
    type = "SystemAssigned"
  }
}

# GPU Node Pool
resource "azurerm_kubernetes_cluster_node_pool" "gpu" {
  name                  = "gpu"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.qem_bench.id
  vm_size              = "Standard_NC6s_v3"
  node_count           = 2
  vnet_subnet_id       = azurerm_subnet.internal.id
}

# PostgreSQL Flexible Server
resource "azurerm_postgresql_flexible_server" "qem_metadata" {
  name                   = "qem-metadata-${var.environment}"
  resource_group_name    = azurerm_resource_group.main.name
  location               = azurerm_resource_group.main.location
  version                = "13"
  administrator_login    = "qem_admin"
  administrator_password = var.db_password
  
  storage_mb = 32768
  sku_name   = "GP_Standard_D2s_v3"
}

# Redis Cache
resource "azurerm_redis_cache" "qem_cache" {
  name                = "qem-cache-${var.environment}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  capacity            = 2
  family              = "C"
  sku_name            = "Standard"
  
  redis_configuration {
    maxmemory_reserved = 10
    maxmemory_delta    = 2
    maxmemory_policy   = "allkeys-lru"
  }
}
```

## Monitoring and Observability

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "qem_bench_rules.yml"

scrape_configs:
  - job_name: 'qem-bench-planning'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        action: keep
        regex: qem-bench-planning
      - source_labels: [__address__]
        action: replace
        target_label: __address__
        regex: '(.+):(?:\d+)'
        replacement: '${1}:9090'

  - job_name: 'qem-bench-scheduler'
    static_configs:
      - targets: ['scheduler-service:9091']

alerting:
  alertmanagers:
    - kubernetes_sd_configs:
        - role: pod
      relabel_configs:
        - source_labels: [__meta_kubernetes_pod_label_app]
          action: keep
          regex: alertmanager
```

### Grafana Dashboards

```json
{
  "dashboard": {
    "title": "QEM-Bench Planning Performance",
    "panels": [
      {
        "title": "Planning Requests/sec",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(qem_planning_requests_total[5m])",
            "legendFormat": "{{method}}"
          }
        ]
      },
      {
        "title": "Optimization Time",
        "type": "graph", 
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(qem_optimization_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "qem_memory_usage_bytes / 1024 / 1024 / 1024",
            "legendFormat": "Memory GB"
          }
        ]
      },
      {
        "title": "GPU Utilization",
        "type": "graph",
        "targets": [
          {
            "expr": "qem_gpu_utilization_percent",
            "legendFormat": "GPU {{device}}"
          }
        ]
      }
    ]
  }
}
```

### Alert Rules

```yaml
# monitoring/qem_bench_rules.yml
groups:
  - name: qem_bench_alerts
    rules:
      - alert: HighPlanningLatency
        expr: histogram_quantile(0.95, rate(qem_optimization_duration_seconds_bucket[5m])) > 30
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High planning latency detected"
          description: "95th percentile planning latency is {{ $value }}s"

      - alert: MemoryUsageHigh
        expr: qem_memory_usage_bytes / qem_memory_limit_bytes > 0.9
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Memory usage very high"
          description: "Memory usage is {{ $value | humanizePercentage }}"

      - alert: GPUUtilizationLow
        expr: avg(qem_gpu_utilization_percent) < 30
        for: 10m
        labels:
          severity: info
        annotations:
          summary: "GPU utilization low"
          description: "Average GPU utilization is {{ $value }}%"

      - alert: PlanningFailureRate
        expr: rate(qem_planning_failures_total[5m]) / rate(qem_planning_requests_total[5m]) > 0.1
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High planning failure rate"
          description: "Planning failure rate is {{ $value | humanizePercentage }}"
```

## Security Hardening

### Network Security

```yaml
# security/network-policies.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: qem-bench-network-policy
spec:
  podSelector:
    matchLabels:
      app: qem-bench-planning
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: qem-bench
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system
    ports:
    - protocol: TCP
      port: 53
    - protocol: UDP
      port: 53
```

### Pod Security

```yaml
# security/pod-security.yaml
apiVersion: v1
kind: Pod
metadata:
  name: qem-bench-planning
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    fsGroup: 2000
  containers:
  - name: qem-bench
    image: qem-bench:latest
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop:
        - ALL
    resources:
      requests:
        memory: "2Gi"
        cpu: "1"
      limits:
        memory: "8Gi"
        cpu: "4"
    livenessProbe:
      httpGet:
        path: /health
        port: 8000
      initialDelaySeconds: 30
      periodSeconds: 10
    readinessProbe:
      httpGet:
        path: /ready
        port: 8000
      initialDelaySeconds: 5
      periodSeconds: 5
```

### RBAC Configuration

```yaml
# security/rbac.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: qem-bench-role
rules:
- apiGroups: [""]
  resources: ["pods", "configmaps", "secrets"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments"]
  verbs: ["get", "list", "watch"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: qem-bench-binding
subjects:
- kind: ServiceAccount
  name: qem-bench-service-account
roleRef:
  kind: Role
  name: qem-bench-role
  apiGroup: rbac.authorization.k8s.io

---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: qem-bench-service-account
```

## Performance Optimization

### CPU Optimization

```bash
# Set CPU governor for performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Disable CPU idle states for low latency
sudo cpupower idle-set -D 0

# Set CPU affinity for planning processes
taskset -c 0-15 python -m qem_bench.planning.server
```

### Memory Optimization

```bash
# Configure huge pages
echo 1024 | sudo tee /proc/sys/vm/nr_hugepages

# Optimize memory allocation
export MALLOC_ARENA_MAX=2
export MALLOC_MMAP_THRESHOLD_=131072
export MALLOC_TOP_PAD_=131072

# Set memory limits
ulimit -m 67108864  # 64GB limit
```

### GPU Optimization

```bash
# Set GPU persistence mode
sudo nvidia-smi -pm 1

# Set maximum performance mode
sudo nvidia-smi -ac 877,1380

# Enable MIG (Multi-Instance GPU) if needed
sudo nvidia-smi -mig 1
```

### Storage Optimization

```bash
# Mount with appropriate options
mount -t ext4 -o noatime,data=writeback /dev/sdb1 /data/qem-bench

# Configure filesystem
echo 'vm.dirty_ratio=5' | sudo tee -a /etc/sysctl.conf
echo 'vm.dirty_background_ratio=2' | sudo tee -a /etc/sysctl.conf
```

## Backup and Disaster Recovery

### Backup Strategy

```bash
#!/bin/bash
# backup/backup-qem-bench.sh

# Database backup
pg_dump -h postgres-cluster.internal -U qem_user qem_bench | \
  gzip > /backups/qem-bench-$(date +%Y%m%d-%H%M%S).sql.gz

# Redis backup
redis-cli --rdb /backups/redis-$(date +%Y%m%d-%H%M%S).rdb

# Checkpoint backup
tar czf /backups/checkpoints-$(date +%Y%m%d-%H%M%S).tar.gz /data/qem-bench/checkpoints/

# Configuration backup
tar czf /backups/config-$(date +%Y%m%d-%H%M%S).tar.gz /etc/qem-bench/

# Upload to cloud storage
aws s3 sync /backups/ s3://qem-bench-backups/$(date +%Y/%m/%d)/
```

### Disaster Recovery

```bash
#!/bin/bash
# disaster-recovery/restore-qem-bench.sh

# Restore database
gunzip -c qem-bench-backup.sql.gz | \
  psql -h postgres-cluster.internal -U qem_user qem_bench

# Restore Redis
redis-cli --rdb redis-backup.rdb
redis-cli flushall
redis-cli --rdb redis-backup.rdb

# Restore checkpoints
tar xzf checkpoints-backup.tar.gz -C /data/qem-bench/

# Restart services
kubectl rollout restart deployment/qem-bench-planning
```

## Troubleshooting

### Common Issues

#### High Memory Usage
```bash
# Check memory usage
kubectl top pods -l app=qem-bench-planning

# Adjust memory limits
kubectl patch deployment qem-bench-planning -p='{"spec":{"template":{"spec":{"containers":[{"name":"qem-bench","resources":{"limits":{"memory":"16Gi"}}}]}}}}'
```

#### GPU Not Detected
```bash
# Verify GPU availability
nvidia-smi

# Check CUDA installation
python -c "import jax; print(jax.devices())"

# Verify Kubernetes GPU support
kubectl describe node | grep nvidia.com/gpu
```

#### Planning Timeout
```bash
# Increase timeout
export QEM_PLANNING_TIMEOUT=600

# Check system resources
top -p $(pgrep -f qem_bench)

# Review optimization parameters
python -c "
from qem_bench.planning import PlanningConfig
config = PlanningConfig(max_iterations=5000, convergence_threshold=1e-4)
print(f'Config: {config}')
"
```

### Log Analysis

```bash
# View planning logs
kubectl logs -l app=qem-bench-planning --tail=100

# Search for errors
kubectl logs -l app=qem-bench-planning | grep -i error

# Monitor planning performance
kubectl logs -l app=qem-bench-planning | grep "optimization_time"
```

### Performance Debugging

```bash
# Profile planning performance
python -m cProfile -o planning.prof examples/planning_example.py

# Analyze memory usage
python -m memory_profiler examples/planning_example.py

# Monitor system resources
watch -n 1 'kubectl top pods -l app=qem-bench-planning'
```

## Scaling Guidelines

### Horizontal Scaling

```yaml
# autoscaling/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: qem-bench-planning-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: qem-bench-planning
  minReplicas: 3
  maxReplicas: 50
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Vertical Scaling

```yaml
# autoscaling/vpa.yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: qem-bench-planning-vpa
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: qem-bench-planning
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: qem-bench
      minAllowed:
        cpu: "1"
        memory: "2Gi"
      maxAllowed:
        cpu: "8"
        memory: "32Gi"
```

### Multi-Cluster Scaling

```yaml
# federation/multi-cluster.yaml
apiVersion: types.kubefed.io/v1beta1
kind: FederatedDeployment
metadata:
  name: qem-bench-planning
spec:
  template:
    metadata:
      labels:
        app: qem-bench-planning
    spec:
      replicas: 3
      selector:
        matchLabels:
          app: qem-bench-planning
      template:
        metadata:
          labels:
            app: qem-bench-planning
        spec:
          containers:
          - name: qem-bench
            image: qem-bench:latest
  placement:
    clusters:
    - name: us-east-1
    - name: us-west-2
    - name: eu-west-1
  overrides:
  - clusterName: us-east-1
    clusterOverrides:
    - path: "/spec/replicas"
      value: 5
  - clusterName: eu-west-1
    clusterOverrides:
    - path: "/spec/replicas"
      value: 3
```

This comprehensive deployment guide covers all aspects of deploying QEM-Bench with quantum planning capabilities from development to enterprise scale. Follow the sections relevant to your deployment scenario and adjust configurations based on your specific requirements.