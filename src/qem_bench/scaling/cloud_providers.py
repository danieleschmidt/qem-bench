"""
Cloud provider integrations for auto-scaling and resource management.

This module provides abstractions for major cloud providers (AWS, Google Cloud, Azure)
with support for auto-scaling groups, spot instance management, serverless functions,
and cost optimization strategies.
"""

import time
import logging
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
import numpy as np
from datetime import datetime, timedelta

from ..security import SecureConfig, CredentialManager


logger = logging.getLogger(__name__)


class CloudProvider(Enum):
    """Supported cloud providers."""
    AWS = "aws"
    GOOGLE_CLOUD = "google_cloud"
    AZURE = "azure"
    IBM_CLOUD = "ibm_cloud"


class InstanceType(Enum):
    """Cloud instance types."""
    MICRO = "micro"        # t3.micro, f1-micro, B1s
    SMALL = "small"        # t3.small, n1-standard-1, B1ms
    MEDIUM = "medium"      # t3.medium, n1-standard-2, B2s
    LARGE = "large"        # t3.large, n1-standard-4, B2ms
    XLARGE = "xlarge"      # t3.xlarge, n1-standard-8, B4ms
    COMPUTE_OPTIMIZED = "compute_optimized"  # c5.large, c2-standard-4, F2s
    MEMORY_OPTIMIZED = "memory_optimized"    # r5.large, n1-highmem-2, E2s
    GPU_ENABLED = "gpu_enabled"              # p3.2xlarge, n1-standard-4-gpu, NC6


class InstanceState(Enum):
    """Instance states."""
    PENDING = "pending"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    TERMINATING = "terminating"
    TERMINATED = "terminated"
    SPOT_INTERRUPTED = "spot_interrupted"


class ScalingAction(Enum):
    """Auto-scaling actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"


@dataclass
class CloudInstance:
    """Represents a cloud compute instance."""
    id: str
    provider: CloudProvider
    instance_type: InstanceType
    state: InstanceState
    
    # Instance specifications
    cpu_cores: int
    memory_gb: float
    gpu_count: int = 0
    storage_gb: float = 20.0
    
    # Network configuration
    public_ip: Optional[str] = None
    private_ip: Optional[str] = None
    availability_zone: str = "us-east-1a"
    
    # Pricing information
    hourly_cost: float = 0.0
    is_spot_instance: bool = False
    spot_price: Optional[float] = None
    
    # Lifecycle management
    launch_time: Optional[float] = None
    termination_time: Optional[float] = None
    uptime_hours: float = 0.0
    
    # Performance metrics
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    network_in_gb: float = 0.0
    network_out_gb: float = 0.0
    
    def is_running(self) -> bool:
        """Check if instance is running."""
        return self.state == InstanceState.RUNNING
    
    def is_spot_available(self) -> bool:
        """Check if spot instance is available (not interrupted)."""
        return self.is_spot_instance and self.state != InstanceState.SPOT_INTERRUPTED
    
    def total_cost(self) -> float:
        """Calculate total cost since launch."""
        if self.launch_time:
            runtime_hours = (time.time() - self.launch_time) / 3600
            return runtime_hours * self.hourly_cost
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert instance to dictionary."""
        return {
            "id": self.id,
            "provider": self.provider.value,
            "instance_type": self.instance_type.value,
            "state": self.state.value,
            "cpu_cores": self.cpu_cores,
            "memory_gb": self.memory_gb,
            "gpu_count": self.gpu_count,
            "public_ip": self.public_ip,
            "private_ip": self.private_ip,
            "availability_zone": self.availability_zone,
            "hourly_cost": self.hourly_cost,
            "is_spot_instance": self.is_spot_instance,
            "spot_price": self.spot_price,
            "uptime_hours": self.uptime_hours,
            "cpu_utilization": self.cpu_utilization,
            "memory_utilization": self.memory_utilization,
            "total_cost": self.total_cost(),
            "is_running": self.is_running()
        }


@dataclass
class AutoScalingGroup:
    """Auto-scaling group configuration."""
    name: str
    provider: CloudProvider
    
    # Scaling configuration
    min_instances: int = 1
    max_instances: int = 10
    desired_instances: int = 2
    instance_type: InstanceType = InstanceType.MEDIUM
    
    # Scaling policies
    scale_up_threshold: float = 75.0    # CPU utilization %
    scale_down_threshold: float = 25.0  # CPU utilization %
    cooldown_seconds: int = 300         # 5 minutes
    
    # Health checks
    health_check_grace_period: int = 300  # 5 minutes
    health_check_type: str = "EC2"        # EC2 or ELB
    
    # Spot instance configuration
    enable_spot_instances: bool = False
    spot_allocation_strategy: str = "diversified"  # diversified, lowest-price
    on_demand_percentage: int = 20  # % of on-demand vs spot
    
    # Current state
    current_instances: List[str] = field(default_factory=list)
    last_scaling_action: Optional[float] = None
    
    def can_scale_up(self) -> bool:
        """Check if group can scale up."""
        return len(self.current_instances) < self.max_instances
    
    def can_scale_down(self) -> bool:
        """Check if group can scale down."""
        return len(self.current_instances) > self.min_instances
    
    def is_in_cooldown(self) -> bool:
        """Check if scaling is in cooldown period."""
        if self.last_scaling_action is None:
            return False
        return time.time() - self.last_scaling_action < self.cooldown_seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "provider": self.provider.value,
            "min_instances": self.min_instances,
            "max_instances": self.max_instances,
            "desired_instances": self.desired_instances,
            "instance_type": self.instance_type.value,
            "scale_up_threshold": self.scale_up_threshold,
            "scale_down_threshold": self.scale_down_threshold,
            "enable_spot_instances": self.enable_spot_instances,
            "current_instances": len(self.current_instances),
            "can_scale_up": self.can_scale_up(),
            "can_scale_down": self.can_scale_down(),
            "is_in_cooldown": self.is_in_cooldown()
        }


class CloudProviderAPI(ABC):
    """Abstract base class for cloud provider APIs."""
    
    @abstractmethod
    async def launch_instance(
        self,
        instance_type: InstanceType,
        availability_zone: str = "us-east-1a",
        spot_instance: bool = False
    ) -> CloudInstance:
        """Launch a new instance."""
        pass
    
    @abstractmethod
    async def terminate_instance(self, instance_id: str) -> bool:
        """Terminate an instance."""
        pass
    
    @abstractmethod
    async def list_instances(self) -> List[CloudInstance]:
        """List all instances."""
        pass
    
    @abstractmethod
    async def get_instance_metrics(self, instance_id: str) -> Dict[str, float]:
        """Get instance performance metrics."""
        pass
    
    @abstractmethod
    async def get_spot_price_history(
        self,
        instance_type: InstanceType,
        availability_zone: str,
        hours: int = 24
    ) -> List[Tuple[float, float]]:
        """Get spot price history."""
        pass
    
    @abstractmethod
    async def create_auto_scaling_group(self, config: AutoScalingGroup) -> bool:
        """Create auto-scaling group."""
        pass
    
    @abstractmethod
    async def update_auto_scaling_group(
        self,
        group_name: str,
        desired_capacity: int
    ) -> bool:
        """Update auto-scaling group desired capacity."""
        pass


class AWSProvider(CloudProviderAPI):
    """AWS cloud provider implementation."""
    
    def __init__(self, credentials: Optional[Dict[str, str]] = None):
        self.credentials = credentials or {}
        self.instances: Dict[str, CloudInstance] = {}
        self.auto_scaling_groups: Dict[str, AutoScalingGroup] = {}
        
        # AWS-specific configuration
        self.region = self.credentials.get("region", "us-east-1")
        self.instance_type_mapping = {
            InstanceType.MICRO: {"type": "t3.micro", "cpu": 2, "memory": 1.0, "cost": 0.0104},
            InstanceType.SMALL: {"type": "t3.small", "cpu": 2, "memory": 2.0, "cost": 0.0208},
            InstanceType.MEDIUM: {"type": "t3.medium", "cpu": 2, "memory": 4.0, "cost": 0.0416},
            InstanceType.LARGE: {"type": "t3.large", "cpu": 2, "memory": 8.0, "cost": 0.0832},
            InstanceType.XLARGE: {"type": "t3.xlarge", "cpu": 4, "memory": 16.0, "cost": 0.1664},
            InstanceType.COMPUTE_OPTIMIZED: {"type": "c5.large", "cpu": 2, "memory": 4.0, "cost": 0.085},
            InstanceType.MEMORY_OPTIMIZED: {"type": "r5.large", "cpu": 2, "memory": 16.0, "cost": 0.126},
            InstanceType.GPU_ENABLED: {"type": "p3.2xlarge", "cpu": 8, "memory": 61.0, "cost": 3.06, "gpu": 1}
        }
        
        logger.info("AWS provider initialized")
    
    async def launch_instance(
        self,
        instance_type: InstanceType,
        availability_zone: str = "us-east-1a",
        spot_instance: bool = False
    ) -> CloudInstance:
        """Launch AWS EC2 instance."""
        instance_id = f"i-{int(time.time() * 1000) % 1000000:06x}"
        
        # Get instance specifications
        spec = self.instance_type_mapping[instance_type]
        
        # Create instance object
        instance = CloudInstance(
            id=instance_id,
            provider=CloudProvider.AWS,
            instance_type=instance_type,
            state=InstanceState.PENDING,
            cpu_cores=spec["cpu"],
            memory_gb=spec["memory"],
            gpu_count=spec.get("gpu", 0),
            availability_zone=availability_zone,
            hourly_cost=spec["cost"],
            is_spot_instance=spot_instance,
            launch_time=time.time()
        )
        
        # Simulate spot pricing
        if spot_instance:
            base_cost = spec["cost"]
            instance.spot_price = base_cost * np.random.uniform(0.3, 0.8)  # 30-80% of on-demand
            instance.hourly_cost = instance.spot_price
        
        # Store instance
        self.instances[instance_id] = instance
        
        # Simulate launch process
        await asyncio.sleep(0.1)  # Simulate API call delay
        instance.state = InstanceState.RUNNING
        instance.private_ip = f"10.0.1.{len(self.instances)}"
        instance.public_ip = f"54.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}"
        
        logger.info(f"Launched AWS instance {instance_id} ({instance_type.value})")
        return instance
    
    async def terminate_instance(self, instance_id: str) -> bool:
        """Terminate AWS EC2 instance."""
        if instance_id not in self.instances:
            return False
        
        instance = self.instances[instance_id]
        instance.state = InstanceState.TERMINATING
        
        # Simulate termination process
        await asyncio.sleep(0.1)
        instance.state = InstanceState.TERMINATED
        instance.termination_time = time.time()
        
        logger.info(f"Terminated AWS instance {instance_id}")
        return True
    
    async def list_instances(self) -> List[CloudInstance]:
        """List all AWS instances."""
        return list(self.instances.values())
    
    async def get_instance_metrics(self, instance_id: str) -> Dict[str, float]:
        """Get AWS CloudWatch metrics."""
        if instance_id not in self.instances:
            return {}
        
        instance = self.instances[instance_id]
        
        # Simulate metrics collection
        return {
            "cpu_utilization": np.random.uniform(10, 90),
            "memory_utilization": np.random.uniform(20, 80),
            "network_in": np.random.uniform(0, 1000),  # MB
            "network_out": np.random.uniform(0, 1000)  # MB
        }
    
    async def get_spot_price_history(
        self,
        instance_type: InstanceType,
        availability_zone: str,
        hours: int = 24
    ) -> List[Tuple[float, float]]:
        """Get AWS spot price history."""
        base_price = self.instance_type_mapping[instance_type]["cost"]
        
        # Generate simulated price history
        history = []
        current_time = time.time()
        
        for i in range(hours):
            timestamp = current_time - (hours - i) * 3600
            # Simulate price fluctuation
            price_factor = np.random.uniform(0.3, 0.9)
            spot_price = base_price * price_factor
            history.append((timestamp, spot_price))
        
        return history
    
    async def create_auto_scaling_group(self, config: AutoScalingGroup) -> bool:
        """Create AWS Auto Scaling Group."""
        self.auto_scaling_groups[config.name] = config
        
        logger.info(f"Created AWS Auto Scaling Group: {config.name}")
        return True
    
    async def update_auto_scaling_group(
        self,
        group_name: str,
        desired_capacity: int
    ) -> bool:
        """Update AWS Auto Scaling Group capacity."""
        if group_name not in self.auto_scaling_groups:
            return False
        
        group = self.auto_scaling_groups[group_name]
        current_instances = len(group.current_instances)
        
        if desired_capacity > current_instances:
            # Scale up
            for _ in range(desired_capacity - current_instances):
                instance = await self.launch_instance(
                    group.instance_type,
                    spot_instance=group.enable_spot_instances
                )
                group.current_instances.append(instance.id)
        
        elif desired_capacity < current_instances:
            # Scale down
            instances_to_terminate = current_instances - desired_capacity
            for _ in range(instances_to_terminate):
                if group.current_instances:
                    instance_id = group.current_instances.pop()
                    await self.terminate_instance(instance_id)
        
        group.last_scaling_action = time.time()
        
        logger.info(f"Updated AWS ASG {group_name} to {desired_capacity} instances")
        return True


class GoogleCloudProvider(CloudProviderAPI):
    """Google Cloud Platform provider implementation."""
    
    def __init__(self, credentials: Optional[Dict[str, str]] = None):
        self.credentials = credentials or {}
        self.instances: Dict[str, CloudInstance] = {}
        self.auto_scaling_groups: Dict[str, AutoScalingGroup] = {}
        
        # GCP-specific configuration
        self.project_id = self.credentials.get("project_id", "quantum-project")
        self.zone = self.credentials.get("zone", "us-central1-a")
        
        self.instance_type_mapping = {
            InstanceType.MICRO: {"type": "f1-micro", "cpu": 1, "memory": 0.6, "cost": 0.0076},
            InstanceType.SMALL: {"type": "n1-standard-1", "cpu": 1, "memory": 3.75, "cost": 0.0475},
            InstanceType.MEDIUM: {"type": "n1-standard-2", "cpu": 2, "memory": 7.5, "cost": 0.095},
            InstanceType.LARGE: {"type": "n1-standard-4", "cpu": 4, "memory": 15.0, "cost": 0.19},
            InstanceType.XLARGE: {"type": "n1-standard-8", "cpu": 8, "memory": 30.0, "cost": 0.38},
            InstanceType.COMPUTE_OPTIMIZED: {"type": "c2-standard-4", "cpu": 4, "memory": 16.0, "cost": 0.1992},
            InstanceType.MEMORY_OPTIMIZED: {"type": "n1-highmem-2", "cpu": 2, "memory": 13.0, "cost": 0.1184},
            InstanceType.GPU_ENABLED: {"type": "n1-standard-4-k80", "cpu": 4, "memory": 15.0, "cost": 0.64, "gpu": 1}
        }
        
        logger.info("Google Cloud provider initialized")
    
    async def launch_instance(
        self,
        instance_type: InstanceType,
        availability_zone: str = "us-central1-a",
        spot_instance: bool = False
    ) -> CloudInstance:
        """Launch GCP Compute Engine instance."""
        instance_id = f"quantum-instance-{int(time.time() * 1000) % 1000000}"
        
        spec = self.instance_type_mapping[instance_type]
        
        instance = CloudInstance(
            id=instance_id,
            provider=CloudProvider.GOOGLE_CLOUD,
            instance_type=instance_type,
            state=InstanceState.PENDING,
            cpu_cores=spec["cpu"],
            memory_gb=spec["memory"],
            gpu_count=spec.get("gpu", 0),
            availability_zone=availability_zone,
            hourly_cost=spec["cost"],
            is_spot_instance=spot_instance,  # GCP calls these "preemptible"
            launch_time=time.time()
        )
        
        if spot_instance:
            # Preemptible instances are ~80% cheaper
            instance.spot_price = spec["cost"] * 0.2
            instance.hourly_cost = instance.spot_price
        
        self.instances[instance_id] = instance
        
        await asyncio.sleep(0.1)
        instance.state = InstanceState.RUNNING
        instance.private_ip = f"10.128.0.{len(self.instances)}"
        instance.public_ip = f"35.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}"
        
        logger.info(f"Launched GCP instance {instance_id}")
        return instance
    
    async def terminate_instance(self, instance_id: str) -> bool:
        """Terminate GCP instance."""
        if instance_id not in self.instances:
            return False
        
        instance = self.instances[instance_id]
        instance.state = InstanceState.TERMINATING
        
        await asyncio.sleep(0.1)
        instance.state = InstanceState.TERMINATED
        instance.termination_time = time.time()
        
        logger.info(f"Terminated GCP instance {instance_id}")
        return True
    
    async def list_instances(self) -> List[CloudInstance]:
        """List GCP instances."""
        return list(self.instances.values())
    
    async def get_instance_metrics(self, instance_id: str) -> Dict[str, float]:
        """Get GCP Monitoring metrics."""
        if instance_id not in self.instances:
            return {}
        
        return {
            "cpu_utilization": np.random.uniform(10, 90),
            "memory_utilization": np.random.uniform(20, 80),
            "network_in": np.random.uniform(0, 1000),
            "network_out": np.random.uniform(0, 1000)
        }
    
    async def get_spot_price_history(
        self,
        instance_type: InstanceType,
        availability_zone: str,
        hours: int = 24
    ) -> List[Tuple[float, float]]:
        """Get GCP preemptible pricing history."""
        base_price = self.instance_type_mapping[instance_type]["cost"] * 0.2
        
        # GCP preemptible pricing is more stable
        history = []
        current_time = time.time()
        
        for i in range(hours):
            timestamp = current_time - (hours - i) * 3600
            price = base_price * np.random.uniform(0.95, 1.05)  # Small variations
            history.append((timestamp, price))
        
        return history
    
    async def create_auto_scaling_group(self, config: AutoScalingGroup) -> bool:
        """Create GCP Managed Instance Group."""
        self.auto_scaling_groups[config.name] = config
        logger.info(f"Created GCP Managed Instance Group: {config.name}")
        return True
    
    async def update_auto_scaling_group(
        self,
        group_name: str,
        desired_capacity: int
    ) -> bool:
        """Update GCP autoscaler."""
        if group_name not in self.auto_scaling_groups:
            return False
        
        group = self.auto_scaling_groups[group_name]
        current_instances = len(group.current_instances)
        
        if desired_capacity > current_instances:
            for _ in range(desired_capacity - current_instances):
                instance = await self.launch_instance(
                    group.instance_type,
                    spot_instance=group.enable_spot_instances
                )
                group.current_instances.append(instance.id)
        
        elif desired_capacity < current_instances:
            instances_to_terminate = current_instances - desired_capacity
            for _ in range(instances_to_terminate):
                if group.current_instances:
                    instance_id = group.current_instances.pop()
                    await self.terminate_instance(instance_id)
        
        group.last_scaling_action = time.time()
        
        logger.info(f"Updated GCP MIG {group_name} to {desired_capacity} instances")
        return True


class AzureProvider(CloudProviderAPI):
    """Microsoft Azure provider implementation."""
    
    def __init__(self, credentials: Optional[Dict[str, str]] = None):
        self.credentials = credentials or {}
        self.instances: Dict[str, CloudInstance] = {}
        self.auto_scaling_groups: Dict[str, AutoScalingGroup] = {}
        
        # Azure-specific configuration
        self.subscription_id = self.credentials.get("subscription_id", "sub-12345")
        self.resource_group = self.credentials.get("resource_group", "quantum-rg")
        self.location = self.credentials.get("location", "East US")
        
        self.instance_type_mapping = {
            InstanceType.MICRO: {"type": "Standard_B1s", "cpu": 1, "memory": 1.0, "cost": 0.0104},
            InstanceType.SMALL: {"type": "Standard_B1ms", "cpu": 1, "memory": 2.0, "cost": 0.0208},
            InstanceType.MEDIUM: {"type": "Standard_B2s", "cpu": 2, "memory": 4.0, "cost": 0.0416},
            InstanceType.LARGE: {"type": "Standard_B2ms", "cpu": 2, "memory": 8.0, "cost": 0.0832},
            InstanceType.XLARGE: {"type": "Standard_B4ms", "cpu": 4, "memory": 16.0, "cost": 0.1664},
            InstanceType.COMPUTE_OPTIMIZED: {"type": "Standard_F2s_v2", "cpu": 2, "memory": 4.0, "cost": 0.085},
            InstanceType.MEMORY_OPTIMIZED: {"type": "Standard_E2s_v3", "cpu": 2, "memory": 16.0, "cost": 0.126},
            InstanceType.GPU_ENABLED: {"type": "Standard_NC6", "cpu": 6, "memory": 56.0, "cost": 0.90, "gpu": 1}
        }
        
        logger.info("Azure provider initialized")
    
    async def launch_instance(
        self,
        instance_type: InstanceType,
        availability_zone: str = "eastus-1",
        spot_instance: bool = False
    ) -> CloudInstance:
        """Launch Azure VM."""
        instance_id = f"quantum-vm-{int(time.time() * 1000) % 1000000}"
        
        spec = self.instance_type_mapping[instance_type]
        
        instance = CloudInstance(
            id=instance_id,
            provider=CloudProvider.AZURE,
            instance_type=instance_type,
            state=InstanceState.PENDING,
            cpu_cores=spec["cpu"],
            memory_gb=spec["memory"],
            gpu_count=spec.get("gpu", 0),
            availability_zone=availability_zone,
            hourly_cost=spec["cost"],
            is_spot_instance=spot_instance,
            launch_time=time.time()
        )
        
        if spot_instance:
            # Azure Spot VMs
            instance.spot_price = spec["cost"] * np.random.uniform(0.1, 0.9)
            instance.hourly_cost = instance.spot_price
        
        self.instances[instance_id] = instance
        
        await asyncio.sleep(0.1)
        instance.state = InstanceState.RUNNING
        instance.private_ip = f"10.0.0.{len(self.instances)}"
        instance.public_ip = f"52.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}"
        
        logger.info(f"Launched Azure VM {instance_id}")
        return instance
    
    async def terminate_instance(self, instance_id: str) -> bool:
        """Terminate Azure VM."""
        if instance_id not in self.instances:
            return False
        
        instance = self.instances[instance_id]
        instance.state = InstanceState.TERMINATING
        
        await asyncio.sleep(0.1)
        instance.state = InstanceState.TERMINATED
        instance.termination_time = time.time()
        
        logger.info(f"Terminated Azure VM {instance_id}")
        return True
    
    async def list_instances(self) -> List[CloudInstance]:
        """List Azure VMs."""
        return list(self.instances.values())
    
    async def get_instance_metrics(self, instance_id: str) -> Dict[str, float]:
        """Get Azure Monitor metrics."""
        if instance_id not in self.instances:
            return {}
        
        return {
            "cpu_utilization": np.random.uniform(10, 90),
            "memory_utilization": np.random.uniform(20, 80),
            "network_in": np.random.uniform(0, 1000),
            "network_out": np.random.uniform(0, 1000)
        }
    
    async def get_spot_price_history(
        self,
        instance_type: InstanceType,
        availability_zone: str,
        hours: int = 24
    ) -> List[Tuple[float, float]]:
        """Get Azure Spot VM pricing history."""
        base_price = self.instance_type_mapping[instance_type]["cost"]
        
        history = []
        current_time = time.time()
        
        for i in range(hours):
            timestamp = current_time - (hours - i) * 3600
            price = base_price * np.random.uniform(0.1, 0.9)
            history.append((timestamp, price))
        
        return history
    
    async def create_auto_scaling_group(self, config: AutoScalingGroup) -> bool:
        """Create Azure Virtual Machine Scale Set."""
        self.auto_scaling_groups[config.name] = config
        logger.info(f"Created Azure VMSS: {config.name}")
        return True
    
    async def update_auto_scaling_group(
        self,
        group_name: str,
        desired_capacity: int
    ) -> bool:
        """Update Azure VMSS capacity."""
        if group_name not in self.auto_scaling_groups:
            return False
        
        group = self.auto_scaling_groups[group_name]
        current_instances = len(group.current_instances)
        
        if desired_capacity > current_instances:
            for _ in range(desired_capacity - current_instances):
                instance = await self.launch_instance(
                    group.instance_type,
                    spot_instance=group.enable_spot_instances
                )
                group.current_instances.append(instance.id)
        
        elif desired_capacity < current_instances:
            instances_to_terminate = current_instances - desired_capacity
            for _ in range(instances_to_terminate):
                if group.current_instances:
                    instance_id = group.current_instances.pop()
                    await self.terminate_instance(instance_id)
        
        group.last_scaling_action = time.time()
        
        logger.info(f"Updated Azure VMSS {group_name} to {desired_capacity} instances")
        return True


class SpotInstanceManager:
    """
    Advanced spot instance management with interruption handling.
    
    Features:
    - Multi-cloud spot instance monitoring
    - Price trend analysis and prediction
    - Interruption probability estimation
    - Automatic failover to on-demand instances
    - Cost optimization strategies
    """
    
    def __init__(self, providers: Dict[CloudProvider, CloudProviderAPI]):
        self.providers = providers
        
        # Spot instance tracking
        self.spot_instances: Dict[str, CloudInstance] = {}
        self.price_history: Dict[Tuple[CloudProvider, InstanceType], List[Tuple[float, float]]] = {}
        self.interruption_events: List[Dict[str, Any]] = []
        
        # Configuration
        self.price_monitoring_interval = 300.0  # 5 minutes
        self.interruption_check_interval = 60.0  # 1 minute
        self.is_monitoring = False
        
        # Thresholds
        self.max_price_volatility = 0.3  # 30% price volatility threshold
        self.interruption_probability_threshold = 0.1  # 10% interruption risk
        
        logger.info("SpotInstanceManager initialized")
    
    async def start_monitoring(self) -> None:
        """Start spot instance monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        logger.info("Starting spot instance monitoring")
        
        # Start monitoring loops
        asyncio.create_task(self._price_monitoring_loop())
        asyncio.create_task(self._interruption_monitoring_loop())
    
    async def stop_monitoring(self) -> None:
        """Stop spot instance monitoring."""
        self.is_monitoring = False
        logger.info("Spot instance monitoring stopped")
    
    async def launch_spot_instance(
        self,
        provider: CloudProvider,
        instance_type: InstanceType,
        max_price: float,
        availability_zone: str = None
    ) -> Optional[CloudInstance]:
        """Launch spot instance with price protection."""
        if provider not in self.providers:
            logger.error(f"Provider {provider.value} not available")
            return None
        
        api = self.providers[provider]
        
        # Check current spot price
        current_price = await self._get_current_spot_price(
            provider, instance_type, availability_zone or "us-east-1a"
        )
        
        if current_price > max_price:
            logger.warning(f"Spot price ${current_price:.4f} exceeds max ${max_price:.4f}")
            return None
        
        # Launch spot instance
        try:
            instance = await api.launch_instance(
                instance_type, 
                availability_zone or "us-east-1a", 
                spot_instance=True
            )
            
            self.spot_instances[instance.id] = instance
            logger.info(f"Launched spot instance {instance.id} at ${current_price:.4f}/hour")
            
            return instance
            
        except Exception as e:
            logger.error(f"Failed to launch spot instance: {e}")
            return None
    
    async def _price_monitoring_loop(self) -> None:
        """Monitor spot price changes."""
        while self.is_monitoring:
            try:
                for provider, api in self.providers.items():
                    for instance_type in InstanceType:
                        await self._update_price_history(provider, api, instance_type)
                
                await asyncio.sleep(self.price_monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in price monitoring: {e}")
                await asyncio.sleep(self.price_monitoring_interval)
    
    async def _interruption_monitoring_loop(self) -> None:
        """Monitor for spot instance interruptions."""
        while self.is_monitoring:
            try:
                await self._check_interruption_risk()
                await asyncio.sleep(self.interruption_check_interval)
                
            except Exception as e:
                logger.error(f"Error in interruption monitoring: {e}")
                await asyncio.sleep(self.interruption_check_interval)
    
    async def _update_price_history(
        self,
        provider: CloudProvider,
        api: CloudProviderAPI,
        instance_type: InstanceType
    ) -> None:
        """Update price history for provider/instance type."""
        try:
            price_history = await api.get_spot_price_history(
                instance_type, "us-east-1a", hours=1
            )
            
            key = (provider, instance_type)
            if key not in self.price_history:
                self.price_history[key] = []
            
            # Add new price data
            self.price_history[key].extend(price_history)
            
            # Keep only recent history (24 hours)
            cutoff_time = time.time() - 24 * 3600
            self.price_history[key] = [
                (timestamp, price) for timestamp, price in self.price_history[key]
                if timestamp > cutoff_time
            ]
            
        except Exception as e:
            logger.debug(f"Failed to update price history for {provider.value}/{instance_type.value}: {e}")
    
    async def _check_interruption_risk(self) -> None:
        """Check interruption risk for running spot instances."""
        for instance_id, instance in self.spot_instances.items():
            if not instance.is_spot_available():
                continue
            
            # Calculate interruption probability
            interruption_prob = await self._estimate_interruption_probability(instance)
            
            if interruption_prob > self.interruption_probability_threshold:
                logger.warning(
                    f"High interruption risk ({interruption_prob:.1%}) for instance {instance_id}"
                )
                
                # Consider migration or backup launch
                await self._handle_high_interruption_risk(instance)
    
    async def _estimate_interruption_probability(self, instance: CloudInstance) -> float:
        """Estimate interruption probability for spot instance."""
        key = (instance.provider, instance.instance_type)
        
        if key not in self.price_history or len(self.price_history[key]) < 10:
            return 0.05  # Default low risk
        
        # Analyze price volatility
        prices = [price for _, price in self.price_history[key][-10:]]  # Last 10 data points
        price_volatility = np.std(prices) / np.mean(prices) if prices else 0
        
        # Higher volatility = higher interruption risk
        base_risk = min(price_volatility / self.max_price_volatility, 1.0) * 0.2
        
        # Add provider-specific factors
        provider_risk_multipliers = {
            CloudProvider.AWS: 1.0,
            CloudProvider.GOOGLE_CLOUD: 0.8,  # More stable preemptible pricing
            CloudProvider.AZURE: 1.2  # More variable spot pricing
        }
        
        multiplier = provider_risk_multipliers.get(instance.provider, 1.0)
        
        return min(base_risk * multiplier, 1.0)
    
    async def _handle_high_interruption_risk(self, instance: CloudInstance) -> None:
        """Handle high interruption risk."""
        # Strategy 1: Launch backup on-demand instance
        if instance.provider in self.providers:
            api = self.providers[instance.provider]
            
            try:
                backup_instance = await api.launch_instance(
                    instance.instance_type,
                    instance.availability_zone,
                    spot_instance=False  # On-demand backup
                )
                
                logger.info(f"Launched backup instance {backup_instance.id} for {instance.id}")
                
                # Record the backup relationship
                self.interruption_events.append({
                    "timestamp": time.time(),
                    "type": "backup_launched",
                    "spot_instance_id": instance.id,
                    "backup_instance_id": backup_instance.id,
                    "reason": "high_interruption_risk"
                })
                
            except Exception as e:
                logger.error(f"Failed to launch backup instance: {e}")
    
    async def _get_current_spot_price(
        self,
        provider: CloudProvider,
        instance_type: InstanceType,
        availability_zone: str
    ) -> float:
        """Get current spot price."""
        key = (provider, instance_type)
        
        if key in self.price_history and self.price_history[key]:
            # Return most recent price
            return self.price_history[key][-1][1]
        
        # Fallback: get fresh price data
        if provider in self.providers:
            api = self.providers[provider]
            try:
                history = await api.get_spot_price_history(
                    instance_type, availability_zone, hours=1
                )
                if history:
                    return history[-1][1]
            except Exception as e:
                logger.debug(f"Failed to get current spot price: {e}")
        
        # Default fallback price
        return 0.05
    
    def analyze_spot_savings(
        self,
        provider: CloudProvider,
        instance_type: InstanceType,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Analyze potential savings from spot instances."""
        key = (provider, instance_type)
        
        if key not in self.price_history:
            return {"error": "No price history available"}
        
        cutoff_time = time.time() - hours * 3600
        recent_prices = [
            price for timestamp, price in self.price_history[key]
            if timestamp > cutoff_time
        ]
        
        if not recent_prices:
            return {"error": "No recent price data"}
        
        # Get on-demand price (approximation)
        if provider == CloudProvider.AWS:
            api = self.providers[provider]
            on_demand_price = api.instance_type_mapping[instance_type]["cost"]
        else:
            # Estimate on-demand price as 3x average spot price
            on_demand_price = np.mean(recent_prices) * 3
        
        avg_spot_price = np.mean(recent_prices)
        min_spot_price = np.min(recent_prices)
        max_spot_price = np.max(recent_prices)
        
        avg_savings = (on_demand_price - avg_spot_price) / on_demand_price
        best_savings = (on_demand_price - min_spot_price) / on_demand_price
        worst_savings = (on_demand_price - max_spot_price) / on_demand_price
        
        price_volatility = np.std(recent_prices) / avg_spot_price
        
        return {
            "provider": provider.value,
            "instance_type": instance_type.value,
            "analysis_period_hours": hours,
            "on_demand_price": on_demand_price,
            "spot_price_stats": {
                "average": avg_spot_price,
                "minimum": min_spot_price,
                "maximum": max_spot_price,
                "volatility": price_volatility
            },
            "savings_analysis": {
                "average_savings": avg_savings,
                "best_case_savings": best_savings,
                "worst_case_savings": worst_savings,
                "monthly_savings_estimate": avg_savings * on_demand_price * 24 * 30
            },
            "recommendation": self._generate_spot_recommendation(avg_savings, price_volatility)
        }
    
    def _generate_spot_recommendation(
        self,
        avg_savings: float,
        volatility: float
    ) -> str:
        """Generate spot instance recommendation."""
        if avg_savings > 0.7 and volatility < 0.2:
            return "Highly recommended - excellent savings with low risk"
        elif avg_savings > 0.5 and volatility < 0.3:
            return "Recommended - good savings with acceptable risk"
        elif avg_savings > 0.3:
            return "Consider with caution - moderate savings but higher risk"
        else:
            return "Not recommended - insufficient savings for the risk"
    
    def get_spot_instance_stats(self) -> Dict[str, Any]:
        """Get comprehensive spot instance statistics."""
        running_spots = len([i for i in self.spot_instances.values() if i.is_running()])
        total_spots = len(self.spot_instances)
        
        # Calculate total cost savings
        total_savings = 0.0
        total_runtime_hours = 0.0
        
        for instance in self.spot_instances.values():
            if instance.launch_time:
                runtime = (time.time() - instance.launch_time) / 3600
                total_runtime_hours += runtime
                
                # Estimate on-demand equivalent cost
                on_demand_cost = runtime * (instance.hourly_cost / 0.6)  # Assume 60% discount
                actual_cost = runtime * instance.hourly_cost
                total_savings += on_demand_cost - actual_cost
        
        # Interruption statistics
        interruptions = len([
            event for event in self.interruption_events
            if event["type"] == "interruption"
        ])
        
        return {
            "active_spot_instances": running_spots,
            "total_spot_instances": total_spots,
            "total_runtime_hours": total_runtime_hours,
            "estimated_cost_savings": total_savings,
            "interruption_events": interruptions,
            "monitoring_active": self.is_monitoring,
            "tracked_price_series": len(self.price_history)
        }


# Factory function for creating cloud providers
def create_cloud_provider(
    provider: CloudProvider,
    credentials: Optional[Dict[str, str]] = None
) -> CloudProviderAPI:
    """Factory function to create cloud provider instances."""
    if provider == CloudProvider.AWS:
        return AWSProvider(credentials)
    elif provider == CloudProvider.GOOGLE_CLOUD:
        return GoogleCloudProvider(credentials)
    elif provider == CloudProvider.AZURE:
        return AzureProvider(credentials)
    else:
        raise ValueError(f"Unsupported cloud provider: {provider.value}")


# Example usage and configuration helpers
def get_recommended_instance_type(
    cpu_cores: int,
    memory_gb: float,
    gpu_required: bool = False
) -> InstanceType:
    """Recommend instance type based on requirements."""
    if gpu_required:
        return InstanceType.GPU_ENABLED
    
    if cpu_cores >= 8 or memory_gb >= 30:
        return InstanceType.XLARGE
    elif cpu_cores >= 4 or memory_gb >= 15:
        return InstanceType.LARGE
    elif cpu_cores >= 2 or memory_gb >= 4:
        return InstanceType.MEDIUM
    elif memory_gb >= 2:
        return InstanceType.SMALL
    else:
        return InstanceType.MICRO


def estimate_monthly_cost(
    instance_type: InstanceType,
    provider: CloudProvider,
    hours_per_day: int = 24,
    spot_instance: bool = False
) -> float:
    """Estimate monthly cost for instance type."""
    # Simplified cost estimation
    base_costs = {
        (CloudProvider.AWS, InstanceType.MICRO): 0.0104,
        (CloudProvider.AWS, InstanceType.SMALL): 0.0208,
        (CloudProvider.AWS, InstanceType.MEDIUM): 0.0416,
        (CloudProvider.AWS, InstanceType.LARGE): 0.0832,
        (CloudProvider.AWS, InstanceType.XLARGE): 0.1664,
        (CloudProvider.GOOGLE_CLOUD, InstanceType.MICRO): 0.0076,
        (CloudProvider.GOOGLE_CLOUD, InstanceType.SMALL): 0.0475,
        (CloudProvider.AZURE, InstanceType.MICRO): 0.0104,
        (CloudProvider.AZURE, InstanceType.SMALL): 0.0208,
    }
    
    hourly_cost = base_costs.get((provider, instance_type), 0.05)
    
    if spot_instance:
        hourly_cost *= 0.4  # Assume 60% discount for spot
    
    monthly_hours = hours_per_day * 30
    return hourly_cost * monthly_hours