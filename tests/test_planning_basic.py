"""
Basic tests for quantum-inspired task planning functionality

Tests core planning components with minimal dependencies.
"""

import pytest
import sys
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Test basic functionality that doesn't require JAX/NumPy
def test_basic_imports():
    """Test that basic planning structures can be imported"""
    try:
        from qem_bench.planning.core import Task, TaskState, PlanningConfig
        from qem_bench.planning.validation import ValidationError, ErrorSeverity
        from qem_bench.planning.recovery import FaultType, RecoveryStrategy
        assert True  # If we get here, imports work
    except ImportError as e:
        pytest.skip(f"Planning modules not available: {e}")


def test_task_creation():
    """Test basic task creation and properties"""
    try:
        from qem_bench.planning.core import Task, TaskState
        
        # Create basic task
        task = Task(
            id="test_task_1",
            name="Test Task",
            complexity=5.0,
            priority=0.8,
            dependencies=["task_0"],
            resources={'cpu': 2.0, 'memory': 1.5}
        )
        
        # Verify properties
        assert task.id == "test_task_1"
        assert task.name == "Test Task"
        assert task.complexity == 5.0
        assert task.priority == 0.8
        assert "task_0" in task.dependencies
        assert task.resources['cpu'] == 2.0
        assert task.state == TaskState.SUPERPOSITION
        
    except ImportError:
        pytest.skip("Planning core not available")


def test_planning_config():
    """Test planning configuration"""
    try:
        from qem_bench.planning.core import PlanningConfig
        
        # Default config
        config = PlanningConfig()
        assert config.max_iterations > 0
        assert 0 < config.convergence_threshold < 1
        
        # Custom config
        custom_config = PlanningConfig(
            max_iterations=500,
            convergence_threshold=1e-5,
            superposition_width=0.2,
            entanglement_strength=0.7
        )
        
        assert custom_config.max_iterations == 500
        assert custom_config.convergence_threshold == 1e-5
        assert custom_config.superposition_width == 0.2
        assert custom_config.entanglement_strength == 0.7
        
    except ImportError:
        pytest.skip("Planning core not available")


def test_validation_error():
    """Test validation error structures"""
    try:
        from qem_bench.planning.validation import ValidationError, ErrorSeverity
        
        error = ValidationError(
            error_id="test_error",
            severity=ErrorSeverity.MEDIUM,
            category="test",
            message="Test error message",
            task_id="task_1",
            suggestion="Fix the issue"
        )
        
        assert error.error_id == "test_error"
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.category == "test"
        assert error.recoverable is True  # Default
        assert isinstance(error.timestamp, datetime)
        
    except ImportError:
        pytest.skip("Planning validation not available")


def test_recovery_strategies():
    """Test recovery strategy enums"""
    try:
        from qem_bench.planning.recovery import RecoveryStrategy, FaultType
        
        # Test strategy enum
        assert RecoveryStrategy.RETRY_WITH_BACKOFF.value == "retry_with_backoff"
        assert RecoveryStrategy.FALLBACK_ALGORITHM.value == "fallback_algorithm"
        assert RecoveryStrategy.QUANTUM_ERROR_CORRECTION.value == "quantum_error_correction"
        
        # Test fault type enum
        assert FaultType.CONVERGENCE_FAILURE.value == "convergence_failure"
        assert FaultType.RESOURCE_EXHAUSTION.value == "resource_exhaustion"
        assert FaultType.QUANTUM_DECOHERENCE.value == "quantum_decoherence"
        
    except ImportError:
        pytest.skip("Planning recovery not available")


def test_performance_config():
    """Test performance configuration"""
    try:
        from qem_bench.planning.performance import PerformanceConfig, ComputeBackend
        
        # Default performance config
        perf_config = PerformanceConfig()
        assert perf_config.backend == ComputeBackend.CPU
        assert perf_config.memory_limit_gb > 0
        assert perf_config.optimization_level >= 0
        
        # Custom performance config
        custom_perf = PerformanceConfig(
            backend=ComputeBackend.GPU,
            memory_limit_gb=16.0,
            max_workers=8,
            enable_jit=True,
            cache_size_mb=2048
        )
        
        assert custom_perf.backend == ComputeBackend.GPU
        assert custom_perf.memory_limit_gb == 16.0
        assert custom_perf.max_workers == 8
        assert custom_perf.enable_jit is True
        
    except ImportError:
        pytest.skip("Planning performance not available")


def test_qem_task_creation():
    """Test QEM-specific task creation"""
    try:
        from qem_bench.planning.integration import QEMTask
        
        qem_task = QEMTask(
            id="qem_test",
            name="QEM Test Task", 
            complexity=3.0,
            circuit_complexity=5.0,
            noise_resilience=0.8,
            requires_zne=True,
            requires_pec=False,
            requires_vd=True
        )
        
        assert qem_task.id == "qem_test"
        assert qem_task.circuit_complexity == 5.0
        assert qem_task.noise_resilience == 0.8
        assert qem_task.requires_zne is True
        assert qem_task.requires_pec is False
        assert qem_task.requires_vd is True
        
    except ImportError:
        pytest.skip("QEM integration not available")


def test_dependency_validation():
    """Test task dependency validation logic"""
    try:
        from qem_bench.planning.core import Task
        
        # Create tasks with dependencies
        task1 = Task(id="task1", name="Task 1", complexity=1.0)
        task2 = Task(id="task2", name="Task 2", complexity=2.0, dependencies=["task1"])
        task3 = Task(id="task3", name="Task 3", complexity=3.0, dependencies=["task1", "task2"])
        
        tasks = {"task1": task1, "task2": task2, "task3": task3}
        
        # Basic dependency checks
        assert len(task1.dependencies) == 0
        assert "task1" in task2.dependencies
        assert "task1" in task3.dependencies
        assert "task2" in task3.dependencies
        
        # Verify no circular dependencies in this simple case
        def has_circular_deps(task_dict):
            for task_id, task in task_dict.items():
                if task_id in task.dependencies:
                    return True  # Self-dependency
            return False
        
        assert not has_circular_deps(tasks)
        
    except ImportError:
        pytest.skip("Planning core not available")


def test_cache_key_generation():
    """Test cache key generation for reproducible results"""
    import hashlib
    
    def create_mock_cache_key(tasks: Dict[str, Any], config_str: str) -> str:
        """Mock implementation of cache key generation"""
        task_hash = hashlib.md5()
        
        # Sort for deterministic ordering
        for task_id in sorted(tasks.keys()):
            task_str = f"{task_id}:{tasks[task_id]}"
            task_hash.update(task_str.encode())
        
        combined = task_hash.hexdigest() + config_str
        return hashlib.md5(combined.encode()).hexdigest()
    
    # Test consistent key generation
    tasks1 = {"task_a": "config_a", "task_b": "config_b"}
    tasks2 = {"task_b": "config_b", "task_a": "config_a"}  # Different order
    
    key1 = create_mock_cache_key(tasks1, "config")
    key2 = create_mock_cache_key(tasks2, "config") 
    
    # Should generate same key regardless of order
    assert key1 == key2
    
    # Different config should generate different key
    key3 = create_mock_cache_key(tasks1, "different_config")
    assert key1 != key3


def test_memory_stats_mock():
    """Test memory statistics collection (mock)"""
    def mock_get_memory_stats():
        """Mock memory statistics"""
        return {
            'limit_gb': 8.0,
            'used_gb': 2.5,
            'available_gb': 5.5,
            'utilization': 0.3125,
            'active_allocations': 3
        }
    
    stats = mock_get_memory_stats()
    
    assert stats['limit_gb'] == 8.0
    assert stats['used_gb'] + stats['available_gb'] == stats['limit_gb']
    assert 0 <= stats['utilization'] <= 1
    assert stats['active_allocations'] >= 0


def test_basic_scheduling_logic():
    """Test basic scheduling logic without quantum components"""
    try:
        from qem_bench.planning.core import Task
        
        # Create simple tasks
        tasks = [
            Task(id="high_priority", name="High Priority", complexity=1.0, priority=0.9),
            Task(id="med_priority", name="Medium Priority", complexity=2.0, priority=0.5),
            Task(id="low_priority", name="Low Priority", complexity=1.5, priority=0.2)
        ]
        
        # Simple priority-based sorting (mock scheduling)
        sorted_by_priority = sorted(tasks, key=lambda t: -t.priority)
        
        assert sorted_by_priority[0].id == "high_priority"
        assert sorted_by_priority[1].id == "med_priority" 
        assert sorted_by_priority[2].id == "low_priority"
        
        # Simple complexity-based sorting
        sorted_by_complexity = sorted(tasks, key=lambda t: t.complexity)
        
        assert sorted_by_complexity[0].complexity == 1.0
        assert sorted_by_complexity[-1].complexity == 2.0
        
    except ImportError:
        pytest.skip("Planning core not available")


if __name__ == "__main__":
    # Run basic tests manually if pytest not available
    test_functions = [
        test_basic_imports,
        test_task_creation, 
        test_planning_config,
        test_validation_error,
        test_recovery_strategies,
        test_performance_config,
        test_qem_task_creation,
        test_dependency_validation,
        test_cache_key_generation,
        test_memory_stats_mock,
        test_basic_scheduling_logic
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test_func in test_functions:
        try:
            test_func()
            print(f"✅ {test_func.__name__}")
            passed += 1
        except pytest.skip.Exception as e:
            print(f"⏭️  {test_func.__name__}: {e}")
            skipped += 1
        except Exception as e:
            print(f"❌ {test_func.__name__}: {e}")
            failed += 1
    
    print(f"\nTest Results: {passed} passed, {failed} failed, {skipped} skipped")