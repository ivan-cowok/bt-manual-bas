"""
Batch Size Profiler

Auto-detect optimal batch size through progressive testing.
Pure ONNX Runtime - no framework dependencies!
"""

import onnxruntime as ort
import numpy as np
import time
import subprocess
import psutil
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .device_detector import DeviceInfo


@dataclass
class ProfilingResult:
    """Result of batch size profiling"""
    batch_size: int
    avg_time_ms: float
    throughput_fps: float
    memory_mb: float
    success: bool
    error: Optional[str] = None


class StandaloneBatchProfiler:
    """
    Profile batch sizes using pure ONNX Runtime.
    Tests different batch sizes to find optimal configuration.
    """
    
    def __init__(self, model_path: str, device_info: DeviceInfo):
        """
        Initialize profiler.
        
        Args:
            model_path: Path to ONNX model
            device_info: Device information
        """
        self.model_path = model_path
        self.device_info = device_info
        self.results: List[ProfilingResult] = []
    
    def get_current_memory_mb(self) -> float:
        """Get current memory usage (GPU or RAM)"""
        if self.device_info.has_gpu:
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=memory.used',
                     '--format=csv,noheader,nounits'],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0:
                    return float(result.stdout.strip())
            except:
                pass
        
        # Fallback to RAM
        process = psutil.Process()
        return process.memory_info().rss / (1024 ** 2)
    
    def create_session(self) -> ort.InferenceSession:
        """Create ONNX Runtime session"""
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        if self.device_info.has_gpu:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        return ort.InferenceSession(
            self.model_path,
            sess_options=sess_options,
            providers=providers
        )
    
    def profile_batch_size(
        self,
        batch_size: int,
        input_shape: Tuple,
        num_warmup: int = 3,
        num_iterations: int = 10
    ) -> ProfilingResult:
        """
        Profile specific batch size.
        
        Args:
            batch_size: Batch size to test
            input_shape: Input shape without batch (e.g., (3, 640, 640))
            num_warmup: Warmup iterations
            num_iterations: Test iterations
        
        Returns:
            ProfilingResult with metrics
        """
        try:
            session = self.create_session()
            input_name = session.get_inputs()[0].name
            
            # Create test input
            full_shape = (batch_size,) + input_shape
            test_input = np.random.randn(*full_shape).astype(np.float32)
            
            # Measure initial memory
            mem_before = self.get_current_memory_mb()
            
            # Warmup
            for _ in range(num_warmup):
                _ = session.run(None, {input_name: test_input})
            
            # Timed runs
            times = []
            for _ in range(num_iterations):
                start = time.perf_counter()
                _ = session.run(None, {input_name: test_input})
                elapsed = time.perf_counter() - start
                times.append(elapsed)
            
            # Measure final memory
            mem_after = self.get_current_memory_mb()
            
            # Calculate metrics
            avg_time = np.mean(times)
            avg_time_ms = avg_time * 1000
            throughput = batch_size / avg_time
            memory_used = max(0, mem_after - mem_before)
            
            return ProfilingResult(
                batch_size=batch_size,
                avg_time_ms=avg_time_ms,
                throughput_fps=throughput,
                memory_mb=memory_used,
                success=True
            )
        
        except Exception as e:
            return ProfilingResult(
                batch_size=batch_size,
                avg_time_ms=0,
                throughput_fps=0,
                memory_mb=0,
                success=False,
                error=str(e)
            )
    
    def find_optimal_batch_size(
        self,
        input_shape: Tuple,
        min_batch: int = 1,
        max_batch: int = 32
    ) -> int:
        """
        Find optimal batch size through progressive testing.
        
        Args:
            input_shape: Model input shape (e.g., (3, 640, 640))
            min_batch: Minimum batch size
            max_batch: Maximum batch size
        
        Returns:
            Optimal batch size
        """
        print(f"  Testing batch sizes {min_batch} to {max_batch}...")
        
        # Test powers of 2
        test_sizes = []
        size = min_batch
        while size <= max_batch:
            test_sizes.append(size)
            size *= 2
        
        # Profile each size
        self.results = []
        for batch_size in test_sizes:
            print(f"    Batch {batch_size}...", end=" ", flush=True)
            
            result = self.profile_batch_size(batch_size, input_shape)
            self.results.append(result)
            
            if result.success:
                print(f"✓ {result.avg_time_ms:.1f}ms ({result.throughput_fps:.0f} FPS)")
            else:
                print(f"✗ {result.error}")
                if "memory" in result.error.lower():
                    break
        
        # Find best
        successful = [r for r in self.results if r.success]
        
        if not successful:
            return min_batch
        
        best = max(successful, key=lambda r: r.throughput_fps)
        
        print(f"  → Optimal: batch_size={best.batch_size} ({best.throughput_fps:.0f} FPS)")
        
        return best.batch_size

