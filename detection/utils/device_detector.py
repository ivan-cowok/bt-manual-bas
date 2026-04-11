"""
Standalone Device Detector

Detects GPU/CPU capabilities using ONLY system tools.
NO PyTorch, NO TensorFlow dependencies!
"""

import onnxruntime as ort
import subprocess
import platform
import psutil
from typing import Optional, Dict, List
from dataclasses import dataclass


@dataclass
class DeviceInfo:
    """Device information"""
    # Providers
    available_providers: List[str]
    selected_provider: str
    
    # GPU
    has_gpu: bool               # True only when hardware + onnxruntime-gpu both present
    gpu_name: Optional[str]
    gpu_memory_total_mb: Optional[float]
    gpu_memory_free_mb: Optional[float]
    
    # CPU
    cpu_name: str
    cpu_cores: int
    cpu_threads: int
    
    # RAM
    ram_total_mb: float
    ram_available_mb: float

    # Raw diagnostic flags (used to explain why GPU is unavailable)
    _has_nvidia_gpu: bool = False       # nvidia-smi found a GPU
    _has_cuda_provider: bool = False    # onnxruntime-gpu is installed


class StandaloneDeviceDetector:
    """
    Device detection using ONLY standard tools.
    No deep learning framework dependencies!
    """
    
    @staticmethod
    def detect_onnx_providers() -> List[str]:
        """Detect available ONNX Runtime providers"""
        return ort.get_available_providers()
    
    @staticmethod
    def detect_gpu_via_nvidia_smi() -> Optional[Dict]:
        """Detect NVIDIA GPU using nvidia-smi command"""
        try:
            cmd = [
                'nvidia-smi',
                '--query-gpu=index,name,memory.total,memory.free,memory.used',
                '--format=csv,noheader,nounits'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5,
                check=True
            )
            
            if result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                parts = [p.strip() for p in lines[0].split(',')]
                
                return {
                    'index': int(parts[0]),
                    'name': parts[1],
                    'memory_total_mb': float(parts[2]),
                    'memory_free_mb': float(parts[3]),
                    'memory_used_mb': float(parts[4])
                }
        
        except Exception:
            return None
        
        return None
    
    @staticmethod
    def detect_cpu_info() -> Dict:
        """Detect CPU information"""
        cpu_name = "Unknown CPU"
        try:
            if platform.system() == "Windows":
                import winreg
                key = winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
                )
                cpu_name = winreg.QueryValueEx(key, "ProcessorNameString")[0]
                winreg.CloseKey(key)
            elif platform.system() == "Linux":
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if 'model name' in line:
                            cpu_name = line.split(':')[1].strip()
                            break
        except Exception:
            pass
        
        return {
            'name': cpu_name,
            'cores': psutil.cpu_count(logical=False) or 1,
            'threads': psutil.cpu_count(logical=True) or 1,
        }
    
    @staticmethod
    def detect_ram_info() -> Dict:
        """Detect RAM information"""
        mem = psutil.virtual_memory()
        
        return {
            'total_mb': mem.total / (1024 ** 2),
            'available_mb': mem.available / (1024 ** 2),
        }
    
    @classmethod
    def detect_all(cls) -> DeviceInfo:
        """Complete device detection"""
        providers = cls.detect_onnx_providers()
        has_cuda_provider = 'CUDAExecutionProvider' in providers

        # Always probe nvidia-smi regardless of onnxruntime build,
        # so we can give a useful diagnostic when the GPU exists but
        # onnxruntime-gpu is not installed.
        gpu_info = cls.detect_gpu_via_nvidia_smi()
        has_gpu = has_cuda_provider  # GPU usable only when both conditions met

        cpu_info = cls.detect_cpu_info()
        ram_info = cls.detect_ram_info()
        
        selected_provider = 'CUDAExecutionProvider' if has_gpu else 'CPUExecutionProvider'
        
        return DeviceInfo(
            available_providers=providers,
            selected_provider=selected_provider,
            has_gpu=has_gpu,
            gpu_name=gpu_info['name'] if gpu_info else None,
            gpu_memory_total_mb=gpu_info['memory_total_mb'] if gpu_info else None,
            gpu_memory_free_mb=gpu_info['memory_free_mb'] if gpu_info else None,
            cpu_name=cpu_info['name'],
            cpu_cores=cpu_info['cores'],
            cpu_threads=cpu_info['threads'],
            ram_total_mb=ram_info['total_mb'],
            ram_available_mb=ram_info['available_mb'],
            # Expose raw flags so callers can explain exactly what failed
            _has_nvidia_gpu=gpu_info is not None,
            _has_cuda_provider=has_cuda_provider,
        )
    
    @classmethod
    def print_device_info(cls) -> DeviceInfo:
        """Print formatted device information"""
        info = cls.detect_all()
        
        print("="*70)
        print("DEVICE INFORMATION")
        print("="*70)
        print()
        
        print("ONNX Runtime Providers:")
        for provider in info.available_providers:
            marker = "→" if provider == info.selected_provider else " "
            print(f"  {marker} {provider}")
        print()
        
        if info.has_gpu:
            print("GPU: ✓ Available")
            print(f"  Name:          {info.gpu_name}")
            print(f"  Total VRAM:    {info.gpu_memory_total_mb:,.0f} MB")
            print(f"  Free VRAM:     {info.gpu_memory_free_mb:,.0f} MB")
        else:
            print("GPU: ✗ Not available (CPU mode)")
        print()
        
        print("CPU:")
        print(f"  Name:          {info.cpu_name}")
        print(f"  Cores:         {info.cpu_cores}")
        print(f"  Threads:       {info.cpu_threads}")
        print()
        
        print("RAM:")
        print(f"  Total:         {info.ram_total_mb:,.0f} MB")
        print(f"  Available:     {info.ram_available_mb:,.0f} MB")
        
        print("="*70)
        print()
        
        return info

