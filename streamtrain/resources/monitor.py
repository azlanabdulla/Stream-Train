import platform
import shutil
import warnings
from typing import Any, Dict, Optional

import psutil

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

def _parse_size(size_str: Optional[str]) -> Optional[int]:
    """Parse sizes like '4GB', '3.5MB' into bytes."""
    if not size_str:
        return None
    if isinstance(size_str, int):
        return size_str
    
    size_str = size_str.upper().strip()
    if size_str.endswith("GB"):
        return int(float(size_str[:-2]) * (1024**3))
    elif size_str.endswith("MB"):
        return int(float(size_str[:-2]) * (1024**2))
    elif size_str.endswith("KB"):
        return int(float(size_str[:-2]) * 1024)
    elif size_str.endswith("B"):
        return int(float(size_str[:-1]))
    else:
        try:
            return int(size_str)
        except ValueError:
            warnings.warn(f"Could not parse size string: {size_str}")
            return None

class ResourceMonitor:
    def __init__(self, max_ram: Optional[str] = None, max_vram: Optional[str] = None, max_cpu_percent: Optional[int] = None):
        self.max_ram_bytes = _parse_size(max_ram)
        self.max_vram_bytes = _parse_size(max_vram)
        self.max_cpu_percent = max_cpu_percent

    def get_system_info(self) -> Dict[str, Any]:
        info = {
            "os": f"{platform.system()} {platform.release()}",
            "cpu": platform.processor(),
            "ram_total_gb": psutil.virtual_memory().total / (1024**3),
        }
        try:
            total, used, free = shutil.disk_usage("/")
            info["disk_free_gb"] = free / (1024**3)
        except Exception:
            info["disk_free_gb"] = "Unavailable"

        if _TORCH_AVAILABLE:
            info["pytorch_version"] = torch.__version__
            info["cuda_available"] = torch.cuda.is_available()
            if info["cuda_available"]:
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["vram_total_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        else:
            info["pytorch_version"] = "Not installed"
            info["cuda_available"] = False
            
        return info
        
    def check_ram_pressure(self) -> bool:
        """Returns True if RAM usage exceeds the configured maximum."""
        if self.max_ram_bytes is None:
            return False
        mem = psutil.virtual_memory()
        # We check against total used or available depending on strategy.
        # simpler: if (total - available) > max_ram_bytes
        used = mem.total - mem.available
        return used > self.max_ram_bytes

    def get_current_ram_usage_gb(self) -> float:
        mem = psutil.virtual_memory()
        return (mem.total - mem.available) / (1024**3)

    def check_cpu_pressure(self) -> bool:
        if self.max_cpu_percent is None:
            return False
        return psutil.cpu_percent(interval=None) > self.max_cpu_percent

    def get_current_vram_usage_gb(self, device_id: int = 0) -> float:
        if not _TORCH_AVAILABLE or not torch.cuda.is_available():
            return 0.0
        # memory_allocated is current tensor memory
        return torch.cuda.memory_allocated(device_id) / (1024**3)
    
    def check_vram_pressure(self, device_id: int = 0) -> bool:
        """Returns True if VRAM usage exceeds the configured maximum."""
        if not _TORCH_AVAILABLE or not torch.cuda.is_available() or self.max_vram_bytes is None:
            return False
        used = torch.cuda.memory_allocated(device_id)
        return used > self.max_vram_bytes
