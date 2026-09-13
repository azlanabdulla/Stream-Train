import gc

from ..exceptions import OOMError

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class BatchSizeOptimizer:
    """
    Manages batch size adjustments when OOM exceptions are encountered.
    """
    def __init__(self, initial_batch_size: int, min_batch_size: int = 1):
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size

    def is_oom_exception(self, exception: Exception) -> bool:
        """
        Determines if an exception is an Out Of Memory error.
        """
        if isinstance(exception, OOMError):
            return True
            
        err_str = str(exception).lower()
        if "out of memory" in err_str:
            return True
        if "oom" in err_str:
            return True
        if "alloc" in err_str and "failed" in err_str:
            return True
            
        if _TORCH_AVAILABLE and isinstance(exception, RuntimeError):
            if "cuda out of memory" in err_str:
                return True
                
        return False

    def reduce_batch_size(self) -> int:
        """
        Reduces the batch size by half. 
        Raises OOMError if it cannot be reduced further.
        """
        new_batch_size = max(self.min_batch_size, self.current_batch_size // 2)
        
        if new_batch_size == self.current_batch_size:
            raise OOMError(f"Cannot reduce batch size below {self.min_batch_size}. Training cannot proceed.")
            
        self.current_batch_size = new_batch_size
        
        # Force garbage collection and empty CUDA cache
        gc.collect()
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return self.current_batch_size

