import os
import glob
import torch
import shutil
from typing import Dict, Any, Optional
from ..exceptions import CheckpointError

class CheckpointManager:
    def __init__(self, directory: str, keep_last: int = 3):
        self.directory = directory
        self.keep_last = keep_last
        os.makedirs(self.directory, exist_ok=True)

    def _get_checkpoint_path(self, step: int) -> str:
        return os.path.join(self.directory, f"step_{step}.pt")

    def save(self, state: Dict[str, Any], step: int):
        """
        Atomically saves the state to disk and enforces retention policy.
        """
        path = self._get_checkpoint_path(step)
        tmp_path = path + ".tmp"
        
        try:
            torch.save(state, tmp_path)
            # Atomic rename (on Windows, os.replace replaces existing, rename can fail if exists)
            os.replace(tmp_path, path)
        except Exception as e:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise CheckpointError(f"Failed to save checkpoint at step {step}: {e}")

        self._cleanup()

    def load_latest(self) -> Optional[Dict[str, Any]]:
        """
        Loads the latest valid checkpoint.
        """
        checkpoints = self._get_all_checkpoints()
        if not checkpoints:
            return None
            
        latest_path = checkpoints[-1]
        try:
            return torch.load(latest_path, map_location="cpu", weights_only=False)
        except Exception as e:
            raise CheckpointError(f"Failed to load checkpoint {latest_path}: {e}")

    def load(self, step: int) -> Dict[str, Any]:
        """
        Loads a specific checkpoint.
        """
        path = self._get_checkpoint_path(step)
        if not os.path.exists(path):
            raise CheckpointError(f"Checkpoint for step {step} not found at {path}")
            
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            raise CheckpointError(f"Failed to load checkpoint {path}: {e}")

    def _get_all_checkpoints(self):
        """Returns sorted list of checkpoint paths based on step number."""
        pattern = os.path.join(self.directory, "step_*.pt")
        files = glob.glob(pattern)
        
        def extract_step(filepath: str) -> int:
            basename = os.path.basename(filepath)
            # step_123.pt -> 123
            try:
                return int(basename.replace("step_", "").replace(".pt", ""))
            except ValueError:
                return -1
                
        files = [f for f in files if extract_step(f) >= 0]
        files.sort(key=extract_step)
        return files

    def _cleanup(self):
        """Removes older checkpoints to enforce keep_last policy."""
        if self.keep_last <= 0:
            return
            
        files = self._get_all_checkpoints()
        if len(files) > self.keep_last:
            files_to_remove = files[:-self.keep_last]
            for f in files_to_remove:
                try:
                    os.remove(f)
                except OSError:
                    pass
