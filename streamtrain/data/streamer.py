import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from typing import Optional, Iterator, Any

class DataStreamer:
    """
    Wraps a PyTorch Dataset or IterableDataset to manage data loading
    without excessive buffering or RAM consumption.
    """
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_workers: int = 0,
        prefetch_factor: Optional[int] = None,
        pin_memory: bool = False
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        
        # When prefetch_factor is given but num_workers is 0, PyTorch raises an error.
        if self.num_workers == 0:
            self.prefetch_factor = None

        self.loader = self._build_loader(self.batch_size)
        self.iterator: Optional[Iterator] = None
        self.samples_yielded = 0

    def _build_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory
        )

    def update_batch_size(self, new_batch_size: int):
        """Update batch size, e.g. after an OOM event."""
        if new_batch_size != self.batch_size:
            self.batch_size = new_batch_size
            self.loader = self._build_loader(self.batch_size)
            # We don't reset the iterator here. 
            # The caller (Trainer) will manage the re-initialization 
            # if we need to resume from a checkpoint.

    def __iter__(self):
        self.iterator = iter(self.loader)
        return self

    def __next__(self) -> Any:
        if self.iterator is None:
            raise RuntimeError("DataStreamer iterator not initialized.")
        try:
            batch = next(self.iterator)
            # Rough heuristic: first dimension is batch size
            if isinstance(batch, (tuple, list)):
                self.samples_yielded += len(batch[0])
            elif isinstance(batch, dict):
                first_key = next(iter(batch.keys()))
                self.samples_yielded += len(batch[first_key])
            elif torch.is_tensor(batch):
                self.samples_yielded += len(batch)
            return batch
        except StopIteration:
            raise StopIteration
