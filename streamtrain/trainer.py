import time
import torch
import torch.nn as nn
from typing import Optional, Callable
from torch.utils.data import Dataset
from .config import StreamTrainConfig
from .data.streamer import DataStreamer
from .resources.monitor import ResourceMonitor
from .optimization.batch_size import BatchSizeOptimizer
from .recovery.checkpoint import CheckpointManager
from .logging.logger import logger
from .exceptions import OOMError

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        dataset: Dataset,
        config: StreamTrainConfig,
        loss_fn: Callable = nn.CrossEntropyLoss(),
        device: str = "cpu"
    ):
        self.config = config
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.dataset = dataset
        self.loss_fn = loss_fn
        
        # Internal state
        self.global_step = 0
        self.current_epoch = 0
        self._stop_requested = False
        
        # Subsystems
        self.resource_monitor = ResourceMonitor(
            max_ram=config.resources.max_ram,
            max_vram=config.resources.max_vram,
            max_cpu_percent=config.resources.max_cpu_percent
        )
        
        initial_batch = 16 if config.training.batch_size == "auto" else int(config.training.batch_size)
        self.batch_optimizer = BatchSizeOptimizer(initial_batch_size=initial_batch)
        
        self.checkpoint_manager = CheckpointManager(
            directory=config.checkpoint.directory,
            keep_last=config.checkpoint.keep_last
        )
        
        self.streamer = DataStreamer(
            dataset=self.dataset,
            batch_size=self.batch_optimizer.current_batch_size
        )
        
        self.grad_accum_steps = (
            1 if config.training.gradient_accumulation == "auto" 
            else int(config.training.gradient_accumulation)
        )
        
        self.scaler = torch.amp.GradScaler(self.device.type) if self.device.type == "cuda" else None
        self.precision_context = self._get_precision_context()
        
    def _get_precision_context(self):
        if self.config.training.precision in ("fp16", "auto") and self.device.type == "cuda":
            return torch.amp.autocast(device_type=self.device.type, dtype=torch.float16)
        elif self.config.training.precision == "bf16" and self.device.type == "cuda":
            return torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16)
        else:
            from contextlib import nullcontext
            return nullcontext()

    def _save_checkpoint(self):
        state = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "epoch": self.current_epoch,
            "batch_size": self.batch_optimizer.current_batch_size,
        }
        if self.scaler:
            state["scaler_state"] = self.scaler.state_dict()
            
        self.checkpoint_manager.save(state, self.global_step)
        logger.info(f"Checkpoint saved at step {self.global_step}")

    def _load_checkpoint_state(self, state):
        self.model.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.global_step = state["global_step"]
        self.current_epoch = state.get("epoch", 0)
        
        # Restore batch size logic
        loaded_batch = state.get("batch_size")
        if loaded_batch:
            self.batch_optimizer.current_batch_size = loaded_batch
            self.streamer.update_batch_size(loaded_batch)
            
        if self.scaler and "scaler_state" in state:
            self.scaler.load_state_dict(state["scaler_state"])

    def resume(self):
        """Attempts to load the latest valid checkpoint."""
        state = self.checkpoint_manager.load_latest()
        if state:
            self._load_checkpoint_state(state)
            logger.info(f"Resumed from checkpoint at step {self.global_step}, epoch {self.current_epoch}, batch size {self.batch_optimizer.current_batch_size}")
        else:
            logger.info("No checkpoint found. Starting from scratch.")

    def stop(self):
        self._stop_requested = True

    def _check_resources(self):
        if self.resource_monitor.check_ram_pressure():
            logger.warning(f"RAM pressure detected! Usage > {self.config.resources.max_ram}")
        if self.device.type == "cuda" and self.resource_monitor.check_vram_pressure():
            logger.warning(f"VRAM pressure detected! Usage > {self.config.resources.max_vram}")
            
    def _handle_oom(self, exception: Exception):
        if not self.batch_optimizer.is_oom_exception(exception):
            raise exception
            
        logger.warning(f"OOM detected: {exception}")
        old_bs = self.batch_optimizer.current_batch_size
        new_bs = self.batch_optimizer.reduce_batch_size()
        
        logger.info(f"Reduced batch size: {old_bs} -> {new_bs}")
        self.streamer.update_batch_size(new_bs)
        
        logger.info("Restoring state from latest checkpoint to recover...")
        self.resume()
        
    def _train_step(self, batch) -> float:
        # Simple heuristic to unpack standard batches
        if isinstance(batch, (list, tuple)):
            inputs = batch[0].to(self.device)
            targets = batch[1].to(self.device)
        elif isinstance(batch, dict):
            inputs = {k: v.to(self.device) for k, v in batch.items() if k != "target"}
            targets = batch["target"].to(self.device)
        else:
            inputs = batch.to(self.device)
            targets = inputs # dummy
            
        with self.precision_context:
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss = loss / self.grad_accum_steps
            
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
            
        return loss.item() * self.grad_accum_steps

    def train(self):
        logger.info("Training started")
        info = self.resource_monitor.get_system_info()
        logger.info(f"Device: {self.device}")
        logger.info(f"RAM limit: {self.config.resources.max_ram}")
        
        self.model.train()
        
        while self.current_epoch < self.config.training.epochs and not self._stop_requested:
            # We recreate iterator each epoch for Dataset; IterableDataset requires care but DataStreamer handles it
            try:
                # Create a fresh iterator unless we just resumed and need to seek
                # (For true dataset seeking, we need a resumable dataset. We use basic iter here)
                for step_idx, batch in enumerate(self.streamer):
                    if self._stop_requested:
                        break
                        
                    try:
                        loss_val = self._train_step(batch)
                        
                        is_accum_step = (step_idx + 1) % self.grad_accum_steps == 0
                        
                        if is_accum_step:
                            if self.scaler:
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                            else:
                                self.optimizer.step()
                            self.optimizer.zero_grad()
                            
                            self.global_step += 1
                            
                            if self.global_step % self.config.logging.interval_steps == 0:
                                logger.info(f"Epoch {self.current_epoch} | Step {self.global_step} | Loss {loss_val:.4f} | Batch {self.batch_optimizer.current_batch_size}")
                                
                            if self.global_step % self.config.checkpoint.interval_steps == 0:
                                self._save_checkpoint()
                                self._check_resources()
                                
                    except Exception as e:
                        self._handle_oom(e)
                        # Break out of inner loop to restart the iterator after restoring state
                        break 
                        
            except Exception as e:
                # Handle potential OOM during data loading or outside train step
                self._handle_oom(e)
                
            if not self._stop_requested:
                self.current_epoch += 1
                
        # Final checkpoint
        if not self._stop_requested:
            self._save_checkpoint()
        logger.info("Training stopped/completed.")
