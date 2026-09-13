import os
import shutil
import unittest

from streamtrain.config import (
    CheckpointConfig,
    LoggingConfig,
    ModelConfig,
    ResourcesConfig,
    StreamTrainConfig,
    TrainingConfig,
)
from streamtrain.exceptions import OOMError
from streamtrain.optimization.batch_size import BatchSizeOptimizer
from streamtrain.recovery.checkpoint import CheckpointManager


class TestStreamTrain(unittest.TestCase):
    def setUp(self):
        self.checkpoint_dir = "./test_checkpoints"
        if os.path.exists(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir)
            
        self.config = StreamTrainConfig(
            model=ModelConfig(name="test"),
            training=TrainingConfig(epochs=2, batch_size=8, gradient_accumulation=1, precision="fp32"),
            resources=ResourcesConfig(mode="balanced", max_ram="2GB", max_vram="1GB", max_cpu_percent=90),
            checkpoint=CheckpointConfig(directory=self.checkpoint_dir, interval_steps=5, keep_last=2),
            logging=LoggingConfig(directory="./test_logs", interval_steps=1)
        )

    def tearDown(self):
        if os.path.exists(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir)

    def test_batch_optimizer(self):
        optimizer = BatchSizeOptimizer(initial_batch_size=16)
        self.assertEqual(optimizer.current_batch_size, 16)
        
        # Test reduction
        new_bs = optimizer.reduce_batch_size()
        self.assertEqual(new_bs, 8)
        self.assertEqual(optimizer.current_batch_size, 8)
        
        # Test min bounds
        optimizer.current_batch_size = 1
        with self.assertRaises(OOMError):
            optimizer.reduce_batch_size()

    def test_checkpointing(self):
        manager = CheckpointManager(directory=self.checkpoint_dir, keep_last=2)
        
        # Save dummy states
        state1 = {"model_state": {"a": 1}}
        manager.save(state1, 10)
        
        state2 = {"model_state": {"a": 2}}
        manager.save(state2, 20)
        
        state3 = {"model_state": {"a": 3}}
        manager.save(state3, 30)
        
        # Check keep_last (should only have step 20 and 30)
        files = manager._get_all_checkpoints()
        self.assertEqual(len(files), 2)
        self.assertTrue("step_20.pt" in files[0])
        self.assertTrue("step_30.pt" in files[1])
        
        # Load latest
        loaded = manager.load_latest()
        self.assertEqual(loaded["model_state"]["a"], 3)

if __name__ == '__main__':
    unittest.main()
