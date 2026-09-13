class StreamTrainError(Exception):
    """Base exception for StreamTrain."""
    pass

class OOMError(StreamTrainError):
    """Raised when an out-of-memory condition is detected and cannot be automatically resolved."""
    pass

class ConfigurationError(StreamTrainError):
    """Raised when an invalid configuration is provided."""
    pass

class CheckpointError(StreamTrainError):
    """Raised when a checkpoint cannot be saved or loaded correctly."""
    pass
