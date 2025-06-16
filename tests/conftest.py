import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock
import torch


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_config():
    """Create a mock configuration object."""
    config = Mock()
    config.hidden_size = 768
    config.num_attention_heads = 12
    config.num_hidden_layers = 12
    config.intermediate_size = 3072
    config.vocab_size = 50257
    config.max_position_embeddings = 1024
    config.num_experts = 8
    config.num_experts_per_tok = 2
    return config


@pytest.fixture
def mock_model():
    """Create a mock model object."""
    model = MagicMock()
    model.config = mock_config()
    model.device = torch.device("cpu")
    return model


@pytest.fixture
def sample_tensor():
    """Create a sample tensor for testing."""
    return torch.randn(2, 10, 768)


@pytest.fixture
def sample_input_ids():
    """Create sample input IDs for testing."""
    return torch.randint(0, 50257, (2, 10))


@pytest.fixture
def sample_attention_mask():
    """Create sample attention mask for testing."""
    return torch.ones(2, 10, dtype=torch.long)


@pytest.fixture(autouse=True)
def reset_torch_seed():
    """Reset PyTorch random seed for reproducibility."""
    torch.manual_seed(42)
    yield
    torch.manual_seed(42)


@pytest.fixture
def mock_transformers_model():
    """Create a mock transformers model."""
    model = MagicMock()
    model.config = mock_config()
    model.forward = MagicMock(return_value=MagicMock(logits=torch.randn(2, 10, 50257)))
    return model


@pytest.fixture
def cleanup_cache(monkeypatch):
    """Clean up any cache directories created during tests."""
    cache_dirs = []
    
    def track_cache_dir(path):
        cache_dirs.append(path)
        return path
    
    monkeypatch.setattr("tempfile.mkdtemp", track_cache_dir)
    
    yield
    
    for cache_dir in cache_dirs:
        if Path(cache_dir).exists():
            shutil.rmtree(cache_dir)


@pytest.fixture
def disable_gpu():
    """Force CPU usage for tests."""
    original_cuda_visible_devices = torch.cuda.CUDA_VISIBLE_DEVICES if hasattr(torch.cuda, 'CUDA_VISIBLE_DEVICES') else None
    torch.cuda.CUDA_VISIBLE_DEVICES = ""
    yield
    if original_cuda_visible_devices is not None:
        torch.cuda.CUDA_VISIBLE_DEVICES = original_cuda_visible_devices