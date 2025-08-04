"""
Pytest fixtures and configuration for MicroMind tests.
"""
import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock, patch
from argparse import Namespace
from pathlib import Path
import tempfile
import shutil

# Import MicroMind modules
from micromind.core import MicroMind, Metric
from micromind.enum import Stage
from micromind.callbacks import (
    TrainingState, TrainingCallback, CallbackManager,
    ProgressCallback, MetricsCallback
)


class DummyMicroMind(MicroMind):
    """Concrete implementation of MicroMind for testing."""
    
    def __init__(self, hparams=None, disable_progress=True):
        super().__init__(hparams, disable_progress)
        # Simple linear model for testing
        self.modules = torch.nn.ModuleDict({
            'classifier': nn.Linear(10, 2)
        })
        self.set_input_shape((10,))
    
    def forward(self, batch):
        """Simple forward pass for testing."""
        if isinstance(batch, list):
            x = batch[0]
        else:
            x = batch
        return self.modules.classifier(x)
    
    def compute_loss(self, pred, batch):
        """Simple loss computation for testing."""
        if isinstance(batch, list):
            target = batch[1] if len(batch) > 1 else torch.zeros(pred.size(0), dtype=torch.long)
        else:
            target = torch.zeros(pred.size(0), dtype=torch.long)
        return nn.CrossEntropyLoss()(pred, target)


class DummyCallback(TrainingCallback):
    """Dummy callback for testing callback system."""
    
    def __init__(self):
        self.calls = []
    
    def on_train_start(self, trainer, state):
        self.calls.append('train_start')
    
    def on_epoch_start(self, trainer, state):
        self.calls.append('epoch_start')
    
    def on_batch_start(self, trainer, state):
        self.calls.append('batch_start')
    
    def on_batch_end(self, trainer, state):
        self.calls.append('batch_end')
    
    def on_epoch_end(self, trainer, state):
        self.calls.append('epoch_end')
    
    def on_train_end(self, trainer, state):
        self.calls.append('train_end')


@pytest.fixture
def mock_accelerator():
    """Mock accelerator for testing."""
    mock_acc = Mock()
    mock_acc.device = torch.device('cpu')
    mock_acc.is_local_main_process = True
    mock_acc.autocast.return_value.__enter__ = Mock(return_value=None)
    mock_acc.autocast.return_value.__exit__ = Mock(return_value=None)
    mock_acc.prepare = Mock(side_effect=lambda *args: args)
    mock_acc.backward = Mock()
    mock_acc.register_for_checkpointing = Mock()
    mock_acc.load_state = Mock()
    return mock_acc


@pytest.fixture
def default_hparams():
    """Default hyperparameters for testing."""
    return Namespace(
        output_folder="test_results",
        experiment_name="test_exp",
        opt="adam",
        lr=0.001,
        debug=True
    )


@pytest.fixture
def dummy_micromind(default_hparams, mock_accelerator):
    """Create a dummy MicroMind instance for testing."""
    with patch('micromind.core.Accelerator', return_value=mock_accelerator):
        model = DummyMicroMind(default_hparams)
        return model


@pytest.fixture
def training_state():
    """Create a training state for testing."""
    return TrainingState(
        epoch=1,
        batch_idx=0,
        total_epochs=5,
        total_batches=10,
        stage=Stage.train
    )


@pytest.fixture
def dummy_dataset():
    """Create a dummy dataset for testing."""
    class DummyDataset:
        def __init__(self, size=100):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __iter__(self):
            for i in range(self.size):
                # Return (input, target) pairs
                x = torch.randn(10)
                y = torch.randint(0, 2, (1,)).squeeze()
                yield [x, y]
    
    return DummyDataset()


@pytest.fixture
def datasets(dummy_dataset):
    """Create datasets dictionary for training."""
    return {
        'train': dummy_dataset,
        'val': dummy_dataset,
        'test': dummy_dataset
    }


@pytest.fixture
def dummy_metric():
    """Create a dummy metric for testing."""
    def accuracy_fn(pred, batch):
        if isinstance(batch, list):
            target = batch[1] if len(batch) > 1 else torch.zeros(pred.size(0), dtype=torch.long)
        else:
            target = torch.zeros(pred.size(0), dtype=torch.long)
        return (pred.argmax(dim=1) == target).float()
    
    return Metric("accuracy", accuracy_fn, reduction="mean")


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_checkpointer():
    """Mock checkpointer for testing."""
    mock_ckpt = Mock()
    mock_ckpt.recover_state.return_value = None
    mock_ckpt.debug = False
    return mock_ckpt


@pytest.fixture
def callback_manager():
    """Create a callback manager for testing."""
    return CallbackManager()


@pytest.fixture
def dummy_callback():
    """Create a dummy callback for testing."""
    return DummyCallback()


@pytest.fixture(autouse=True)
def mock_tqdm():
    """Mock tqdm globally to avoid progress bar output in tests."""
    with patch('micromind.callbacks.tqdm') as mock_tqdm:
        mock_tqdm.return_value.__enter__ = Mock(return_value=Mock())
        mock_tqdm.return_value.__exit__ = Mock(return_value=None)
        mock_tqdm.return_value.set_description = Mock()
        mock_tqdm.return_value.set_postfix = Mock()
        mock_tqdm.return_value.update = Mock()
        mock_tqdm.return_value.close = Mock()
        yield mock_tqdm


@pytest.fixture(autouse=True)
def mock_logger():
    """Mock logger globally to avoid log output in tests."""
    with patch('micromind.core.logger') as mock_logger, \
         patch('micromind.callbacks.logger') as mock_callback_logger:
        yield mock_logger, mock_callback_logger


@pytest.fixture
def sample_batch():
    """Create a sample batch for testing."""
    return [torch.randn(4, 10), torch.randint(0, 2, (4,))]


@pytest.fixture
def sample_prediction():
    """Create a sample prediction for testing."""
    return torch.randn(4, 2)


# Parametrized fixtures for different scenarios
@pytest.fixture(params=['adam', 'sgd'])
def optimizer_type(request):
    """Parametrize optimizer types."""
    return request.param


@pytest.fixture(params=[Stage.train, Stage.val, Stage.test])
def stage(request):
    """Parametrize training stages."""
    return request.param


@pytest.fixture(params=[1, 5, 10])
def num_epochs(request):
    """Parametrize number of epochs."""
    return request.param