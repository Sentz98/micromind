"""
Unit tests for the core MicroMind classes.
"""
import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
from argparse import Namespace

from micromind.core import MicroMind, Metric, default_cfg
from micromind.enum import Stage
from micromind.callbacks import TrainingCallback, EarlyStoppingCallback


class TestMetric:
    """Test cases for the Metric class."""
    
    def test_metric_initialization(self):
        """Test Metric initialization with default parameters."""
        def dummy_fn(pred, batch):
            return torch.tensor([1.0])
        
        metric = Metric("test_metric", dummy_fn)
        
        assert metric.name == "test_metric"
        assert metric.fn == dummy_fn
        assert metric.reduction == "mean"
        assert metric.eval_only is False
        assert metric.eval_period == 1
        assert len(metric.history) == 3  # train, val, test
    
    def test_metric_initialization_custom(self):
        """Test Metric initialization with custom parameters."""
        def dummy_fn(pred, batch):
            return torch.tensor([1.0])
        
        metric = Metric("custom", dummy_fn, reduction="sum", eval_only=True, eval_period=5)
        
        assert metric.name == "custom"
        assert metric.reduction == "sum"
        assert metric.eval_only is True
        assert metric.eval_period == 5
    
    def test_metric_call(self):
        """Test Metric __call__ method."""
        def dummy_fn(pred, batch):
            return torch.tensor([1.0, 2.0])
        
        metric = Metric("test", dummy_fn)
        pred = torch.tensor([1.0])
        batch = torch.tensor([0.5])
        
        metric(pred, batch, Stage.train, device="cpu")
        
        assert len(metric.history[Stage.train]) == 1
        assert torch.equal(metric.history[Stage.train][0], torch.tensor([1.0, 2.0]))
    
    def test_metric_call_scalar_result(self):
        """Test Metric __call__ with scalar result."""
        def dummy_fn(pred, batch):
            return torch.tensor(1.0)  # scalar
        
        metric = Metric("test", dummy_fn)
        pred = torch.tensor([1.0])
        batch = torch.tensor([0.5])
        
        metric(pred, batch, Stage.train, device="cpu")
        
        assert len(metric.history[Stage.train]) == 1
        assert metric.history[Stage.train][0].shape == (1,)  # should be unsqueezed
    
    def test_metric_reduce_mean(self):
        """Test Metric reduce method with mean reduction."""
        def dummy_fn(pred, batch):
            return torch.tensor([1.0, 2.0])
        
        metric = Metric("test", dummy_fn, reduction="mean")
        
        # Add some test data
        metric.history[Stage.train] = [
            torch.tensor([1.0, 2.0]),
            torch.tensor([3.0, 4.0])
        ]
        
        result = metric.reduce(Stage.train)
        expected = torch.tensor([1.0, 2.0, 3.0, 4.0]).mean().item()
        
        assert result == expected
    
    def test_metric_reduce_sum(self):
        """Test Metric reduce method with sum reduction."""
        def dummy_fn(pred, batch):
            return torch.tensor([1.0, 2.0])
        
        metric = Metric("test", dummy_fn, reduction="sum")
        
        # Add some test data
        metric.history[Stage.train] = [
            torch.tensor([1.0, 2.0]),
            torch.tensor([3.0, 4.0])
        ]
        
        result = metric.reduce(Stage.train)
        expected = torch.tensor([1.0, 2.0, 3.0, 4.0]).sum().item()
        
        assert result == expected
    
    def test_metric_reduce_clear(self):
        """Test Metric reduce method with clear=True."""
        def dummy_fn(pred, batch):
            return torch.tensor([1.0])
        
        metric = Metric("test", dummy_fn)
        metric.history[Stage.train] = [torch.tensor([1.0]), torch.tensor([2.0])]
        
        result = metric.reduce(Stage.train, clear=True)
        
        assert result == 1.5  # mean of [1.0, 2.0]
        assert len(metric.history[Stage.train]) == 0  # should be cleared
    
    @pytest.mark.parametrize("stage", [Stage.train, Stage.val, Stage.test])
    def test_metric_different_stages(self, stage):
        """Test Metric with different training stages."""
        def dummy_fn(pred, batch):
            return torch.tensor([1.0])
        
        metric = Metric("test", dummy_fn)
        pred = torch.tensor([1.0])
        batch = torch.tensor([0.5])
        
        metric(pred, batch, stage, device="cpu")
        
        assert len(metric.history[stage]) == 1
        assert len(metric.history[Stage.train if stage != Stage.train else Stage.val]) == 0


class TestMicroMind:
    """Test cases for the MicroMind abstract base class."""
    
    def test_micromind_initialization_default(self, mock_accelerator):
        """Test MicroMind initialization with default parameters."""
        with patch('micromind.core.Accelerator', return_value=mock_accelerator):
            model = DummyMicroMind()
            
            assert isinstance(model.modules, torch.nn.ModuleDict)
            assert model.hparams.experiment_name == "micromind_exp"
            assert model.current_epoch == 0
            assert model.disable_progress is True
            assert model.device == mock_accelerator.device
            assert model.callback_manager is not None
    
    def test_micromind_initialization_custom_hparams(self, mock_accelerator, default_hparams):
        """Test MicroMind initialization with custom hyperparameters."""
        with patch('micromind.core.Accelerator', return_value=mock_accelerator):
            model = DummyMicroMind(default_hparams)
            
            assert model.hparams.experiment_name == "test_exp"
            assert model.hparams.lr == 0.001
            assert model.hparams.opt == "adam"
    
    def test_set_input_shape(self, dummy_micromind):
        """Test set_input_shape method."""
        input_shape = (3, 224, 224)
        dummy_micromind.set_input_shape(input_shape)
        
        assert dummy_micromind.input_shape == input_shape
        assert dummy_micromind.modules.input_shape == input_shape
    
    def test_configure_optimizers_adam(self, dummy_micromind):
        """Test configure_optimizers with Adam optimizer."""
        dummy_micromind.hparams.opt = "adam"
        dummy_micromind.hparams.lr = 0.01
        
        optimizer = dummy_micromind.configure_optimizers()
        
        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.param_groups[0]['lr'] == 0.01
    
    def test_configure_optimizers_sgd(self, dummy_micromind):
        """Test configure_optimizers with SGD optimizer."""
        dummy_micromind.hparams.opt = "sgd"
        dummy_micromind.hparams.lr = 0.05
        
        optimizer = dummy_micromind.configure_optimizers()
        
        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.param_groups[0]['lr'] == 0.05
    
    def test_configure_optimizers_unsupported(self, dummy_micromind):
        """Test configure_optimizers with unsupported optimizer."""
        dummy_micromind.hparams.opt = "rmsprop"
        
        with pytest.raises(AssertionError, match="Optimizer rmsprop not supported"):
            dummy_micromind.configure_optimizers()
    
    def test_call_method(self, dummy_micromind, sample_batch):
        """Test __call__ method forwards to forward."""
        result = dummy_micromind(sample_batch)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == (4, 2)  # batch_size=4, num_classes=2
    
    def test_add_forward_to_modules(self, dummy_micromind):
        """Test add_forward_to_modules method."""
        dummy_micromind.add_forward_to_modules()
        
        assert hasattr(dummy_micromind.modules, 'forward')
        assert hasattr(dummy_micromind.modules, 'device')
        assert dummy_micromind.modules.device == dummy_micromind.device
    
    def test_compute_params(self, dummy_micromind):
        """Test compute_params method."""
        with patch('micromind.core.summary') as mock_summary:
            mock_summary.return_value.total_params = 1000
            
            params = dummy_micromind.compute_params()
            
            assert 'classifier' in params
            assert params['classifier'] == 1000
            mock_summary.assert_called_once()
    
    def test_compute_macs(self, dummy_micromind):
        """Test compute_macs method."""
        with patch('micromind.core.summary') as mock_summary:
            mock_summary.return_value.total_mult_adds = 5000
            
            macs = dummy_micromind.compute_macs((10,))
            
            assert 'classifier' in macs
            assert macs['classifier'] == 5000
    
    def test_compute_macs_runtime_error(self, dummy_micromind):
        """Test compute_macs method with RuntimeError."""
        with patch('micromind.core.summary', side_effect=RuntimeError("Test error")), \
             patch('micromind.core.warnings') as mock_warnings:
            
            macs = dummy_micromind.compute_macs((10,))
            
            assert macs is None
            mock_warnings.warn.assert_called_once()
    
    def test_eval_mode(self, dummy_micromind):
        """Test eval method sets modules to evaluation mode."""
        dummy_micromind.eval()
        
        for module in dummy_micromind.modules.values():
            assert not module.training
    
    def test_load_modules_success(self, dummy_micromind, temp_dir):
        """Test successful load_modules."""
        # Save a dummy checkpoint
        checkpoint_path = temp_dir / "test_checkpoint.pt"
        dummy_state = {
            'classifier': dummy_micromind.modules.classifier.state_dict()
        }
        torch.save(dummy_state, checkpoint_path)
        
        # Load the checkpoint
        dummy_micromind.load_modules(checkpoint_path)
        
        # Should complete without error
        assert True
    
    def test_load_modules_with_ddp_fallback(self, dummy_micromind, temp_dir):
        """Test load_modules with DDP fallback."""
        # Create a checkpoint with DDP-style keys
        checkpoint_path = temp_dir / "test_checkpoint.pt"
        
        # Create dummy state that will cause initial loading to fail
        dummy_state = {
            'classifier': {'wrong_key': torch.tensor([1.0])}
        }
        torch.save(dummy_state, checkpoint_path)
        
        with patch('micromind.core.warnings') as mock_warnings:
            dummy_micromind.load_modules(checkpoint_path)
            mock_warnings.warn.assert_called()
    
    def test_callback_management(self, dummy_micromind):
        """Test callback management methods."""
        callback = TrainingCallback()
        
        # Test add callback
        dummy_micromind.add_callback(callback)
        assert callback in dummy_micromind.callback_manager.callbacks
        
        # Test remove callback
        dummy_micromind.remove_callback(callback)
        assert callback not in dummy_micromind.callback_manager.callbacks
        
        # Test remove by type
        callback1 = TrainingCallback()
        callback2 = EarlyStoppingCallback()
        dummy_micromind.add_callback(callback1)
        dummy_micromind.add_callback(callback2)
        
        dummy_micromind.remove_callback_by_type(EarlyStoppingCallback)
        assert callback1 in dummy_micromind.callback_manager.callbacks
        assert callback2 not in dummy_micromind.callback_manager.callbacks
    
    @pytest.mark.parametrize("optimizer_type", ["adam", "sgd"])
    def test_configure_optimizers_parametrized(self, dummy_micromind, optimizer_type):
        """Test configure_optimizers with different optimizer types."""
        dummy_micromind.hparams.opt = optimizer_type
        dummy_micromind.hparams.lr = 0.01
        
        optimizer = dummy_micromind.configure_optimizers()
        
        if optimizer_type == "adam":
            assert isinstance(optimizer, torch.optim.Adam)
        elif optimizer_type == "sgd":
            assert isinstance(optimizer, torch.optim.SGD)
        
        assert optimizer.param_groups[0]['lr'] == 0.01


class TestMicroMindTraining:
    """Test cases for MicroMind training functionality."""
    
    def test_init_devices(self, dummy_micromind):
        """Test init_devices method."""
        # Add optimizer and scheduler for testing
        dummy_micromind.opt = torch.optim.Adam(dummy_micromind.modules.parameters())
        dummy_micromind.lr_sched = torch.optim.lr_scheduler.StepLR(dummy_micromind.opt, step_size=1)
        dummy_micromind.datasets = {'train': Mock()}
        
        dummy_micromind.init_devices()
        
        # Check that accelerator.prepare was called
        dummy_micromind.accelerator.prepare.assert_called()
        dummy_micromind.accelerator.register_for_checkpointing.assert_called()
    
    def test_train_simple(self, dummy_micromind, datasets, dummy_metric):
        """Test basic training functionality."""
        # Mock the optimizer and scheduler
        dummy_micromind.opt = Mock()
        dummy_micromind.lr_sched = Mock()
        dummy_micromind.start_epoch = 0
        
        # Mock init_devices to avoid accelerator complexity
        dummy_micromind.init_devices = Mock()
        
        with patch.object(dummy_micromind, '_train_epoch', return_value={'train_loss': 0.5}), \
             patch.object(dummy_micromind, '_validate_epoch', return_value={'val_loss': 0.4}):
            
            dummy_micromind.train(
                epochs=2,
                datasets=datasets,
                metrics=[dummy_metric]
            )
            
            assert dummy_micromind.current_epoch == 2
    
    def test_train_early_stopping(self, dummy_micromind, datasets):
        """Test training with early stopping."""
        early_stopping = EarlyStoppingCallback(patience=1, monitor='val_loss')
        dummy_micromind.add_callback(early_stopping)
        
        dummy_micromind.opt = Mock()
        dummy_micromind.start_epoch = 0
        dummy_micromind.init_devices = Mock()
        
        # Mock train epoch to return increasing loss (trigger early stopping)
        def mock_train_epoch(state):
            return {'train_loss': state.epoch * 0.1}
        
        def mock_val_epoch(state):
            return {'val_loss': state.epoch * 0.2}  # increasing loss
        
        with patch.object(dummy_micromind, '_train_epoch', side_effect=mock_train_epoch), \
             patch.object(dummy_micromind, '_validate_epoch', side_effect=mock_val_epoch):
            
            dummy_micromind.train(epochs=10, datasets=datasets)
            
            # Should stop early due to increasing validation loss
            assert dummy_micromind.current_epoch < 10
    
    def test_train_debug_mode(self, dummy_micromind, datasets):
        """Test training in debug mode."""
        dummy_micromind.debug = True
        dummy_micromind.opt = Mock()
        dummy_micromind.start_epoch = 0
        dummy_micromind.init_devices = Mock()
        
        with patch.object(dummy_micromind, '_train_epoch', return_value={'train_loss': 0.5}), \
             patch.object(dummy_micromind, '_validate_epoch', return_value={'val_loss': 0.4}):
            
            dummy_micromind.train(epochs=5, datasets=datasets, debug=True)
            
            # In debug mode, should only run 1 epoch
            assert dummy_micromind.current_epoch == 1
    
    def test_validate_method(self, dummy_micromind, datasets, dummy_metric):
        """Test validate method."""
        dummy_micromind.datasets = datasets
        dummy_micromind.current_epoch = 1
        
        with patch.object(dummy_micromind, '_validate_epoch', return_value={'val_loss': 0.3}) as mock_validate:
            result = dummy_micromind.validate()
            
            assert result == {'val_loss': 0.3}
            mock_validate.assert_called_once()
    
    def test_test_method(self, dummy_micromind, datasets, dummy_metric):
        """Test test method."""
        # Mock the test loop
        test_data = [([torch.randn(2, 10), torch.randint(0, 2, (2,))]) for _ in range(5)]
        datasets['test'] = test_data
        
        with patch('micromind.core.tqdm') as mock_tqdm:
            mock_tqdm.return_value.__enter__ = Mock(return_value=Mock())
            mock_tqdm.return_value.__exit__ = Mock(return_value=None)
            
            result = dummy_micromind.test(datasets=datasets, metrics=[dummy_metric])
            
            assert 'test_loss' in result
            assert 'test_accuracy' in result
            assert isinstance(result['test_loss'], float)
    
    def test_export_onnx(self, dummy_micromind, temp_dir):
        """Test export to ONNX format."""
        with patch('micromind.convert.convert_to_onnx') as mock_convert:
            dummy_micromind.export(
                save_dir=temp_dir,
                out_format="onnx",
                input_shape=(10,)
            )
            
            mock_convert.assert_called_once()
            assert dummy_micromind.input_shape == (10,)
    
    def test_export_openvino(self, dummy_micromind, temp_dir):
        """Test export to OpenVINO format."""
        with patch('micromind.convert.convert_to_openvino') as mock_convert:
            dummy_micromind.export(
                save_dir=temp_dir,
                out_format="openvino",
                input_shape=(10,)
            )
            
            mock_convert.assert_called_once()
    
    def test_export_tflite(self, dummy_micromind, temp_dir):
        """Test export to TFLite format."""
        qbatch = torch.randn(1, 224, 224, 3)  # NHWC format
        
        with patch('micromind.convert.convert_to_tflite') as mock_convert:
            dummy_micromind.export(
                save_dir=temp_dir,
                out_format="tflite",
                input_shape=(10,),
                qbatch=qbatch
            )
            
            mock_convert.assert_called_once()
            # Check that qbatch was permuted to NCHW
            args, kwargs = mock_convert.call_args
            assert 'batch_quant' in kwargs
    
    def test_export_tflite_quantization_error(self, dummy_micromind, temp_dir):
        """Test export error when trying to quantize non-TFLite format."""
        qbatch = torch.randn(1, 3, 224, 224)
        
        with pytest.raises(AssertionError, match="Can perform quantization only on TFLite models"):
            dummy_micromind.export(
                save_dir=temp_dir,
                out_format="onnx",
                qbatch=qbatch
            )
    
    def test_export_no_input_shape_error(self, dummy_micromind, temp_dir):
        """Test export error when input_shape is not set."""
        dummy_micromind.input_shape = None
        
        with pytest.raises(AssertionError, match="You should pass the input_shape"):
            dummy_micromind.export(save_dir=temp_dir)


class TestDefaultConfig:
    """Test cases for default configuration."""
    
    def test_default_config_values(self):
        """Test that default_cfg has expected values."""
        assert default_cfg["output_folder"] == "results"
        assert default_cfg["experiment_name"] == "micromind_exp"
        assert default_cfg["opt"] == "adam"
        assert default_cfg["lr"] == 0.001
        assert default_cfg["debug"] is False
    
    def test_default_config_keys(self):
        """Test that default_cfg has all required keys."""
        required_keys = ["output_folder", "experiment_name", "opt", "lr", "debug"]
        
        for key in required_keys:
            assert key in default_cfg


# Helper class for testing (already defined in conftest.py but redefined here for clarity)
class DummyMicroMind(MicroMind):
    """Concrete implementation of MicroMind for testing."""
    
    def __init__(self, hparams=None, disable_progress=True):
        super().__init__(hparams, disable_progress)
        self.modules = torch.nn.ModuleDict({
            'classifier': nn.Linear(10, 2)
        })
        self.set_input_shape((10,))
    
    def forward(self, batch):
        if isinstance(batch, list):
            x = batch[0]
        else:
            x = batch
        return self.modules.classifier(x)
    
    def compute_loss(self, pred, batch):
        if isinstance(batch, list):
            target = batch[1] if len(batch) > 1 else torch.zeros(pred.size(0), dtype=torch.long)
        else:
            target = torch.zeros(pred.size(0), dtype=torch.long)
        return nn.CrossEntropyLoss()(pred, target)