"""
Unit tests for PhiNet architecture.

Run with: pytest test_phinet.py -v
"""
import pytest
import torch
import torch.nn as nn

from micromind.networks.phinet import (
    get_xpansion_factor,
    ReLUMax, SEBlock, DepthwiseConv2d, SeparableConv2d,
    PhiNetConvBlock, PhiNet, PhiNetArchConfig, PhiNetConfig
)

class TestActivationLayers:
    """Test suite for custom activation layers."""
    
    def test_relumax_forward(self):
        """Test ReLUMax forward pass."""
        relu_max = ReLUMax(max=6.0)
        x = torch.tensor([-1.0, 0.0, 3.0, 8.0])
        output = relu_max(x)
        
        assert torch.allclose(output, torch.tensor([0.0, 0.0, 3.0, 6.0]))
    
    def test_relumax_gradient(self):
        """Test ReLUMax gradient computation."""
        relu_max = ReLUMax(max=6.0)
        x = torch.tensor([3.0], requires_grad=True)
        output = relu_max(x)
        output.backward()
        
        assert x.grad is not None


class TestSEBlock:
    """Test suite for Squeeze-and-Excitation block."""
    
    def test_se_block_creation(self):
        """Test SEBlock initialization."""
        se = SEBlock(in_channels=24, out_channels=4, h_swish=True)
        assert isinstance(se, nn.Module)
    
    def test_se_block_forward_shape(self):
        """Test SEBlock output shape."""
        se = SEBlock(in_channels=24, out_channels=4, h_swish=True)
        x = torch.randn(2, 24, 7, 7)
        output = se(x)
        
        assert output.shape == x.shape
    
    def test_se_block_with_relu_max(self):
        """Test SEBlock with ReLUMax activation."""
        se = SEBlock(in_channels=24, out_channels=4, h_swish=False)
        x = torch.randn(2, 24, 7, 7)
        output = se(x)
        
        assert output.shape == x.shape


class TestConvLayers:
    """Test suite for convolution layers."""
    
    def test_depthwise_conv2d_creation(self):
        """Test DepthwiseConv2d initialization."""
        dw_conv = DepthwiseConv2d(in_channels=32, depth_multiplier=1, kernel_size=3)
        assert dw_conv.groups == 32
        assert dw_conv.out_channels == 32
    
    def test_depthwise_conv2d_forward(self):
        """Test DepthwiseConv2d forward pass."""
        dw_conv = DepthwiseConv2d(
            in_channels=32, 
            depth_multiplier=1, 
            kernel_size=3,
            padding=1
        )
        x = torch.randn(2, 32, 14, 14)
        output = dw_conv(x)
        
        assert output.shape == (2, 32, 14, 14)
    
    def test_separable_conv2d_creation(self):
        """Test SeparableConv2d initialization."""
        sep_conv = SeparableConv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=0
        )
        assert isinstance(sep_conv, nn.Module)
    
    def test_separable_conv2d_forward(self):
        """Test SeparableConv2d forward pass."""
        sep_conv = SeparableConv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )
        x = torch.randn(2, 3, 28, 28)
        output = sep_conv(x)
        
        assert output.shape[0] == 2
        assert output.shape[1] == 32


class TestPhiNetConvBlock:
    """Test suite for PhiNet convolutional block."""
    
    def test_conv_block_creation(self):
        """Test PhiNetConvBlock initialization."""
        block = PhiNetConvBlock(
            in_shape=(24, 28, 28),
            expansion=6,
            stride=1,
            filters=24,
            has_se=True,
            block_id=1,
            res=True,
            h_swish=True,
            k_size=3
        )
        assert isinstance(block, nn.Module)
    
    def test_conv_block_forward_same_shape(self):
        """Test PhiNetConvBlock forward with residual connection."""
        block = PhiNetConvBlock(
            in_shape=(24, 28, 28),
            expansion=6,
            stride=1,
            filters=24,
            has_se=True,
            block_id=1,
            res=True,
            h_swish=True,
            k_size=3
        )
        x = torch.randn(2, 24, 28, 28)
        output = block(x)
        
        assert output.shape == x.shape
        assert block.skip_conn is True
    
    def test_conv_block_forward_downsampling(self):
        """Test PhiNetConvBlock forward with stride=2."""
        block = PhiNetConvBlock(
            in_shape=(24, 28, 28),
            expansion=6,
            stride=2,
            filters=48,
            has_se=True,
            block_id=1,
            res=True,
            h_swish=True,
            k_size=3
        )
        x = torch.randn(2, 24, 28, 28)
        output = block(x)
        
        assert output.shape == (2, 48, 14, 14)
        assert block.skip_conn is False
    
    def test_conv_block_without_se(self):
        """Test PhiNetConvBlock without SE block."""
        block = PhiNetConvBlock(
            in_shape=(24, 28, 28),
            expansion=6,
            stride=1,
            filters=24,
            has_se=False,
            block_id=1,
            res=True,
            h_swish=True,
            k_size=3
        )
        x = torch.randn(2, 24, 28, 28)
        output = block(x)
        
        assert output.shape == x.shape


class TestPhiNet:
    """Test suite for PhiNet model."""
    
    def test_phinet_creation_basic(self):
        """Test basic PhiNet initialization."""
        arch = PhiNetArchConfig()
        conf = PhiNetConfig()
        model = PhiNet(
            arch, 
            conf,
            input_shape=[3, 224, 224],
        )
        breakpoint()
        assert isinstance(model, nn.Module)
    
    def test_phinet_forward_no_classifier(self):
        """Test PhiNet forward pass without classifier."""
        model = PhiNet(
            input_shape=[3, 224, 224],
            num_layers=7,
            alpha=0.2,
            beta=1.0,
            include_top=False
        )
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        
        assert output.shape[0] == 2
        assert len(output.shape) == 4  # (B, C, H, W)
    
    def test_phinet_forward_with_classifier(self):
        """Test PhiNet forward pass with classifier."""
        model = PhiNet(
            input_shape=[3, 224, 224],
            num_layers=7,
            alpha=0.2,
            beta=1.0,
            include_top=True,
            num_classes=10
        )
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        
        assert output.shape == (2, 10)
    
    def test_phinet_different_input_sizes(self):
        """Test PhiNet with different input resolutions."""
        for size in [32, 64, 128, 224]:
            model = PhiNet(
                input_shape=[3, size, size],
                num_layers=7,
                alpha=0.2,
                include_top=True,
                num_classes=10
            )
            x = torch.randn(1, 3, size, size)
            output = model(x)
            assert output.shape == (1, 10)
    
    def test_phinet_compatibility_mode(self):
        """Test PhiNet with compatibility mode enabled."""
        model = PhiNet(
            input_shape=[3, 224, 224],
            num_layers=7,
            alpha=0.2,
            compatibility=True,
            include_top=False
        )
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        
        assert output is not None
    
    def test_phinet_custom_downsampling_layers(self):
        """Test PhiNet with custom downsampling layers."""
        model = PhiNet(
            input_shape=[3, 224, 224],
            num_layers=10,
            alpha=0.2,
            downsampling_layers=[3, 6, 9],
            include_top=False
        )
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        
        assert output is not None
    
    def test_phinet_get_stage_names(self):
        """Test getting stage names from PhiNet."""
        model = PhiNet(
            input_shape=[3, 224, 224],
            num_layers=7,
            alpha=0.2,
            include_top=False
        )
        stage_names = model.get_stage_names()
        
        assert isinstance(stage_names, list)
        assert 'stem' in stage_names
        assert len(stage_names) > 0
    
    def test_phinet_different_alphas(self):
        """Test PhiNet with different width multipliers."""
        for alpha in [0.1, 0.2, 0.5, 1.0]:
            model = PhiNet(
                input_shape=[3, 224, 224],
                num_layers=7,
                alpha=alpha,
                include_top=False
            )
            x = torch.randn(1, 3, 224, 224)
            output = model(x)
            assert output is not None
    
    def test_phinet_complexity_metrics(self):
        """Test PhiNet complexity calculation methods."""
        model = PhiNet(
            input_shape=[3, 224, 224],
            num_layers=7,
            alpha=0.2,
            include_top=True,
            num_classes=10
        )
        
        complexity = model.get_complexity()
        assert 'MAC' in complexity
        assert 'params' in complexity
        assert complexity['MAC'] > 0
        assert complexity['params'] > 0
        
        mac = model.get_MAC()
        params = model.get_params()
        assert mac == complexity['MAC']
        assert params == complexity['params']
    
    def test_phinet_conv2d_input(self):
        """Test PhiNet with conv2d_input=True."""
        model = PhiNet(
            input_shape=[3, 224, 224],
            num_layers=7,
            alpha=0.2,
            conv2d_input=True,
            include_top=False
        )
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        
        assert output is not None
    
    def test_phinet_with_pooling(self):
        """Test PhiNet with pooling enabled."""
        model = PhiNet(
            input_shape=[3, 224, 224],
            num_layers=7,
            alpha=0.2,
            pool=True,
            include_top=False
        )
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        
        assert output is not None
    
    def test_phinet_gradient_flow(self):
        """Test gradient flow through PhiNet."""
        model = PhiNet(
            input_shape=[3, 32, 32],
            num_layers=5,
            alpha=0.2,
            include_top=True,
            num_classes=10
        )
        x = torch.randn(2, 3, 32, 32, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        # Check that some model parameters have gradients
        has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_grad


class TestPhiNetEdgeCases:
    """Test suite for edge cases and error handling."""
    
    def test_phinet_single_layer(self):
        """Test PhiNet with minimum number of layers."""
        model = PhiNet(
            input_shape=[3, 224, 224],
            num_layers=4,  # Minimum to test basic functionality
            alpha=0.2,
            include_top=False
        )
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        
        assert output is not None
    
    def test_phinet_batch_size_one(self):
        """Test PhiNet with batch size of 1."""
        model = PhiNet(
            input_shape=[3, 224, 224],
            num_layers=7,
            alpha=0.2,
            include_top=True,
            num_classes=10
        )
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        
        assert output.shape == (1, 10)
    
    def test_phinet_large_batch(self):
        """Test PhiNet with large batch size."""
        model = PhiNet(
            input_shape=[3, 32, 32],
            num_layers=5,
            alpha=0.2,
            include_top=True,
            num_classes=10
        )
        x = torch.randn(32, 3, 32, 32)
        output = model(x)
        
        assert output.shape == (32, 10)
    
    def test_phinet_eval_mode(self):
        """Test PhiNet in evaluation mode."""
        model = PhiNet(
            input_shape=[3, 224, 224],
            num_layers=7,
            alpha=0.2,
            include_top=True,
            num_classes=10
        )
        model.eval()
        
        with torch.no_grad():
            x = torch.randn(2, 3, 224, 224)
            output1 = model(x)
            output2 = model(x)
            
            # In eval mode, same input should give same output
            assert torch.allclose(output1, output2)
    
    def test_phinet_different_channels(self):
        """Test PhiNet with different input channel numbers."""
        for channels in [1, 3, 4]:
            model = PhiNet(
                input_shape=[channels, 64, 64],
                num_layers=5,
                alpha=0.2,
                include_top=False
            )
            x = torch.randn(1, channels, 64, 64)
            output = model(x)
            assert output is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])