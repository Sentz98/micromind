"""
Code for PhiNets (https://doi.org/10.1145/3510832).

Authors:
    - Francesco Paissan, 2023
    - Alberto Ancilotto, 2023
    - Matteo Beltrami, 2023
    - Matteo Tremonti, 2023
    - Gabriele Santini, 2025
"""
from dataclasses import dataclass

import torch
import torch.ao.nn.quantized as nnq
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

from ._utils import _make_divisible, correct_pad
from ..utils.helpers import get_logger

logger = get_logger()

__all__ = [
    "PhiNet",
    "PhiNetArchConfig", 
    "PhiNetConfig",
    "SEBlock",
    "PhiNetConvBlock",
]

def get_xpansion_factor(t_zero, beta, block_id, num_blocks):
    """Compute the expansion factor based on the formula from the paper.

    Arguments
    ---------
    t_zero : float
        The base expansion factor.
    beta : float
        The shape factor.
    block_id : int
        The identifier of the current block.
    num_blocks : int
        The total number of blocks.

    Returns
    -------
    float
        The computed expansion factor.
    """
    return (t_zero * beta) * block_id / num_blocks + t_zero * (
        num_blocks - block_id
    ) / num_blocks


class ReLUMax(torch.nn.Module):
    """Implements ReLUMax.

    Arguments
    ---------
    max_value : float
        The maximum value for the clamp operation.

    """

    def __init__(self, max):
        super(ReLUMax, self).__init__()
        self.max = max

    def forward(self, x):
        return torch.clamp(x, min=0, max=self.max)


class SEBlock(torch.nn.Module):
    """Implements squeeze-and-excitation block.

    Arguments
    ---------
    in_channels : int
        Input number of channels.
    out_channels : int
        Output number of channels.
    h_swish : bool, optional
        Whether to use the h_swish (default is True).

    """

    def __init__(self, in_channels, out_channels, h_swish=True):
        super(SEBlock, self).__init__()

        self.se_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            bias=False,
        )

        self.se_conv2 = nn.Conv2d(
            out_channels, in_channels, kernel_size=1, bias=False, padding=0
        )

        if h_swish:
            self.activation = nn.Hardswish(inplace=True)
        else:
            self.activation = ReLUMax(6)

        # It serves for the quantization.
        # The behavior remains equivalent for the unquantized models.
        self.mult = nnq.FloatFunctional()

    def forward(self, x):
        inp = x
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = self.se_conv(x)
        x = self.activation(x)
        x = self.se_conv2(x)
        x = torch.sigmoid(x)

        return self.mult.mul(inp, x)  # Equivalent to ``torch.mul(a, b)``


class DepthwiseConv2d(torch.nn.Conv2d):
    """Depthwise 2D convolution layer.

    Arguments
    ---------
    in_channels : int
        Number of input channels.
    depth_multiplier : int, optional
        The channel multiplier for the output channels (default is 1).
    kernel_size : int or tuple, optional
        Size of the convolution kernel (default is 3).
    stride : int or tuple, optional
        Stride of the convolution (default is 1).
    padding : int or tuple, optional
        Zero-padding added to both sides of the input (default is 0).
    dilation : int or tuple, optional
        Spacing between kernel elements (default is 1).
    bias : bool, optional
        If True, adds a learnable bias to the output (default is False).
    padding_mode : str, optional
        'zeros' or 'circular'. Padding mode for convolution (default is 'zeros').

    """

    def __init__(
        self,
        in_channels,
        depth_multiplier=1,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        padding_mode="zeros",
    ):
        out_channels = in_channels * depth_multiplier
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
            padding_mode=padding_mode,
        )


class SeparableConv2d(torch.nn.Module):
    """Implements SeparableConv2d.

    Arguments
    ---------
    in_channels : int
        Input number of channels.
    out_channels : int
        Output number of channels.
    activation : function, optional
        Activation function to apply (default is torch.nn.functional.relu).
    kernel_size : int, optional
        Kernel size (default is 3).
    stride : int, optional
        Stride for convolution (default is 1).
    padding : int, optional
        Padding for convolution (default is 0).
    dilation : int, optional
        Dilation factor for convolution (default is 1).
    bias : bool, optional
        If True, adds a learnable bias to the output (default is True).
    padding_mode : str, optional
        Padding mode for convolution (default is 'zeros').
    depth_multiplier : int, optional
        Depth multiplier (default is 1).

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        activation: nn.Module | None = None,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        padding_mode="zeros",
        depth_multiplier=1, #TODO remove unused params?
    ):
        super().__init__()

        layers: list[nn.Module] = []

        depthwise = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=stride,
            padding=0,
            dilation=1,
            groups=in_channels,
            bias=bias,
            padding_mode=padding_mode,
        )

        spatialConv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=dilation,
            # groups=in_channels,
            bias=bias,
            padding_mode=padding_mode,
        )

        bn = torch.nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.999) # TODO TensorFlow-style momentum for compatibility (keep?)

        layers.append(depthwise)
        layers.append(spatialConv)
        layers.append(bn)
        if activation is None:
            activation = nn.ReLU(inplace=True)
        layers.append(activation)

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class PhiNetConvBlock(nn.Module):
    """
    Implements PhiNet's convolutional block.

    Structure: Expand -> Depthwise -> SE (optional) -> Project -> Residual

    Arguments
    ---------
    in_shape : tuple
        Input shape of the conv block.
    expansion : float
        Expansion coefficient for this convolutional block.
    stride: int
        Stride for the conv block.
    filters : int
        Output channels of the convolutional block.
    block_id : int
        ID of the convolutional block.
    has_se : bool
        Whether to include use Squeeze and Excite or not.
    res : bool
        Whether to use the residual connection or not.
    h_swish : bool
        Whether to use HSwish or not.
    k_size : int
        Kernel size for the depthwise convolution.

    """

    def __init__(
        self,
        in_shape,
        expansion,
        stride,
        filters,
        has_se,
        block_id=None,
        res=True,
        h_swish=True,
        k_size=3,
        dp_rate=0.05,
        divisor=1,
    ):
        super(PhiNetConvBlock, self).__init__()

        self.param_count = 0

        self.skip_conn = False

        layers: list[nn.Module] = []
        in_channels = in_shape[0]

        # Define activation function
        if h_swish:
            activation = nn.Hardswish(inplace=True)
        else:
            activation = ReLUMax(6)

        if block_id:
            # Expand
            conv1 = nn.Conv2d(
                in_channels,
                _make_divisible(int(expansion * in_channels), divisor=divisor),
                kernel_size=1,
                padding=0,
                bias=False,
            )

            bn1 = nn.BatchNorm2d(
                _make_divisible(int(expansion * in_channels), divisor=divisor),
                eps=1e-3,
                momentum=0.999, # TODO TensorFlow-style momentum for compatibility (keep?)
            )

            layers.append(conv1)
            layers.append(bn1)
            layers.append(activation)

        if stride == 2:
            padding = correct_pad([res, res], 3) # TODO è sbagliato, res ora è un bool

        layers.append(nn.Dropout2d(dp_rate))

        d_mul = 1
        in_channels_dw = (
            _make_divisible(int(expansion * in_channels), divisor=divisor)
            if block_id
            else in_channels
        )
        out_channels_dw = in_channels_dw * d_mul
        dw1 = DepthwiseConv2d(
            in_channels=in_channels_dw,
            depth_multiplier=d_mul,
            kernel_size=k_size,
            stride=stride,
            bias=False,
            padding=k_size // 2 if stride == 1 else (padding[1], padding[3]),
        )

        bn_dw1 = nn.BatchNorm2d(
            out_channels_dw,
            eps=1e-3,
            momentum=0.999, # TODO TensorFlow-style momentum for compatibility (keep?)
        )

        # It is necessary to reinitialize the activation
        # for functions using Module.children() to work properly.
        # Module.children() does not return repeated layers.
        if h_swish:
            activation = nn.Hardswish(inplace=True)
        else:
            activation = ReLUMax(6)

        layers.append(dw1)
        layers.append(bn_dw1)
        layers.append(activation)

        if has_se:
            num_reduced_filters = _make_divisible(
                max(1, int(out_channels_dw / 6)), divisor=divisor
            )
            se_block = SEBlock(out_channels_dw, num_reduced_filters, h_swish=h_swish)
            layers.append(se_block)

        conv2 = nn.Conv2d(
            in_channels=out_channels_dw,
            out_channels=filters,
            kernel_size=1,
            padding=0,
            bias=False,
        )

        bn2 = nn.BatchNorm2d(
            filters,
            eps=1e-3,
            momentum=0.999, # TODO TensorFlow-style momentum for compatibility (keep?)
        )

        layers.append(conv2)
        layers.append(bn2)
        self.block = nn.Sequential(*layers)

        if res and in_channels == filters and stride == 1:
            self.skip_conn = True
            # It serves for the quantization.
            # The behavior remains equivalent for the unquantized models.
            self.op = nnq.FloatFunctional()

    def forward(self, x):
        result = self.block(x)
        if self.skip_conn:
            return self.op.add(x, result)
        return result
    
@dataclass
class PhiNetArchConfig:
    """Architecture-defining parameters (these vary between model variants)."""
    num_layers: int = 7
    alpha: float = 0.2          # Width multiplier
    beta: float = 1.0           # Expansion shape factor
    t_zero: float = 6.0         # Base expansion
    downsampling_layers: list[int] | None = None
    conv5_percent: float = 0.0  # When to use 5x5 kernels
    # Stem configuration
    first_conv_stride: int = 2            # Stem stride
    use_separable_stem: bool = True       # SeparableConv vs regular Conv
    
    def __post_init__(self):
        if self.downsampling_layers is None:
            self.downsampling_layers = [5, 7]
        
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")
        if any(layer > self.num_layers for layer in self.downsampling_layers):
            raise ValueError("downsampling_layers contains indices beyond num_layers")
        
        # if not 0 < self.alpha <= 2:
        #     raise ValueError(f"alpha should be in (0, 2], got {self.alpha}")
        #TODO check 4 beta?
        if not 2 <= self.t_zero <= 8:
            logger.warning("Is recommended to keep t_zero between 2 and 8, " \
            "using by default 6 for networks larger than 5MMACC and 5 for networks smaller than that")

@dataclass  
class PhiNetConfig:
    """Implementation and platform-specific settings."""   
    h_swish = False                         # Use hard-swish vs ReLU6
    squeeze_excite: bool = True             # Use SE blocks            
    residuals: bool = True                # Use residual connections
    divisor: int = 8                      # Channel divisibility
    pool: bool = False
    # Compatibility mode (for embedded platforms)
    compatibility: bool = False           # Disable hard operations

    init_weights: bool = False
    
    def __post_init__(self):
        if self.compatibility:
            self.h_swish = False          # Use hard-swish vs ReLU6
            self.squeeze_excite: bool = True   # Use SE blocks

class PhiNet(nn.Module):   
    def __init__(
        self,
        arch_config: PhiNetArchConfig,
        config: PhiNetConfig,
        # Task-specific
        input_shape: list[int],
        num_classes: int = 1000,
        include_top: bool = True,
        # Training hyperparams 
        dropout_rate: float = 0.05, # TODO implement, now is an hyperparameter
        stochastic_depth_prob: float = 0.0, # TODO implement sdp? usually good for training residual nets
    ) -> None:
        super().__init__()
        
        self.alpha = arch_config.alpha
        self.beta = arch_config.beta
        self.t_zero = arch_config.t_zero
        self.num_layers = arch_config.num_layers
        self.num_classes = num_classes
        self.include_top = include_top
        self.input_shape = input_shape

        # Base filters configuration
        first_conv_filters = 48
        b1_filters = 24
        b2_filters = 48
        
        assert len(input_shape) == 3, "Expected 3 elements list as input_shape."
        in_channels = int(input_shape[0])
        H = int(input_shape[1])
        W = int(input_shape[2])

        activation = nn.Hardswish(inplace=True) if config.h_swish else ReLUMax(6)

        # parse first_conv_stride: can be int or (h,w) tuple. We only use tuple
        # for the stem convolution; internal block strides remain integer 1 or 2
        if isinstance(arch_config.first_conv_stride, tuple) or isinstance(arch_config.first_conv_stride, list):
            first_s_h, first_s_w = int(arch_config.first_conv_stride[0]), int(arch_config.first_conv_stride[1])
        else:
            first_s_h = first_s_w = int(arch_config.first_conv_stride)
        
        # -------------------
        # Stem stage (configurable, separable vs regular conv)
        # -------------------
        if arch_config.use_separable_stem:
            stem_layers = []

            # compute correct pad for (H, W) and kernel_size 3
            pad = nn.ZeroPad2d(padding=correct_pad((H, W), 3))
            stem_layers.append(pad)

            stem_out_channels = _make_divisible(int(first_conv_filters * self.alpha), divisor=config.divisor)

            # SeparableConv2d supports stride as a tuple (we implemented that earlier)
            sep1 = SeparableConv2d(
                in_channels,
                stem_out_channels,
                kernel_size=3,
                stride=(first_s_h, first_s_w),
                padding=0,
                bias=False,
                activation=activation,
            )
            stem_layers.append(sep1)

            # block1 input spatial size computed from per-axis stem stride
            block1_h = max(1, H // first_s_h)
            block1_w = max(1, W // first_s_w)

            block1 = PhiNetConvBlock(
                in_shape=(
                    stem_out_channels,
                    block1_h,
                    block1_w,
                ),
                filters=_make_divisible(int(b1_filters * self.alpha), divisor=config.divisor),
                stride=1,
                expansion=1,
                has_se=False,
                res=config.residuals,
                h_swish=config.h_swish,
                divisor=config.divisor,
            )
            stem_layers.append(block1)

            # Track spatial dimensions after stem
            current_h = block1_h
            current_w = block1_w
        else:
            # Simple conv stem
            stem_out = _make_divisible(int(b1_filters * self.alpha), divisor=config.divisor)
            stem_layers.extend([
                nn.Conv2d(in_channels, stem_out, kernel_size=3, stride=first_s_h, padding=1, bias=False),
                nn.BatchNorm2d(stem_out),
                activation,
            ])
            
            # Track spatial dimensions after stem
            current_h = max(1, H // first_s_h)
            current_w = max(1, W // first_s_w)
        
        self.stem = nn.Sequential(*stem_layers)

        # -------------------
        # Build Feature Layers
        # -------------------
        feature_layers = []
        
        # Current channel count
        current_channels = _make_divisible(int(b1_filters * self.alpha), divisor=config.divisor)

        # Block 2 (first block in features)
        block2 = PhiNetConvBlock(
            in_shape=(current_channels, current_h, current_w),
            filters=_make_divisible(int(b1_filters * self.alpha), divisor=config.divisor),
            stride=2 if not config.pool else 1,
            expansion=get_xpansion_factor(self.t_zero, self.beta, 1, self.num_layers),
            block_id=1,
            has_se=config.squeeze_excite,
            res=config.residuals,
            h_swish=config.h_swish,
            divisor=config.divisor,
        )
        feature_layers.append(block2)
        
        if config.pool:
            feature_layers.append(nn.MaxPool2d((2, 2)))
        
        # Update spatial dimensions after first downsample
        current_h = max(1, current_h // 2)
        current_w = max(1, current_w // 2)

        # Block 3
        block3 = PhiNetConvBlock(
            in_shape=(current_channels, current_h, current_w),
            filters=_make_divisible(int(b1_filters * self.alpha), divisor=config.divisor),
            stride=1,
            expansion=get_xpansion_factor(self.t_zero, self.beta, 2, self.num_layers),
            block_id=2,
            has_se=config.squeeze_excite,
            res=config.residuals,
            h_swish=config.h_swish,
            divisor=config.divisor,
        )
        feature_layers.append(block3)

        # Block 4 (transition to b2_filters)
        block4 = PhiNetConvBlock(
            in_shape=(current_channels, current_h, current_w),
            filters=_make_divisible(int(b2_filters * self.alpha), divisor=config.divisor),
            stride=2 if not config.pool else 1,
            expansion=get_xpansion_factor(self.t_zero, self.beta, 3, self.num_layers),
            block_id=3,
            has_se=config.squeeze_excite,
            res=config.residuals,
            h_swish=config.h_swish,
            divisor=config.divisor,
        )
        feature_layers.append(block4)
        
        if config.pool:
            feature_layers.append(nn.MaxPool2d((2, 2)))
        
        # Update spatial dimensions and channels
        current_h = max(1, current_h // 2)
        current_w = max(1, current_w // 2)
        current_channels = _make_divisible(int(b2_filters * self.alpha), divisor=config.divisor)

        # -------------------
        # Dynamic blocks (block_id 4 to num_layers)
        # -------------------
        block_id = 4
        block_filters = b2_filters

        while block_id <= self.num_layers:
            # Double filters at downsampling layers
            if block_id in arch_config.downsampling_layers:
                block_filters *= 2
                if config.pool:
                    feature_layers.append(nn.MaxPool2d((2, 2)))

            # Determine stride for this block
            block_stride = 2 if (block_id in arch_config.downsampling_layers and not config.pool) else 1

            # Determine kernel size based on position in network
            k_size = 5 if (block_id / float(self.num_layers)) > (1.0 - float(arch_config.conv5_percent)) else 3

            # Create block
            pn_block = PhiNetConvBlock(
                in_shape=(current_channels, current_h, current_w),
                filters=_make_divisible(int(block_filters * self.alpha), divisor=config.divisor),
                stride=block_stride,
                expansion=get_xpansion_factor(self.t_zero, self.beta, block_id, self.num_layers),
                block_id=block_id,
                has_se=config.squeeze_excite,
                res=config.residuals,
                h_swish=config.h_swish,
                k_size=k_size,
                divisor=config.divisor,
            )
            feature_layers.append(pn_block)

            # Update state for next block
            current_channels = _make_divisible(int(block_filters * self.alpha), divisor=config.divisor)
            if block_stride == 2:
                current_h = max(1, current_h // 2)
                current_w = max(1, current_w // 2)

            block_id += 1

        self.features = nn.Sequential(*feature_layers)

        # -------------------
        # Classification Head
        # -------------------

        if include_top:
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(
                    _make_divisible(int(block_filters * self.alpha), divisor=config.divisor),
                    num_classes,
                    bias=True,
                ),
            )
        else:
            self.classifier = nn.Identity()
        

        
        if config.init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass for clean export."""
        x = self.stem(x)
        x = self.features(x)
        return self.classifier(x)
    
    def _initialize_weights(self) -> None:
        """Initialize weights following best practices."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def get_complexity(self):
        """Returns MAC and number of parameters of initialized architecture."""
        temp = summary(
            self, input_data=torch.zeros([1] + list(self.input_shape)), verbose=0
        )
        return {"MAC": temp.total_mult_adds, "params": temp.total_params}

    @torch.jit.ignore
    def get_MAC(self):
        """Returns number of MACs for this architecture."""
        return self.get_complexity()["MAC"]

    @torch.jit.ignore
    def get_params(self):
        """Returns number of params for this architecture."""
        return self.get_complexity()["params"]
    