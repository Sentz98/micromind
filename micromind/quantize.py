"""
Post training pytorch eager mode quantization pipeline for micromind model.

Authors:
    - Gabriele Santini, 2024
"""
import os
import copy
from pathlib import Path
from typing import Union

import torch
import torch.ao.quantization as tq
import torch.nn as nn
from tqdm import tqdm

from .utils.helpers import get_logger
import micromind as mm
from micromind.utils import quantizer as qutil

logger = get_logger()

from micromind.networks.phinet import PhiNetConvBlock, SeparableConv2d, DepthwiseConv2d

def phinet_fuse_modules(model):
    for basic_block_name, basic_block in model._layers.named_children():
        if isinstance(basic_block, SeparableConv2d):
            torch.ao.quantization.fuse_modules(basic_block, [["_layers.1", "_layers.2"]], inplace=True)
        if isinstance(basic_block, PhiNetConvBlock) and len(basic_block._layers) == 6:
            torch.ao.quantization.fuse_modules(basic_block, [["_layers.1", "_layers.2"], ["_layers.4", "_layers.5"]], inplace=True)
        elif isinstance(basic_block, PhiNetConvBlock):
            torch.ao.quantization.fuse_modules(basic_block, [["_layers.0", "_layers.1"], ["_layers.4", "_layers.5"]], inplace=True)

def remove_depthwise(model):
    def convert_to_conv2d(depthwise_conv2d):
        in_channels = depthwise_conv2d.in_channels
        depth_multiplier = depthwise_conv2d.out_channels // in_channels
        kernel_size = depthwise_conv2d.kernel_size
        stride = depthwise_conv2d.stride
        padding = depthwise_conv2d.padding
        dilation = depthwise_conv2d.dilation
        bias = depthwise_conv2d.bias is not None
        padding_mode = depthwise_conv2d.padding_mode

        # Create an equivalent nn.Conv2d layer
        conv2d_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels * depth_multiplier,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,  # Set groups to in_channels for depthwise convolution
            bias=bias,
            padding_mode=padding_mode,
        )

        # If bias was not used in the original depthwise_conv2d, set bias to None in conv2d_layer
        if not bias:
            conv2d_layer.bias = None
        
        # Transfer the weights and bias (if applicable)
        with torch.no_grad():
            conv2d_layer.weight.copy_(depthwise_conv2d.weight)
            if bias:
                conv2d_layer.bias.copy_(depthwise_conv2d.bias)

        return conv2d_layer

    for name, module in model._layers.named_children():
        if isinstance(module, PhiNetConvBlock):
            for i, layer in enumerate(module._layers.children()):
                if isinstance(layer, DepthwiseConv2d):
                    module._layers[i] = convert_to_conv2d(layer)

def replace_silu_with_relu(model):
        # Recursively replace all instances of nn.SiLU with nn.ReLU
        for name, module in model.named_children():
            # If the module itself contains children, apply the function recursively
            replace_silu_with_relu(module)
            
            # If the module is SiLU, replace it with ReLU
            if isinstance(module, nn.SiLU):
                setattr(model, name, nn.ReLU())
                

def get_input_shape(dataloader):
    for inputs, _ in dataloader:
        if isinstance(inputs, tuple):
            inputs = inputs[0]  
        return tuple(inputs.shape)

@torch.no_grad()
def quantize_pt(
    mind: Union[nn.Module, mm.MicroMind],
    module: str,
    save_path: Union[Path, str],
    calibration_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    metrics: list,
    max_cal = 20,
) -> None:
    """ Post Training Quantization (PTQ) pipeline.

    Arguments
    ---------
    mind : nn.Module
        PyTorch module to be quantized.
    save_path : Union[Path, str]
        Output path for the quantized model.
    calibration : torch.utils.data.DataLoader
        Calibration dataloader used for quantization.
    test_loader : torch.utils.data.DataLoader
        Test dataloader used for evaluation. 
    """
    model_fp32 = copy.deepcopy(mind.modules[module])
    input_shape = get_input_shape(calibration_loader)

    # change depthwise
    remove_depthwise(mind.modules[module])

    mind.modules.to('cpu')
    mind.modules.eval()

    # fuse modules
    phinet_fuse_modules(mind.modules[module])

    #check model prepared
    assert qutil.model_equivalence(
        model_1=model_fp32,
        model_2=mind.modules[module], 
        device="cpu", 
        rtol=1e-03, 
        atol=1e-06, 
        num_tests=10, 
        input_size=input_shape), "Fused model is not equivalent to the original model!"
    

    # add quant and dequant layers
    qutil.inject_quant(mind.modules[module])

    mind.device = "cpu" # quant operations run only on cpu
    mind.modules = mind.modules.to(mind.device)

    # #insert the observer in the model
    qconf = tq.get_default_qconfig("qnnpack")
    mind.modules.qconfig = qconf

    tq.prepare(mind.modules, inplace=True)

    print(mind.modules)

    # calibrate
    logger.info("Calibrating the quantizer")
    with torch.no_grad(): 
        for i,inputs in tqdm(enumerate(calibration_loader), "Calibration", total=max_cal):
            _ = mind(inputs) 
            if i >= max_cal:  
                break
              
    # quantize the model
    tq.convert(mind.modules, inplace=True)


    # # Save the quantized model
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(mind.modules.state_dict(), save_path.joinpath("model_int8.pt"))

    logger.info(f"Saved quantized model to {save_path}.")


    mind.test(datasets={"test": test_loader}, metrics=metrics)

    print("FP32 size:")
    qutil.print_size_of_model(model_fp32)

    print("INT8 size:")
    qutil.print_size_of_model(mind.modules[module])

    fp32_cpu_inference_latency  = qutil.measure_inference_latency(model_fp32, device ="cpu", input_shape=(256,3,32,32))
    int8_cpu_inference_latency  = qutil.measure_inference_latency(mind.modules[module], device ="cpu", input_shape=(256,3,32,32))

    print("FP32 CPU Inference Latency: {:.3f} ms".format(fp32_cpu_inference_latency[0]))
    print("INT8 CPU Inference Latency: {:.3f} ms".format(int8_cpu_inference_latency[0]))
