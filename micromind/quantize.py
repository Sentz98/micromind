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
import numpy as np 
import time
from tqdm import tqdm

from .utils.helpers import get_logger
import micromind as mm

logger = get_logger()

def find_layers_in_order(model, layer_types):
    layer_names = []
    modules = list(model.named_modules())

    for i in range(len(modules)):
        block = []
        name, module = modules[i]

        if type(module) == layer_types[0]:
            block.append(name)

            for j in range(1, len(layer_types)):
                if i + 1 < len(modules):
                    name, module = modules[i + j]
                    if type(module) == layer_types[j]:
                        block.append(name)

        if len(block) == len(layer_types):
            layer_names.append(block)

    return layer_names


def fuse_modules(model, layers_types, is_qat=False, inplace=False):
    layer2fuse = find_layers_in_order(model, layers_types)

    if not is_qat:
        model.eval()

    fuse_func = tq.fuse_modules_qat if is_qat else tq.fuse_modules
    model_fused = fuse_func(model, layer2fuse, inplace=False)

    if not inplace:
        return model_fused


def inject_quant(model):
    def forward_injected(self, x):
        x = self.quant(x)
        x = original_forward(self, x)
        x = self.dequant(x)
        return x

    # add quant and dequant layers
    model.quant = tq.QuantStub()
    model.dequant = tq.DeQuantStub()

    # Bind the new method to the instance
    original_forward = model.__class__.forward
    bound_forward = forward_injected.__get__(model, model.__class__)
    setattr(model.__class__, "forward", bound_forward)

def print_size_of_model(model):
    """
    Print the size of the model.
    """
    torch.save(model.state_dict(), "temp.p")
    size_in_bytes = os.path.getsize("temp.p")
    if size_in_bytes < 1048576:
        size_in_kb = size_in_bytes / 1024
        print("{:.3f} KB".format(size_in_kb))
    else:
        size_in_mb = size_in_bytes / 1048576
        print("{:.3f} MB".format(size_in_mb))
    os.remove('temp.p')

def measure_inference_latency(model, input_shape, device = None, repetitions=100, warmup_it = 10):
    """
    Measures the inference time of the provided neural mindwork model.

    Args:
        model: The neural mindwork model to evaluate.
        input_shape: The shape of the input data expected by the model.

    Returns:
        tuple: A tuple containing the mean and standard deviation of the inference time
               measured in milliseconds.
    """
    if device is None:
        device = next(model.parameters()).device.type  # Get the device where the model is located
    
    if len(input_shape) == 4:
        # Remove the batch
        input_shape = input_shape[1:]
    
    dummy_input = torch.randn(1, *input_shape, dtype=torch.float).to(device)
    
    # Set model to evaluation mode
    model.to(device)
    model.eval()
    
    # GPU warm-up
    for _ in range(warmup_it):
        _ = model(dummy_input)

    # Measure inference time
    timings = []
    with torch.no_grad():
        if device == 'cuda':
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            for rep in range(repetitions):
                starter.record()
                _ = model(dummy_input)
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings.append(curr_time)
        else:  # CPU
            for rep in range(repetitions):
                start_time = time.time()
                _ = model(dummy_input)
                end_time = time.time()
                elapsed_time = (end_time - start_time) * 1000.0  # Convert to milliseconds
                timings.append(elapsed_time)

    # Calculate mean and std
    mean_time = np.mean(timings)
    std_time = np.std(timings)

    return mean_time, std_time

def model_equivalence(model_1, model_2, device, rtol=1e-05, atol=1e-08, num_tests=100, input_size=(1,3,32,32), verbose=False):
    """
    Tests whether two models are equivalent by comparing their outputs on random inputs.
    """

    model_1.to(device)
    model_2.to(device)

    for i in range(num_tests):
        print(f"Running test {i+1}/{num_tests}") if verbose else None
        x = torch.rand(size=input_size).to(device)
        y1 = model_1(x).detach().cpu().numpy()
        y2 = model_2(x).detach().cpu().numpy()
        print("Difference: ", np.max(np.abs(y1-y2))) if verbose else None
        if np.allclose(a=y1, b=y2, rtol=rtol, atol=atol, equal_nan=False) == False:
            print("Model equivalence test sample failed: ")
            print(y1)
            print(y2)
            return False
    print("Model equivalence test passed!")
    return True

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
    assert model_equivalence(
        model_1=model_fp32,
        model_2=mind.modules[module], 
        device="cpu", 
        rtol=1e-03, 
        atol=1e-06, 
        num_tests=10, 
        input_size=(1,3,224,224)), "Fused model is not equivalent to the original model!"
    

    # add quant and dequant layers
    inject_quant(mind.modules[module])

    mind.device = "cpu" # quant operations run only on cpu
    mind.modules = mind.modules.to(mind.device)

    # #insert the observer in the model
    mind.modules.qconfig = tq.get_default_qconfig("x86")
    tq.prepare(mind.modules, inplace=True)

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
    print_size_of_model(model_fp32)

    print("INT8 size:")
    print_size_of_model(mind.modules[module])

    fp32_cpu_inference_latency  = measure_inference_latency(model_fp32, device ="cpu", input_shape=(256,3,32,32))
    int8_cpu_inference_latency  = measure_inference_latency(mind.modules[module], device ="cpu", input_shape=(256,3,32,32))

    print("FP32 CPU Inference Latency: {:.3f} ms".format(fp32_cpu_inference_latency[0]))
    print("INT8 CPU Inference Latency: {:.3f} ms".format(int8_cpu_inference_latency[0]))
