import os
import torch
import torch.ao.quantization as tq
import time 
import numpy as np


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
