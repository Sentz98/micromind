from typing import Optional

def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo. It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py

    It ensures that all layers have a channel number that is divisible by divisor.

    Arguments
    ---------
    v : int
        The original number of channels.
    divisor : int, optional
        The divisor to ensure divisibility (default is 8).
    min_value : int or None, optional
        The minimum value for the divisible channels (default is None).

    Returns
    -------
    int
        The adjusted number of channels.
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def correct_pad(input_shape, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.

    Arguments
    ---------
    input_shape : tuple or list
        Shape of the input tensor (height, width).
    kernel_size : int or tuple
        Size of the convolution kernel.

    Returns
    -------
    tuple
        A tuple representing the zero-padding in the format (left, right, top, bottom).
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_shape[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_shape[0] % 2, 1 - input_shape[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return (
        int(correct[1] - adjust[1]),
        int(correct[1]),
        int(correct[0] - adjust[0]),
        int(correct[0]),
    )