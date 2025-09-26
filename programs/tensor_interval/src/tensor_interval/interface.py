from .interval import Interval
from torch import Tensor


def interval_init_inputs(*inputs):
    for input in inputs:
        if isinstance(input, Tensor):
            input.interval = Interval.from_tensor(input)
    if len(inputs) == 1:
        return inputs[0]
    return inputs
