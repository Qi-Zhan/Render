import torch
import numpy as np


def test_in(a, left, right):
    a = a.cpu().numpy().reshape(left.shape)
    if np.all(a >= left) and np.all(a <= right):
        return True
    # calculate the number of elements that are out of range
    count = np.sum((a < left) | (a > right))
    all_count = np.prod(a.shape)
    print(f"out of range elements: {count}/{all_count}")
    i = 0
    # for x, y, z in zip(a.flatten(), left.flatten(), right.flatten()):
    #     if x < y or x > z:
    #         print(f"out of range: {x} not in [{y}, {z} in index {i}")
    #     i += 1
    return False


# reference.pt  interval.npy
triton_reference = torch.load("reference.pt", weights_only=True)
reference = triton_reference["torch_output"]
triton_output = triton_reference["triton_output"]

with open("interval.npy", "rb") as f:
    left = np.load(f, allow_pickle=True)
    right = np.load(f, allow_pickle=True)
print("reference")
print(reference, reference.shape)
print("triton_output")
print(triton_output, triton_output.shape)
print("interval")
print(left, left.shape)
print(right, right.shape)
if not test_in(triton_output, left, right):
    print("triton_output failed")
else:
    print("triton_output in range")

if not test_in(reference, left, right):
    print("reference failed")
else:
    print("reference in range")
