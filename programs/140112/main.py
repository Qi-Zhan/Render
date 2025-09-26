import torch
import torch.nn as nn

torch.set_printoptions(precision=10, sci_mode=False)

input = torch.tensor([[[1.3047, 0.8789]]], dtype=torch.bfloat16)
print("input", input)

module = nn.BatchNorm1d(1, affine=False, track_running_stats=False)
module = module.to(torch.bfloat16)

cpu_res = module(input)
print("cpu", cpu_res)

input = input.to("cuda")
module = module.to("cuda")
gpu_res = module(input)
print("gpu", gpu_res.to("cpu"))

model = torch.compile(module)
gpu_res = model(input)
print("gpu compiled", gpu_res.to("cpu"))
