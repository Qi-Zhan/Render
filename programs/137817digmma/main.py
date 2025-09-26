import torch
import numpy as np

a = torch.tensor(0.00615, dtype=torch.float16)

out_cpu = torch.digamma(a)
out_gpu = torch.digamma(a.cuda())

np.testing.assert_allclose(out_cpu.numpy(), out_gpu.cpu().numpy(), atol=0.01)

# AssertionError:
# Not equal to tolerance rtol=1e-07, atol=0.01

# Mismatched elements: 1 / 1 (100%)
# Max absolute difference: 0.125
# Max relative difference: 0.000766
#  x: array(-163.1, dtype=float16)
#  y: array(-163.2, dtype=float16)
