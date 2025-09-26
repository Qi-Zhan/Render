import triton
import triton.language as tl
import torch
import numpy as np


@triton.jit
def test_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_size,
    STORE_FLAG: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offset = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_size
    x = tl.load(x_ptr + offset, mask=mask)
    y = tl.load(y_ptr + offset, mask=mask)
    out = x * y
    if STORE_FLAG:
        tl.store(out_ptr + offset, out, mask=mask)
        out = tl.load(out_ptr + offset, mask=mask)
    out = out + x

    tl.store(out_ptr + offset, out, mask=mask)


from interval import IntervalArray

torch.manual_seed(0)

x = torch.randn((100, 100), dtype=torch.float32, device="cuda")
y = torch.randn((100, 100), dtype=torch.float32, device="cuda")
n_size = x.numel()

ref = x * y + x

x_interval = IntervalArray.from_float(x.cpu().numpy(), np.finfo(np.float32).eps)
y_interval = IntervalArray.from_float(y.cpu().numpy(), np.finfo(np.float32).eps)
out_interval = x_interval * y_interval + x_interval

left, right = out_interval.lo, out_interval.hi
print(left, right)
# with open("interval_RQ3.npy", "wb") as f:
with open("interval.npy", "wb") as f:
    np.save(f, left)
    np.save(f, right)
