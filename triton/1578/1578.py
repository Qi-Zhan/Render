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



torch.manual_seed(0)

x = torch.randn((100, 100), dtype=torch.float32, device="cuda")
y = torch.randn((100, 100), dtype=torch.float32, device="cuda")
n_size = x.numel()

ref = x * y + x

out2 = torch.zeros_like(x)

grid = lambda meta: (triton.cdiv(n_size, meta["BLOCK_SIZE"]),)


test_kernel[grid](x, y, out2, n_size, STORE_FLAG=False, BLOCK_SIZE=1024)
diff = (ref - out2).abs().max().item()
allclose = torch.allclose(ref, out2)
print("diff:", diff, "allclose:", allclose)
with open("reference.pt", "wb") as f:
    output = {
        "torch_output": ref,
        "triton_output": out2,
    }
    torch.save(output, f)
