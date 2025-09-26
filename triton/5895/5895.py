import torch
import math
import triton
import triton.language as tl

DEVICE = "cuda"


@triton.jit
def ffn_kernel(
    a_ptr,
    b_ptrt,
    output_ptr,
    d1: tl.constexpr,
    d2: tl.constexpr,
    d3: tl.constexpr,
    bs1: tl.constexpr,
    bs2: tl.constexpr,
    bs3: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # ASM: tl.constexpr = "cvt.rna.tf32.f32 $0, $1;"
    pid_d1 = tl.program_id(0)
    pid_d3 = tl.program_id(1)

    offsets_d1 = tl.arange(0, bs1) + bs1 * pid_d1
    offsets_d3 = tl.arange(0, bs3) + bs3 * pid_d3
    t_val = tl.zeros((bs1, bs3), dtype=tl.float32)
    a_ptrs = a_ptr + (offsets_d1[:, None] * d2 + tl.arange(0, BLOCK_SIZE)[None, :])
    b_ptrs = b_ptrt + (offsets_d3[:, None] * d2 + tl.arange(0, BLOCK_SIZE)[None, :])
    for index in range(0, tl.cdiv(d2, BLOCK_SIZE)):
        mask_a = (offsets_d1[:, None] < d1) & (
            tl.arange(0, BLOCK_SIZE)[None, :] < d2 - index * BLOCK_SIZE
        )
        a_val = tl.load(a_ptrs, mask=mask_a, other=0.0)

        mask_b = (offsets_d3[:, None] < d3) & (
            tl.arange(0, BLOCK_SIZE)[None, :] < d2 - index * BLOCK_SIZE
        )
        b_val = tl.load(b_ptrs, mask=mask_b, other=0.0)
        b_val = b_val.trans(1, 0)

        # a_val = tl.inline_asm_elementwise(ASM, "=r, r", [a_val], dtype=tl.float32, is_pure=True, pack=1)
        # b_val = tl.inline_asm_elementwise(ASM, "=r, r", [b_val], dtype=tl.float32, is_pure=True, pack=1)

        t_val += tl.dot(a_val, b_val)

        a_ptrs += BLOCK_SIZE
        b_ptrs += BLOCK_SIZE

    output_ptrs = output_ptr + (
        offsets_d1[:, None] * d3 + tl.arange(0, bs3)[None, :] + pid_d3 * bs3
    )
    tl.store(
        output_ptrs,
        t_val,
        mask=(offsets_d1[:, None] < d1) & (tl.arange(0, bs3) < d3 - pid_d3 * bs3),
    )


def matrix_multiplication(a, b):
    """
    a: d1 x d2
    b: d3 x d2
    """
    block_size = 16
    bs1 = block_size
    bs2 = block_size
    bs3 = block_size
    bs_hidden = block_size
    (d1, d2) = a.shape
    d3 = b.shape[0]
    output = torch.zeros((d1, d3), dtype=a.dtype, device=a.device)
    grid = lambda META: (triton.cdiv(d1, bs1), triton.cdiv(d3, bs3))
    ffn_kernel[grid](a, b, output, d1, d2, d3, bs1, bs2, bs3, BLOCK_SIZE=bs_hidden)
    return output


def torch_func(a, b):
    output = torch.mm(a, torch.t(b))
    return output


M = 3
N = 2
dim_hidden = 4
torch.manual_seed(91356303)

a = torch.empty((M, N), requires_grad=True, device=DEVICE, dtype=torch.float32)
with torch.no_grad():
    a.normal_(mean=0.0, std=10.0)
b = (
    torch.empty(dim_hidden, N)
    .uniform_(-1.0 * math.sqrt(1.0 / N), math.sqrt(1.0 / N))
    .to(a.device)
    .to(a.dtype)
)
output_torch = torch_func(a, b)
output_triton = matrix_multiplication(a, b)

print(f"a: ", a)
print(f"b: ", b)
print(f"output_torch: ", output_torch)
print(f"output_triton: ", output_triton)

diff = torch.max(torch.abs(output_triton - output_torch)).item()
print(f"The torch difference is {diff}")

with open("reference.pt", "wb") as f:
    output = {
        "torch_output": output_torch.detach(),
        "triton_output": output_triton,
    }
    torch.save(output, f)
