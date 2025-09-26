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
        # a:  tensor([[  9.9862,  -4.8059],
        # [ 23.0462,   4.9598],
        # [ 22.2345, -11.8536]], device='cuda:0', requires_grad=True)
        mask_b = (offsets_d3[:, None] < d3) & (
            tl.arange(0, BLOCK_SIZE)[None, :] < d2 - index * BLOCK_SIZE
        )
        b_val = tl.load(b_ptrs, mask=mask_b, other=0.0)
# b:  tensor([[ 0.5711, -0.6026],
#         [ 0.0541,  0.1556],
#         [-0.0702,  0.6694],
#         [-0.5888,  0.0057]], device='cuda:0')
        b_val = b_val.trans(1, 0)
        t_val += tl.dot(a_val, b_val)
# t_val:  tensor([[  8.5991,  -0.2071,  -3.9181,  -5.9075],
#         [ 10.1727,   2.0197,   1.7027, -13.5413],
#         [ 19.8409,  -0.6407,  -9.4958, -13.1599]]

# Interval        [[[  8.57387074,  8.62428865], [ -0.21092314, -0.20336826], [ -3.92963521, -3.90666242], [ -5.92483591, -5.8901991 ]]
#  [[ 10.1254107 , 10.22010172], [  2.01376328,  2.02560504], [  1.68824084,  1.71719161], [-13.58122575, -13.50149663]]
#  [[ 19.78281741, 19.89914856], [ -0.64958951, -0.63171627], [ -9.52364348, -9.46796791], [-13.19847322, -13.12131446]]]

        a_ptrs += BLOCK_SIZE
        b_ptrs += BLOCK_SIZE

    output_ptrs = output_ptr + (
        offsets_d1[:, None] * d3 + tl.arange(0, bs3)[None, :] + pid_d3 * bs3
    )
    interpreter_builder.record(
        output_ptrs.handle,
        t_val.handle,
        mask=(
            (offsets_d1[:, None] < d1) & (tl.arange(0, bs3) < d3 - pid_d3 * bs3)
        ).handle,
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

output_triton = matrix_multiplication(a, b)

print(f"a: ", a)
print(f"b: ", b)
print(f"output_triton: ", output_triton)

interpreter_builder.store_interval()
