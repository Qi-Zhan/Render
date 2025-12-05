import torch
import triton
import triton.language as tl
import numpy as np


@triton.jit
def _kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    ACC_TYPE: tl.constexpr,
):
    # matrix multiplication
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K * SPLIT_K):
        a = tl.load(A, mask=rk[None, :] < k, other=0.0)
        b = tl.load(B, mask=rk[:, None] < k, other=0.0)
        acc += tl.dot(a, b)
        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk
    acc = acc.to(C.dtype.element_ty)
    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # handles write-back with reduction-splitting
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)


class _matmul(torch.autograd.Function):
    kernel = _kernel

    _locks = {}

    @staticmethod
    def _call(a, b):
        device = a.device
        # handle non-contiguous inputs if necessary
        if a.stride(0) > 1 and a.stride(1) > 1:
            a = a.contiguous()
        if b.stride(0) > 1 and b.stride(1) > 1:
            b = b.contiguous()
        # checks constraints
        assert a.shape[1] == b.shape[0], "incompatible dimensions"
        M, K = a.shape
        _, N = b.shape
        # allocates output
        c = torch.empty((M, N), device=device, dtype=a.dtype)
        # accumulator types
        ACC_TYPE = (
            tl.float32
            if a.dtype in [torch.float16, torch.bfloat16, torch.float32]
            else tl.int32
        )
        # launch kernel
        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
            META["SPLIT_K"],
        )
        _kernel[grid](
            a,
            b,
            c,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            GROUP_M=8,
            ACC_TYPE=ACC_TYPE,
            BLOCK_M=128,
            BLOCK_N=128,
            BLOCK_K=32,
            # SPLIT_K=4,
            SPLIT_K=1,
        )
        return c

    @staticmethod
    def forward(ctx, a, b):
        return _matmul._call(a, b)


matmul = _matmul.apply

M, N, K = 2048, 2048, 2048
# torch.manual_seed(0)
# a = torch.randn((M, K), device="cuda", dtype=torch.float16)
# b = torch.randn((K, N), device="cuda", dtype=torch.float16)
inputs = torch.load("inputs.pt", weights_only=True)
a = inputs["a"]
b = inputs["b"]
from interval import _matmul

# as the code use atomic add, we use the most conservative estimate
mul_eps = float(np.finfo(np.float16).eps)
add_eps = float(np.finfo(np.float32).eps)
a_low = a_high = a.cpu().double().numpy()
b_low = b_high = b.cpu().double().numpy()
c_low, c_high = _matmul(a_low, a_high, b_low, b_high, mul_eps, add_eps)
print(c_low, c_high)

with open("interval.npy", "wb") as f:
    np.save(f, c_low)
    np.save(f, c_high)
