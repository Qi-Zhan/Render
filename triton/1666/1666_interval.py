import torch

import triton
import triton.language as tl
import numpy as np
torch.set_printoptions(precision=10)

"""
Modification of the tutorial https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html

except instead of calculating A @ B = C, calculate A.T @ B = C

That is instead of shapes:
   A: [M, K]
   B: [K, N]
   C: [M, N]
   
We
   A: [K, M]
   B: [K, N]
   C: [M, N]
   
"""


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=5,
            num_warps=2,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_M] pointers # NOTE: SWAPPED DIMS
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetics` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_k[:, None] * stride_ak + offs_am[None, :] * stride_am
    )  # NOTE: SWAPPED DIMS
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(
            a_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0
        )  # NOTE: SWAPPED DIMS
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a.T, b)  # NOTE: SWAPPED DIMS
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    interpreter_builder.record(c_ptrs.handle, c.handle, mask=c_mask.handle)
    tl.store(c_ptrs, c, mask=c_mask)


# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `_matmul`.
@triton.jit
def leaky_relu(x):
    x = x + 1
    return tl.where(x >= 0, x, 0.01 * x)


def matmul(a, b, activation=""):
    # Check constraints.
    assert (
        a.shape[0] == b.shape[0]
    ), "Incompatible dimensions"  # NOTE: updated shape stuff
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"

    # NOTE: updated the shape stuff
    K, M = a.shape
    _, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    matmul_kernel[grid](
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
        ACTIVATION=activation,
    )
    return c


if __name__ == "__main__":
    torch.manual_seed(0)

    a = torch.randn((1, 512), device="cuda", dtype=torch.float16)
    b = torch.randn((1, 512), device="cuda", dtype=torch.float16)
    torch_output = torch.matmul(a.T, b)
    from interval import IntervalArray


    a_interval = IntervalArray.from_float(a.cpu().numpy().T, np.finfo(np.float32).eps)
    b_interval = IntervalArray.from_float(b.cpu().numpy(), np.finfo(np.float32).eps)

    d_interval = a_interval.matmul(b_interval, 0, float(np.finfo(np.float16).eps), np.finfo(np.float32).eps, np.array([0]))
    d_interval = d_interval.cast_eps(np.finfo(np.float16).eps)
    print(d_interval)
    # eps = float(np.finfo(np.float16).eps)
    # left = [a.cast_eps(eps).left for a in d_interval.ravel()]
    # right = [a.cast_eps(eps).right for a in d_interval.ravel()]
    # left = np.array(left).reshape(torch_output.shape)
    # right = np.array(right).reshape(torch_output.shape)
    left, right = d_interval.lo, d_interval.hi
    index = 44182
    row = index // 512
    col = index % 512
    print(row, col)
    print(a.flatten()[row], b.flatten()[col])
    print(left.flatten()[index], torch_output.flatten()[index], right.flatten()[index])
    # with open("interval_RQ3.npy", "wb") as f:
    with open("interval.npy", "wb") as f:
        np.save(f, left)
        np.save(f, right)
