# -*- coding: utf-8 -*-

# code adapted from
# https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html

from typing import Optional

import torch
import triton
import triton.language as tl

import contextlib
import functools
from typing import Any, Callable, Dict, Literal, Optional, Tuple


device = "cuda"
autocast_custom_fwd = functools.partial(torch.amp.custom_fwd, device_type=device)
autocast_custom_bwd = functools.partial(torch.amp.custom_bwd, device_type=device)


def custom_device_ctx(type: str, index: int):
    return torch.device(type, index)


def input_guard(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """
    A decorator to make sure all input tensors are contiguous and set the device based on input tensors.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        contiguous_args = (
            i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args
        )
        contiguous_kwargs = {
            k: (v if not isinstance(v, torch.Tensor) else v.contiguous())
            for k, v in kwargs.items()
        }

        tensor = None
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensor = arg
                break
        if tensor is None:
            for value in kwargs.values():
                if isinstance(value, torch.Tensor):
                    tensor = value
                    break

        if tensor is not None:
            ctx = custom_device_ctx(tensor.device.type, tensor.device.index)
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            return fn(*contiguous_args, **contiguous_kwargs)

    return wrapper


contiguous = input_guard


@triton.autotune(
    configs=[
        triton.Config(
            {"BM": 32, "BK": 32, "BN": 64, "G": 4}, num_stages=0, num_warps=2
        ),
    ],
    key=["M", "N", "K"],
)
@triton.heuristics(
    {
        "HAS_INPUT": lambda args: args["input"] is not None,
        "HAS_ALPHA": lambda args: args["alpha"] is not None,
        "HAS_BETA": lambda args: args["beta"] is not None,
    }
)
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a,
    b,
    c,
    input,
    alpha,
    beta,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `s_am` is how much to increase `a`
    # by to get the element one row down (A has M rows).
    s_ap,
    s_am,
    s_ak,
    s_bp,
    s_bk,
    s_bn,
    s_cp,
    s_cm,
    s_cn,
    # Meta-parameters
    # BP: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    G: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_INPUT: tl.constexpr,
    HAS_ALPHA: tl.constexpr,
    HAS_BETA: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    NM, NN = tl.num_programs(1), tl.num_programs(2)
    i_p, i_m, i_n = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_m, i_n = tl.swizzle2d(i_m, i_n, NM, NN, G)
    tl.static_print("i_p:", i_p)

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `p_a` is a block of [BM, BK] pointers
    # `p_b` is a block of [BK, BN] pointers
    # See above `Pointer Arithmetic` section for details
    o_am = (i_m * BM + tl.arange(0, BM)) % M
    o_bn = (i_n * BN + tl.arange(0, BN)) % N
    o_k = tl.arange(0, BK)

    # here !
    p_a = a + i_p * s_ap + (o_am[:, None] * s_am + o_k[None, :] * s_ak)
    p_b = b + i_p * s_bp + (o_k[:, None] * s_bk + o_bn[None, :] * s_bn)

    b_acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        b_a = tl.load(p_a, mask=o_k[None, :] < K - k * BK, other=0.0).to(tl.float32)
        b_b = tl.load(p_b, mask=o_k[:, None] < K - k * BK, other=0.0).to(tl.float32)
        # We accumulate along the K dimension.
        b_acc += tl.dot(b_a, b_b, allow_tf32=False)
        # Advance the ptrs to the next K block.
        p_a += BK * s_ak
        p_b += BK * s_bk

    o_cm = i_m * BM + tl.arange(0, BM)
    o_cn = i_n * BN + tl.arange(0, BN)
    mask = (o_cm[:, None] < M) & (o_cn[None, :] < N)

    b_c = b_acc
    if ACTIVATION == "leaky_relu":
        b_c = leaky_relu(b_c)
    if HAS_ALPHA:
        b_c *= tl.load(alpha)
    if HAS_INPUT:
        p_i = input + i_p * s_cp + s_cm * o_cm[:, None] + s_cn * o_cn[None, :]
        b_i = tl.load(p_i, mask=mask, other=0.0).to(tl.float32)
        if HAS_BETA:
            b_i *= tl.load(beta)
        b_c += b_i

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    p_c = c + i_p * s_cp + s_cm * o_cm[:, None] + s_cn * o_cn[None, :]

    tl.store(p_c, b_c.to(c.dtype.element_ty))


# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `matmul_kernel`.
@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)


@contiguous
def addmm(
    input: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    inplace: Optional[bool] = False,
) -> torch.Tensor:
    # assert a.shape[2] == b.shape[1], 'Incompatible dimensions (A: {}x{}x{}, B: {}x{}x{})'.format(*a.shape, *b.shape)

    P, M, K = a.shape
    _, K, N = b.shape
    # Allocates output.
    c = a.new_zeros(P, M, N)
    print(c.shape, c.dtype)  #
    print(P)

    def grid(meta):
        return (
            triton.cdiv(P, 1),
            triton.cdiv(M, meta["BM"]),
            triton.cdiv(N, meta["BN"]),
        )

    matmul_kernel[grid](
        a,
        b,
        c,
        input,
        alpha,
        beta,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        c.stride(0),
        c.stride(1),
        c.stride(2),
        ACTIVATION=None,
    )
    return c


device = "cuda"

torch.manual_seed(342)
a = torch.randn((3, 4, 3), device=device, dtype=torch.float16)
b = torch.randn((3, 3, 4), device=device, dtype=torch.float16)
c = torch.randn((3, 4, 4), device=device, dtype=torch.float16).uniform_(-1, 1)


xx = addmm(c, a, b)
print('triton_output:', xx)
d = a @ b + c
print('torch_output:', d)
with open("reference.pt", "wb") as f:
    output = {
        "torch_output": d,
        "triton_output": xx,
    }
    torch.save(output, f)

torch.testing.assert_close(xx, d)
