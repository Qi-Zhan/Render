from typing import Any

import torch
from torch import nn

import triton
import triton.language as tl
from triton.runtime import driver

capability = torch.cuda.get_device_capability()
current_target = driver.active.get_current_target()
properties = driver.active.utils.get_device_properties(torch.device("cuda:0").index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]


def cdiv(x: int, y: int):
    return (x + y - 1) // y


class FlashFFNFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inputs: torch.Tensor,
        weight0: torch.Tensor,
        weight1: torch.Tensor,
        weight2: torch.Tensor,
        weight3: torch.Tensor,
        skip_connection: bool = False,
    ):
        M, K, N = inputs.size(0), weight1.size(0), weight1.size(1)
        with torch.no_grad():
            L = torch.rsqrt(torch.sum(inputs * inputs, dim=-1) / K)
            x1 = inputs * L.view(-1, 1) * weight0.view(1, -1)
            x2 = torch.matmul(x1, weight1)
            y2 = torch.matmul(x1, weight2)
            x3 = nn.functional.silu(x2) * y2
            if skip_connection:
                out = torch.matmul(x3, weight3) + inputs
            else:
                out = torch.matmul(x3, weight3)
        ctx.save_for_backward(inputs, weight0, weight1, weight2, weight3, L)
        ctx.M = M
        ctx.K = K
        ctx.N = N
        ctx.skip_connection = skip_connection
        return out

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        inputs, weight0, weight1, weight2, weight3, L = ctx.saved_tensors
        with torch.no_grad():
            x1 = inputs * L.view(-1, 1) * weight0.view(1, -1)
            x2 = torch.matmul(x1, weight1)
            y2 = torch.matmul(x1, weight2)
            x3 = nn.functional.silu(x2) * y2

            do = grad_outputs[0].contiguous()
            dx3 = torch.matmul(do, weight3.T)
            dw3 = torch.matmul(x3.T, do)
            sigmoid_x2 = nn.functional.sigmoid(x2)
            dy2 = dx3 * sigmoid_x2 * x2
            dw2 = torch.matmul(x1.T, dy2)
            dx2 = dx3 * (y2 * sigmoid_x2 + x3 * (1 - sigmoid_x2))
            dw1 = torch.matmul(x1.T, dx2)
            dx1 = torch.matmul(dy2, weight2.T) + torch.matmul(dx2, weight1.T)
            dw0 = (dx1 * inputs * L.view(-1, 1)).sum(dim=0).view(-1)
            coefficient = torch.sum(dx1 * x1, dim=-1) * (L**2) / ctx.K
            dx = (
                dx1 * L.view(-1, 1) * weight0.view(1, -1)
                - coefficient.view(-1, 1) * inputs
            )

            if ctx.skip_connection:
                dx += do
            return dx, dw0, dw1, dw2, dw3, None


flash_ffn = FlashFFNFunction.apply


@triton.jit
def _rms_norm_fwd_kernel(
    X,
    W,
    L,
    OUT,
    M,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # reverse square mean root
    pid = tl.program_id(0)
    offset_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_k = tl.arange(0, K)
    mask = offset_m < M

    w0 = tl.load(W + offset_k)
    x = tl.load(
        X + offset_m[:, None] * K + offset_k[None, :], mask=mask[:, None], other=1e-8
    )

    l_i = tl.rsqrt(tl.sum(x * x, axis=1) / K)
    tl.store(
        OUT + offset_m[:, None] * K + offset_k[None, :],
        (x * l_i[:, None] * w0[None, :]).to(OUT.dtype.element_ty),
        mask=mask[:, None],
    )
    tl.store(L + offset_m, l_i.to(L.dtype.element_ty), mask=mask)


@triton.jit
def _rms_norm_bwd_kernel(
    X,
    W,
    L,
    Y,
    DY,
    Locks,
    DX,
    DW,
    M,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    offset_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_k = tl.arange(0, K)
    mask = offset_m < M

    w0 = tl.load(W + offset_k)
    x = tl.load(
        X + offset_m[:, None] * K + offset_k[None, :], mask=mask[:, None], other=1e-8
    )
    y = tl.load(
        Y + offset_m[:, None] * K + offset_k[None, :], mask=mask[:, None], other=1e-8
    )
    dy = tl.load(
        DY + offset_m[:, None] * K + offset_k[None, :], mask=mask[:, None], other=1e-8
    )
    l_i = tl.load(L + offset_m, mask=mask, other=float(0.0))

    coefficient = tl.sum(dy * y, axis=1) * l_i * l_i / K
    dx = dy * l_i[:, None] * w0[None, :] - coefficient[:, None] * x
    tl.store(
        DX + offset_m[:, None] * K + offset_k[None, :],
        dx.to(DX.dtype.element_ty),
        mask=mask[:, None],
    )
    while tl.atomic_cas(Locks, 0, 1) == 1:
        pass
    previous_dw = tl.load(DW + offset_k)
    dw = tl.sum(dy * x * l_i[:, None], axis=0)
    tl.store(DW + offset_k, (previous_dw + dw).to(W.dtype.element_ty))
    tl.atomic_xchg(Locks, 0)


@triton.jit
def _silu_gate_fwd_kernel(
    X2, Y2, X3, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offset_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offset_m < M
    mask_n = offset_n < N
    x2 = tl.load(
        X2 + offset_m[:, None] * N + offset_n[None, :],
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0,
    )
    y2 = tl.load(
        Y2 + offset_m[:, None] * N + offset_n[None, :],
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0,
    )
    tl.store(
        X3 + offset_m[:, None] * N + offset_n[None, :],
        (x2 * y2 * tl.sigmoid(x2)).to(X3.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def _silu_gate_bwd_kernel(
    X2, Y2, X3, DX3, DX2, DY2, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    # reverse square mean root
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offset_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offset_m < M
    mask_n = offset_n < N
    x2 = tl.load(
        X2 + offset_m[:, None] * N + offset_n[None, :],
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0,
    )
    y2 = tl.load(
        Y2 + offset_m[:, None] * N + offset_n[None, :],
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0,
    )
    # x3 = tl.load(X3 + offset_m[:, None] * N + offset_n[None, :],
    #              mask=mask_m[:, None] & mask_n[None, :], other=0.)
    dx3 = tl.load(
        DX3 + offset_m[:, None] * N + offset_n[None, :],
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0,
    )

    sigmoid_x2 = tl.sigmoid(x2)
    x3 = x2 * y2 * sigmoid_x2
    tl.store(
        DX2 + offset_m[:, None] * N + offset_n[None, :],
        (dx3 * (y2 * sigmoid_x2 + x3 * (1 - sigmoid_x2))).to(DX2.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )
    tl.store(
        DY2 + offset_m[:, None] * N + offset_n[None, :],
        (dx3 * x2 * sigmoid_x2).to(DY2.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


class FlashFFNFunctionV2(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inputs: torch.Tensor,
        weight0: torch.Tensor,
        weight1: torch.Tensor,
        weight2: torch.Tensor,
        weight3: torch.Tensor,
        skip_connection: bool = False,
        use_fused_rmsnorm: bool = True,
        use_fused_silu_gate: bool = False,
    ):
        if inputs.ndim == 3:
            M, head_size, head_dim = inputs.shape
            K = head_size * head_dim
        else:
            assert inputs.ndim == 2
            M, K = inputs.size(0), inputs.size(1)
        N = weight1.size(1)
        assert N == weight3.size(0) and N == weight2.size(1)
        assert weight2.size(0) == weight1.size(0) == weight3.size(1) == K
        BLOCK_M, BLOCK_N = min(M, 256), min(N, 128)
        with torch.no_grad():
            ctx.use_fused_rmsnorm = use_fused_rmsnorm
            if use_fused_rmsnorm:
                L = torch.empty(M, dtype=torch.float32, device=inputs.device)
                x1 = torch.empty_like(inputs)
                rms_norm_grid = lambda meta: (cdiv(meta["M"], meta["BLOCK_M"]),)
                _rms_norm_fwd_kernel[rms_norm_grid](
                    inputs, weight0, L, x1, M=M, K=K, BLOCK_M=BLOCK_M, num_warps=4
                )
                ctx.rms_norm_grid = rms_norm_grid
            else:
                L = torch.rsqrt(torch.sum(inputs * inputs, dim=-1) / K)
                x1 = inputs * L.view(-1, 1) * weight0.view(1, -1)
            x2 = torch.matmul(x1, weight1)
            y2 = torch.matmul(x1, weight2)
            ctx.use_fused_silu_gate = use_fused_silu_gate
            if use_fused_silu_gate:
                x3 = torch.empty_like(x2)
                silu_gate_grid = lambda meta: (
                    cdiv(meta["M"], meta["BLOCK_M"]),
                    cdiv(meta["N"], meta["BLOCK_N"]),
                )
                _silu_gate_fwd_kernel[silu_gate_grid](
                    x2, y2, x3, M=M, N=N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=4
                )
                ctx.silu_gate_grid = silu_gate_grid
            else:
                x3 = nn.functional.silu(x2) * y2
            if skip_connection:
                out = torch.matmul(x3, weight3) + inputs
            else:
                out = torch.matmul(x3, weight3)
        ctx.save_for_backward(inputs, weight0, weight1, weight2, weight3, L)
        ctx.M = M
        ctx.K = K
        ctx.N = N
        ctx.BLOCK_M = BLOCK_M
        ctx.BLOCK_N = BLOCK_N
        ctx.skip_connection = skip_connection
        return out

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        inputs, weight0, weight1, weight2, weight3, L = ctx.saved_tensors

        with torch.no_grad():
            # forward recompute
            x1 = inputs * L.view(-1, 1) * weight0.view(1, -1)
            x2 = torch.matmul(x1, weight1)
            y2 = torch.matmul(x1, weight2)
            x3 = nn.functional.silu(x2) * y2

            # backward compute
            do = grad_outputs[0].contiguous()
            dx3 = torch.matmul(do, weight3.T)
            dw3 = torch.matmul(x3.T, do)

            if ctx.use_fused_silu_gate:
                dx2 = torch.empty_like(x2)
                dy2 = torch.empty_like(y2)
                _silu_gate_bwd_kernel[ctx.silu_gate_grid](
                    x2,
                    y2,
                    x3,
                    dx3,
                    dx2,
                    dy2,
                    M=ctx.M,
                    N=ctx.N,
                    BLOCK_M=ctx.BLOCK_M,
                    BLOCK_N=ctx.BLOCK_N,
                    num_warps=4,
                )
            else:
                sigmoid_x2 = nn.functional.sigmoid(x2)
                dy2 = dx3 * sigmoid_x2 * x2
                dx2 = dx3 * (y2 * sigmoid_x2 + x3 * (1 - sigmoid_x2))
            dw1 = torch.matmul(x1.T, dx2)
            dw2 = torch.matmul(x1.T, dy2)
            dx1 = torch.matmul(dy2, weight2.T) + torch.matmul(dx2, weight1.T)
            if ctx.use_fused_rmsnorm:
                dx = torch.empty_like(inputs)
                dw0 = torch.zeros_like(weight0)
                Locks = torch.zeros(1, dtype=torch.int32, device=inputs.device)
                _rms_norm_bwd_kernel[ctx.rms_norm_grid](
                    inputs,
                    weight0,
                    L,
                    x1,
                    dx1,
                    Locks,  # X, W, L, Y, DY, Locks,
                    dx,
                    dw0,
                    M=ctx.M,
                    K=ctx.K,
                    BLOCK_M=ctx.BLOCK_M,
                )
            else:
                dw0 = (dx1 * inputs * L.view(-1, 1)).sum(dim=0).view(-1)
                coefficient = torch.sum(dx1 * x1, dim=-1) * (L**2) / ctx.K
                dx = (
                    dx1 * L.view(-1, 1) * weight0.view(1, -1)
                    - coefficient.view(-1, 1) * inputs
                )

        if ctx.skip_connection:
            dx += do
        return dx, dw0, dw1, dw2, dw3, None, None, None


flash_ffn_v2 = FlashFFNFunctionV2.apply


@triton.jit
def _flash_ffn_fwd_kernel(
    X,
    W0,
    W1,
    W2,
    W3,
    L,
    Out,
    skip_connection,
    M,
    K: tl.constexpr,  # K is BLOCK_DMODEL
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    # BLOCK_DMODEL = tl.multiple_of(BLOCK_DMODEL, BLOCK_N)
    offset_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offset_m < M
    offset_k = tl.arange(0, K)
    w0 = tl.load(W0 + offset_k)
    x1 = tl.load(
        X + offset_m[:, None] * K + offset_k[None, :],
        mask=mask_m[:, None],
        other=float(0.0),
    )  # (BLOCK_M, BLOCK_DMODEL)
    if skip_connection:
        acc = tl.zeros([BLOCK_M, K], dtype=tl.float32) + x1
    else:
        acc = tl.zeros([BLOCK_M, K], dtype=tl.float32)

    # 1) RMSNorm
    l_i = tl.rsqrt(tl.sum(x1 * x1, axis=1) / K)
    x1 *= l_i[:, None] * w0[None, :]
    tl.store(L + offset_m, l_i.to(L.dtype.element_ty), mask=mask_m)

    num_wx_blocks = tl.cdiv(N, BLOCK_N)
    tl.device_assert(N % BLOCK_N == 0, "BLOCK_DMODEL % BLOCK_N must be zeros!")
    for block_idx in range(num_wx_blocks):
        offset_w = block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        w1 = tl.load(
            W1 + offset_k[:, None] * N + offset_w[None, :]
        )  # (BLOCK_DMODEL, BLOCK_N)
        w2 = tl.load(
            W2 + offset_k[:, None] * N + offset_w[None, :]
        )  # (BLOCK_DMODEL, BLOCK_N)
        w3 = tl.load(
            W3 + offset_w[:, None] * K + offset_k[None, :]
        )  # (BLOCK_N, BLOCK_DMODEL)

        # 2) X * W12
        x2 = tl.dot(x1, w1)  # (BLOCK_M, BLOCK_N)
        y2 = tl.dot(x1, w2)  # (BLOCK_M, BLOCK_N)

        # 3) SiLU + Gate
        x3 = x2 * y2 * tl.sigmoid(x2)  # (BLOCK_M, BLOCK_N)

        # 4) acc add
        acc += tl.dot(x3, w3)  # (BLOCK_M, BLOCK_DMODEL)

    interpreter_builder.record(
        (Out + offset_m[:, None] * K + offset_k[None, :]).handle,
        (acc.to(Out.type.element_ty)).handle,
        # mask=(mask_m[:, None].handle),
    )
    tl.store(
        Out + offset_m[:, None] * K + offset_k[None, :],
        acc.to(Out.dtype.element_ty),
        mask=mask_m[:, None],
    )


@triton.jit
def _flash_ffn_bwd_kernel(
    X,
    W0,
    W1,
    W2,
    W3,
    L,
    MLocks,
    NLocks,
    DO,
    DX,
    DW0,
    DW1,
    DW2,
    DW3,
    M,
    K: tl.constexpr,  # K is BLOCK_DMODEL
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tl.device_assert(N % BLOCK_N == 0, "BLOCK_DMODEL % BLOCK_N must be zeros!")
    wx_block_idx = tl.program_id(0)

    offset_k = tl.arange(0, K)
    offset_w = wx_block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    # (BLOCK_DMODEL, BLOCK_N)
    w0 = tl.load(W0 + offset_k)
    w1 = tl.load(W1 + offset_k[:, None] * N + offset_w[None, :])
    w2 = tl.load(W2 + offset_k[:, None] * N + offset_w[None, :])
    # (BLOCK_N, BLOCK_DMODEL)
    w3 = tl.load(W3 + offset_w[:, None] * K + offset_k[None, :])

    dw0 = tl.zeros([K], dtype=tl.float32)
    dw1 = tl.zeros([K, BLOCK_N], dtype=tl.float32)
    dw2 = tl.zeros([K, BLOCK_N], dtype=tl.float32)
    dw3 = tl.zeros([BLOCK_N, K], dtype=tl.float32)

    num_x_blocks = tl.cdiv(M, BLOCK_M)
    for x_block_idx in range(num_x_blocks):
        offset_m = x_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offset_m < M
        # shape of x, o, do: (BLOCK_M, BLOCK_DMODEL)
        x = tl.load(
            X + offset_m[:, None] * K + offset_k[None, :],
            mask=mask_m[:, None],
            other=0.0,
        )
        do = tl.load(
            DO + offset_m[:, None] * K + offset_k[None, :],
            mask=mask_m[:, None],
            other=0.0,
        )
        l_i = tl.load(
            L + offset_m, mask=mask_m, other=0.0
        )  # \sqrt{\frac{1}{n}\sum_j x_j^2}

        # stage 1: recompute middle variables
        x1 = x * l_i[:, None] * w0[None, :]  # RMSNorm, (BLOCK_M, BLOCK_DMODEL)
        x2 = tl.dot(x1, w1)  # (BLOCK_M, BLOCK_N)
        y2 = tl.dot(x1, w2)  # (BLOCK_M, BLOCK_N)
        sigma_x2 = tl.sigmoid(x2)  # (BLOCK_M, BLOCK_N)
        x3 = x2 * y2 * sigma_x2  # (BLOCK_M, BLOCK_N)
        # o = tl.dot(x3, w3)  # (BLOCK_M, BLOCK_DMODEL)

        # stage 2: backward
        # dx = do * w3.T， (BLOCK_M, BLOCK_DMODEL) * (BLOCK_N, BLOCK_DMODEL)^T -> (BLOCK_M, BLOCK_N)
        dx3 = tl.dot(do, tl.trans(w3))  # (BLOCK_M, BLOCK_N)

        # dw3 = x3.T @ do, (BLOCK_M, BLOCK_N)^T * (BLOCK_N, BLOCK_DMODEL) -> (BLOCK_N, BLOCK_DMODEL)
        dw3 += tl.dot(tl.trans(x3), do)

        # dx3 * [y * sigmoid(x) + z * (1-sigmoid(x))], (BLOCK_M, BLOCK_N)
        dx2w1 = dx3 * (y2 * sigma_x2 + x3 * (1.0 - sigma_x2))
        dw1 += tl.dot(tl.trans(x1), dx2w1)  # (BLOCK_DMODEL, BLOCK_N)

        # dx3 * x * sigmoid(x), (BLOCK_M, BLOCK_N)
        dy2w2 = dx3 * x2 * sigma_x2
        dw2 += tl.dot(tl.trans(x1), dy2w2)  # (BLOCK_DMODEL, BLOCK_N)

        # (BLOCK_M, BLOCK_N) * (BLOCK_DMODEL, BLOCK_N)^T -> (BLOCK_M, BLOCK_DMODEL)
        dx1 = tl.dot(dx2w1, tl.trans(w1)) + tl.dot(dy2w2, tl.trans(w2))
        dw0 += tl.sum(dx1 * x * l_i[:, None], axis=0)  # (BLOCK_DMODEL,)
        coefficient = tl.sum(x1 * dx1, axis=1) * l_i * l_i / K  # (BLOCK_DMODEL,)
        dx = (
            dx1 * l_i[:, None] * w0[None, :] - coefficient[:, None] * x
        )  # (BLOCK_M, BLOCK_DMODEL)
        while tl.atomic_cas(MLocks + x_block_idx, 0, 1) == 1:
            pass
        previous_dx = tl.load(
            DX + offset_m[:, None] * K + offset_k[None, :],
            mask=mask_m[:, None],
            other=0.0,
        )
        tl.store(
            DX + offset_m[:, None] * K + offset_k[None, :],
            (previous_dx + dx).to(DX.dtype.element_ty),
            mask=mask_m[:, None],
        )
        tl.atomic_xchg(MLocks + x_block_idx, 0)

    while tl.atomic_cas(NLocks + wx_block_idx, 0, 1) == 1:
        pass
    previous_dw0 = tl.load(DW0 + offset_k)
    tl.store(DW0 + offset_k, (previous_dw0 + dw0).to(DW0.dtype.element_ty))
    tl.atomic_xchg(NLocks + wx_block_idx, 0)

    tl.store(
        DW1 + offset_k[:, None] * N + offset_w[None, :], dw1.to(DW1.dtype.element_ty)
    )
    tl.store(
        DW2 + offset_k[:, None] * N + offset_w[None, :], dw2.to(DW2.dtype.element_ty)
    )
    tl.store(
        DW3 + offset_w[:, None] * K + offset_k[None, :], dw3.to(DW3.dtype.element_ty)
    )


class FlashFFNFunctionV3(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inputs: torch.Tensor,
        weight0: torch.Tensor,
        weight1: torch.Tensor,
        weight2: torch.Tensor,
        weight3: torch.Tensor,
        skip_connection: bool = False,
    ):
        BLOCK_M, BLOCK_N = 16, 64
        if inputs.ndim == 3:
            M, head_size, head_dim = inputs.shape
            K = head_size * head_dim
        else:
            assert inputs.ndim == 2
            M, K = inputs.size(0), inputs.size(1)
        assert weight0.numel() == K
        assert weight1.size(0) == K and weight1.ndim == 2
        N = weight1.size(1)
        assert N % BLOCK_N == 0, "BLOCK_DMODEL % BLOCK_N must be zeros!"
        assert weight2.size(0) == K and weight2.ndim == 2
        assert weight2.size(1) == N
        assert weight3.size(0) == N
        assert weight3.size(1) == K

        Out = torch.zeros_like(inputs)
        L = torch.zeros(M, dtype=torch.float32, device=inputs.device)
        _flash_ffn_fwd_kernel[((M + BLOCK_M - 1) // BLOCK_M,)](
            X=inputs,
            W0=weight0,
            W1=weight1,
            W2=weight2,
            W3=weight3,
            L=L,
            Out=Out,
            M=M,
            K=K,
            N=N,
            skip_connection=skip_connection,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_stages=2,
        )
        ctx.save_for_backward(inputs, weight0, weight1, weight2, weight3, L)
        ctx.BLOCK_M = BLOCK_M
        ctx.BLOCK_N = BLOCK_N
        ctx.M = M
        ctx.K = K
        ctx.N = N
        ctx.skip_connection = skip_connection
        return Out

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        inputs, weight0, weight1, weight2, weight3, L = ctx.saved_tensors
        with torch.no_grad():
            DO = grad_outputs[0].contiguous()
            if ctx.skip_connection:
                DX = DO.clone().detach_()
            else:
                DX = torch.zeros_like(inputs)
            DW0 = torch.zeros_like(weight0)
            DW1 = torch.empty_like(weight1)
            DW2 = torch.empty_like(weight2)
            DW3 = torch.empty_like(weight3)
            num_m_blocks = (ctx.M + ctx.BLOCK_M - 1) // ctx.BLOCK_M
            MLocks = torch.zeros(num_m_blocks, dtype=torch.int32, device=inputs.device)
            num_n_blocks = (ctx.N + ctx.BLOCK_N - 1) // ctx.BLOCK_N
            NLocks = torch.zeros(num_n_blocks, dtype=torch.int32, device=inputs.device)
            _flash_ffn_bwd_kernel[(ctx.N // ctx.BLOCK_N,)](
                X=inputs,
                W0=weight0,
                W1=weight1,
                W2=weight2,
                W3=weight3,
                L=L,
                MLocks=MLocks,
                NLocks=NLocks,
                DO=DO,
                DX=DX,
                DW0=DW0,
                DW1=DW1,
                DW2=DW2,
                DW3=DW3,
                M=ctx.M,
                K=ctx.K,
                N=ctx.N,
                BLOCK_M=ctx.BLOCK_M,
                BLOCK_N=ctx.BLOCK_N,
                num_stages=2,
            )
            return DX, DW0, DW1, DW2, DW3, None


flash_ffn_v3 = FlashFFNFunctionV3.apply


class FlashFFN(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_inner: int | None = None,
        dtype=torch.float32,
        skip_connection: bool = False,
        version: int = 0,
        **kwargs
    ):
        super().__init__()
        assert version in {0, 1, 2, 3}
        self.d_model = d_model
        self.d_inner = d_inner or d_model
        self.dtype = dtype
        self.version = version
        self.skip_connection = skip_connection
        self.weight0 = torch.nn.Parameter(torch.ones(self.d_model, dtype=dtype))
        self.weight1 = torch.nn.Parameter(
            torch.empty(self.d_model, self.d_inner, dtype=dtype)
        )
        self.weight2 = torch.nn.Parameter(
            torch.empty(self.d_model, self.d_inner, dtype=dtype)
        )
        self.weight3 = torch.nn.Parameter(
            torch.empty(self.d_inner, self.d_model, dtype=dtype)
        )
        self.kwargs = kwargs
        torch.nn.init.normal_(self.weight1, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.weight2, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.weight3, mean=0.0, std=0.02)

    def forward(self, inputs):
        if self.version == 0:
            if self.d_model <= 128:
                version = 3
            else:
                version = 2
        else:
            version = self.version
        if version == 1:
            return flash_ffn(
                inputs,
                self.weight0,
                self.weight1,
                self.weight2,
                self.weight3,
                self.skip_connection,
            )
        elif version == 2:
            use_fused_rmsnorm = self.kwargs.get("use_fused_rmsnorm", True)
            use_fused_silu_gate = self.kwargs.get("use_fused_silu_gate", False)
            return flash_ffn_v2(
                inputs,
                self.weight0,
                self.weight1,
                self.weight2,
                self.weight3,
                self.skip_connection,
                use_fused_rmsnorm,
                use_fused_silu_gate,
            )
        else:
            return flash_ffn_v3(
                inputs,
                self.weight0,
                self.weight1,
                self.weight2,
                self.weight3,
                self.skip_connection,
            )


import torch
from torch import nn


torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
device = torch.device("cuda")
dmodel = 128
inputs = torch.randn(
    (10000, dmodel), dtype=torch.float32, device=device, requires_grad=True
)
device = device
dmodel = dmodel

skip_connection = False
ffn = FlashFFN(
    dmodel,
    d_inner=dmodel // 2,
    version=3,
    skip_connection=skip_connection,
    use_fused_rmsnorm=True,
    use_fused_silu_gate=False,
).to(device)
inputs = inputs.clone().detach()
inputs.requires_grad = True
w0 = ffn.weight0.data.clone().detach()
w0.requires_grad = True
w1 = ffn.weight1.data.clone().detach()
w1.requires_grad = True
w2 = ffn.weight2.data.clone().detach()
w2.requires_grad = True
w3 = ffn.weight3.data.clone().detach()
w3.requires_grad = True

# torch impl
output = ffn(inputs)

interpreter_builder.store_interval()
