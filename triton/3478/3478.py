import torch
import triton
import triton.language as tl


TYPE = torch.float16  # does not work


@triton.autotune(configs=[triton.Config({"BLOCK_SIZE_M": 32})], key=["M"])
@triton.jit
def simple_block_dot(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    stride_am,
    stride_an,
    stride_bm,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid // num
    pid_n = pid % num

    a_addr = (
        a_ptr
        + (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]) * stride_am
        + (pid_n * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[None, :]) * stride_an
    )
    b_addr = (
        b_ptr
        + (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]) * stride_am
        + (pid_n * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[None, :]) * stride_an
    )
    c_addr = (
        c_ptr
        + (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]) * stride_am
        + (pid_n * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[None, :]) * stride_an
    )

    a = tl.load(a_addr)
    b = tl.load(b_addr)
    c = tl.dot(a, b)
    tl.store(c_addr, c)


def block_dot(a, b):
    assert a.shape == b.shape, "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    assert a.shape[0] == a.shape[1]

    c = torch.empty((M, M), device=a.device, dtype=a.dtype)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) ** 2,)

    simple_block_dot[grid](
        a,
        b,
        c,
        M,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
    )

    return c


M = 32  # equal to block size for verification
a = torch.randn((M, M), device="cuda", dtype=TYPE)
b = torch.randn((M, M), device="cuda", dtype=TYPE)

c = block_dot(a, b)
print("a", a)
print("b", b)
print("triton", c)
print("torch", torch.matmul(a, b))

if torch.allclose(c, torch.matmul(a, b), atol=1e-2, rtol=1e-3):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")
