import torch
import triton
import triton.language as tl


@triton.jit
def test_kernel(
    X,
    Y,
    O,
    stride_x_m,
    stride_x_k,
    stride_y_k,
    stride_y_n,
    stride_o_m,
    stride_o_n,
    m: tl.constexpr,
    k: tl.constexpr,
    n: tl.constexpr,
):
    offset_x = (
        tl.arange(0, m)[:, None] * stride_x_m + tl.arange(0, k)[None, :] * stride_x_k
    )
    x = tl.load(X + offset_x)
    offset_y = (
        tl.arange(0, k)[:, None] * stride_y_k + tl.arange(0, n)[None, :] * stride_y_n
    )
    y = tl.load(Y + offset_y)
    o = tl.dot(x, y)
    offset_o = (
        tl.arange(0, m)[:, None] * stride_o_m + tl.arange(0, n)[None, :] * stride_o_n
    )
    interpreter_builder.record((O + offset_o).handle, o.handle)
    tl.store(O + offset_o, o)


m, k, n = 64, 64, 64

torch.manual_seed(0)
x = torch.randn(size=[m, k], dtype=torch.float64, device="cuda")
y = torch.randn(size=[k, n], dtype=torch.float64, device="cuda")
o = torch.zeros(size=[m, n], dtype=torch.float32, device="cuda")
test_kernel[(1,)](
    x.to(torch.float32),
    y.to(torch.float32),
    o,
    x.stride(0),
    x.stride(1),
    y.stride(0),
    y.stride(1),
    o.stride(0),
    o.stride(1),
    m,
    k,
    n,
)
interpreter_builder.store_interval()
