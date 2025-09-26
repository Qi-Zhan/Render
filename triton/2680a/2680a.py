import torch

import triton
import triton.language as tl
from triton.language.extra import libdevice

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def _fused_bias_gelu_fwd_kernel(
    bias_ptr, input_ptr, output_ptr, stride, N, BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    input_start = input_ptr + row * stride
    output_start = output_ptr + row * stride

    ranges = tl.arange(0, BLOCK_SIZE)
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + ranges
        mask = cols < N

        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(input_start + cols, mask=mask, other=0.0).to(tl.float32)

        y = bias + x
        y = y * 0.5 * (1.0 + libdevice.tanh(0.79788456 * y * (1 + 0.044715 * y * y)))
        tl.store(output_start + cols, y, mask=mask)


@triton.jit
def _fused_bias_gelu_bwd_kernel(
    grad_ptr, input_ptr, bias_ptr, output_ptr, stride, N, BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    grad_start = grad_ptr + row * stride
    input_start = input_ptr + row * stride
    output_start = output_ptr + row * stride

    ranges = tl.arange(0, BLOCK_SIZE)
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + ranges
        mask = cols < N

        grad = tl.load(grad_start + cols, mask=mask, other=0.0)
        ############################ _input = tl.load(input_start + cols, mask=mask, other=0.0.tanh)
        ############################ bias = tl.load(bias_ptr + cols, mask=mask, other=0.0)
        _input = tl.load(input_start + cols, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        x = _input + bias
        tanh_out = libdevice.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
        ff = 0.5 * x * (
            (1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)
        ) + 0.5 * (1 + tanh_out)
        ff = ff * grad

        tl.store(output_start + cols, ff, mask=mask)


class FusedBiasGeLU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, _input, bias):
        _output = torch.empty_like(_input)

        N = _input.shape[-1]
        _input_arg = _input.view(-1, N)
        M = _input_arg.shape[0]
        _output_args = _output.view(-1, N)

        _fused_bias_gelu_fwd_kernel[(M,)](
            bias, _input_arg, _output_args, _input_arg.stride(0), N, BLOCK_SIZE=1024
        )

        ctx.save_for_backward(_input, bias)
        return _output

    @staticmethod
    def backward(ctx, grad):
        (
            _input,
            bias,
        ) = ctx.saved_tensors
        grad_input = torch.empty_like(_input)

        N = _input.shape[-1]
        _input_arg = _input.view(-1, N)
        M = _input_arg.shape[0]
        grad_input_arg = grad_input.view(-1, N)
        breakpoint()
        _fused_bias_gelu_bwd_kernel[(M,)](
            grad,
            _input_arg,
            bias,
            grad_input_arg,
            _input_arg.stride(0),
            N,
            BLOCK_SIZE=1024,
        )

        return grad_input, grad_input


fused_bias_gelu_triton = FusedBiasGeLU.apply
import torch

torch.manual_seed(0)
device = torch.cuda.current_device()
dtype = torch.float16

a = torch.rand(3, 3*1024, 5*1024, requires_grad=True, device=device, dtype=dtype)
a_bias = torch.rand(5*1024, requires_grad=True, device=device, dtype=dtype)

b = a.detach().clone().requires_grad_(True)
b_bias = a_bias.detach().clone().requires_grad_(True)

grad = torch.rand(3, 3*1024, 5*1024, device=device, dtype=dtype)

# output = bias_gelu_impl(a, a_bias)
target_output = fused_bias_gelu_triton(b, b_bias)

# assert torch.allclose(output, target_output, atol=1e-2, rtol=0), f"torch jit:\n{output}\ntriton jit:\n{target_output}\n\n"
print(a)
# output.backward(grad)
target_output.backward(grad)
print(b_bias.grad)
torch_bias = b_bias.grad + 4
print(torch_bias)

with open("reference.pt", "wb") as f:
    output = {
        "torch_output": torch_bias.detach(),
        "triton_output": b_bias.grad.detach(),
    }
    torch.save(output, f)

# nonzero_idx = torch.nonzero(a_bias.grad-b_bias.grad).view(-1)
# print(f"{nonzero_idx}")
# print(f"a bias grad:\n{a_bias.grad[nonzero_idx]}\nb bias grad:\n{b_bias.grad[nonzero_idx]}")
# assert torch.allclose(a.grad, b.grad, atol=1e-2, rtol=0), f"a grad:\n{a.grad}\n\nb grad:\n{b.grad}\n"
# assert torch.allclose(a_bias.grad, b_bias.grad, atol=1e-2, rtol=0), f"a bias grad:\n{a_bias.grad}\n\nb bias grad:\n{b_bias.grad}\n