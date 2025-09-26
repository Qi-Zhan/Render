import torch

def calculate_scale(inp):
    amax = torch.abs(torch.max(inp))
    scale = 448.0 / torch.clamp(amax, min=1e-12)
    scale = scale.to(torch.float32)
    return scale

dtype = torch.bfloat16
torch.manual_seed(0)
inp = torch.randn(16, 16, 768, dtype=dtype, device="cuda")
eager_scale = calculate_scale(inp)
compile_scale = torch.compile(calculate_scale)(inp)
torch.testing.assert_close(eager_scale, compile_scale)