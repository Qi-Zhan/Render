import torch
import triton.language as tl
import triton

torch.manual_seed(0)

N = 64
A = torch.rand(N, N, dtype=torch.float32).cuda() * 10.0
B = torch.rand(N, N, dtype=torch.float32).cuda() * 10.0

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, N: tl.constexpr, allow_tf32: tl.constexpr):
    offs = tl.arange(0, N)[:, None] * N + tl.arange(0, N)
    a = tl.load(a_ptr + offs)
    b = tl.load(b_ptr + offs)
    c = tl.dot(a, b, allow_tf32=allow_tf32)
    tl.store(c_ptr + offs, c)

def matmul(a, b, allow_tf32):
    N = a.shape[0]
    c = torch.empty((N, N), dtype=a.dtype, device=a.device)
    grid = lambda META: (1, 1)
    matmul_kernel[grid](a, b, c, N, allow_tf32)
    return c

torch.backends.cuda.matmul.allow_tf32 = False
C_ref_no_tf32 = A @ B

torch.backends.cuda.matmul.allow_tf32 = True
C_ref_tf32 = A @ B

C_triton_no_tf32 = matmul(A, B, allow_tf32=False)
C_triton_tf32 = matmul(A, B, allow_tf32=True)

# print(f"{torch.abs(C_ref_no_tf32 - C_ref_tf32).mean() = }")
# print(f"{torch.abs(C_ref_no_tf32 - C_triton_no_tf32).mean() = }")
# print(f"{torch.abs(C_ref_no_tf32 - C_triton_tf32).mean() = }")
print(f"{torch.abs(C_ref_tf32 - C_triton_tf32).mean() = }")

print(f"max diff: {torch.abs(C_ref_tf32 - C_triton_tf32).max()}")
with open("reference.pt", "wb") as f:
    output = {
        "torch_output": C_ref_tf32,
        "triton_output": C_triton_tf32,
    }
    torch.save(output, f)
