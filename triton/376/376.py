import torch
import triton
import triton.language as tl
from torch.autograd import Function


@triton.jit
def transform_kernel(
    f_ptr,
    i_ptr,
    w_ptr,
    o_ptr,
    stride_fn,
    stride_fd,
    stride_in,
    stride_wm,
    stride_wd1,
    stride_wd2,
    stride_on,
    stride_od,
    BLOCK_SIZE_D: tl.constexpr
):

    n_idx = tl.program_id(axis=0)
    f_start_ptr = f_ptr + n_idx * stride_fn
    i_start_ptr = i_ptr + n_idx * stride_in
    o_start_ptr = o_ptr + n_idx * stride_on

    d_offsets = tl.arange(0, BLOCK_SIZE_D)
    f_ptrs = f_start_ptr + d_offsets[None, :] * stride_fd
    i_ptrs = i_start_ptr
    o_ptrs = o_start_ptr + d_offsets[None, :] * stride_od
    #########################################
    f_ptrs = tl.broadcast_to(f_ptrs, (BLOCK_SIZE_D, BLOCK_SIZE_D))
    o_ptrs = tl.broadcast_to(o_ptrs, (BLOCK_SIZE_D, BLOCK_SIZE_D))
    #########################################
    f = tl.load(f_ptrs)
    i = tl.load(i_ptrs)
    w_ptrs = (
        w_ptr
        + i * stride_wm
        + d_offsets[:, None] * stride_wd1
        + d_offsets[None, :] * stride_wd2
    )
    w = tl.load(w_ptrs)
    mul = tl.dot(f, w)
    tl.store(o_ptrs, mul)


class MyFunction(Function):
    @staticmethod
    def forward(ctx, features, indices, weights):

        assert features.is_contiguous()
        assert indices.is_contiguous()
        assert weights.is_contiguous()

        N, K, D = features.shape
        outputs = torch.empty((N, K, D), device=features.device, dtype=features.dtype)

        features_flatten = features.view(-1, D)
        indices_flatten = indices.view(
            -1,
        )
        outputs_flatten = outputs.view(-1, D)
        N_ = indices_flatten.size(0)
        transform_kernel[(N_,)](
            features_flatten,
            indices_flatten,
            weights,
            outputs_flatten,
            features_flatten.stride(0),
            features_flatten.stride(1),
            indices_flatten.stride(0),
            weights.stride(0),
            weights.stride(1),
            weights.stride(2),
            outputs_flatten.stride(0),
            outputs_flatten.stride(1),
            BLOCK_SIZE_D=D,
        )
        return outputs_flatten.view(N, K, D)


my_function = MyFunction.apply

torch.manual_seed(0)

N = 1
K = 128
D = 64
M = 10

torch_features = torch.randn(N, K, D).cuda()
torch_weights = torch.randn(M, D, D).cuda()
triton_features = torch_features.clone().detach()
triton_weights = torch_weights.clone().detach()

indices = torch.randint(low=0, high=M, size=(N, K)).long().cuda()
triton_output = my_function(triton_features, indices.int(), triton_weights)

torch_output = torch.zeros(N, K, D).cuda()
for n in range(N):
    for k in range(K):
        index = indices[n, k]
        torch_output[n, k, :] += torch.matmul(
            torch_features[n, k], torch_weights[index]
        )

print(triton_output)
print(torch_output)
print(f'max diff: {torch.max(torch.abs(triton_output - torch_output))}')

with open("reference.pt", "wb") as f:
    torch.save({"torch_output": torch_output, "triton_output": triton_output}, f)


if torch.testing.assert_close(triton_output, torch_output):
    print("✅ Forward match")
else:
    print("❌ Forward differ")
