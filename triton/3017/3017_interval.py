import triton
import triton.language as tl
import torch


@triton.jit
def test_kernel(
    # Inputs
    k,
    k_stride_i,
    k_stride_d,
    v,
    v_stride_i,
    v_stride_e,
    # Outputs
    output,
    out_stride_i,
    out_stride_e,
):
    i = tl.arange(0, 16)

    k_loaded = tl.load(
        k + i[:, None] * k_stride_i + i[None, :] * k_stride_d,
    )
    v_loaded = tl.load(
        v + i[:, None] * v_stride_i + i[None, :] * v_stride_e,
    )
    tl.debug_barrier()

    kv = k_loaded[:, :, None] * v_loaded[:, None, :]
    tl.debug_barrier()

    context = tl.cumsum(kv, axis=0)
    tl.debug_barrier()

    out = tl.sum(context, axis=1)
    tl.debug_barrier()
    interpreter_builder.record(
        (output + i[:, None] * out_stride_i + i[None, :] * out_stride_e).handle,
        out.handle,
    )
    tl.store(output + i[:, None] * out_stride_i + i[None, :] * out_stride_e, out)


def test_case():
    torch.random.manual_seed(1)

    T = 16
    k = torch.randn((T, T), device="cuda")
    v = torch.randn((T, T), device="cuda")
    out = torch.zeros((T, T), device="cuda")

    test_kernel[(1,)](
        k,
        *k.stride(),
        v,
        *v.stride(),
        out,
        *out.stride(),
    )

    interpreter_builder.show()
    interpreter_builder.store_interval()


if __name__ == "__main__":
    test_case()
