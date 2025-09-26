import pytest
import torch

import triton


def test_matmul(MODE, TRANS_A, TRANS_B, BLOCK, DTYPE, Z, H, M, N, K):
    seed = 0
    torch.manual_seed(seed)
    is_sdd = MODE == "sdd"
    is_dsd = MODE == "dsd"
    is_dds = MODE == "dds"
    do_sparsify = lambda x: triton.testing.sparsify_tensor(x, layout, BLOCK)
    do_mask = lambda x: triton.testing.mask_tensor(x, layout, BLOCK)
    a_shape = (Z, H, K, M) if TRANS_A else (Z, H, M, K)
    b_shape = (Z, H, N, K) if TRANS_B else (Z, H, K, N)
    c_shape = (Z, H, M, N)
    shape = {
        "sdd": (M, N),
        "dsd": (a_shape[2], a_shape[3]),
        "dds": (b_shape[2], b_shape[3]),
    }[MODE]
    layout = torch.randint(2, (H, shape[0] // BLOCK, shape[1] // BLOCK))
    # layout[1, 2, :] = 0
    # layout[1, :, 1] = 0
    # create data
    a_ref, a_tri = triton.testing.make_pair(a_shape, alpha=0.1, dtype=DTYPE)
    b_ref, b_tri = triton.testing.make_pair(b_shape, alpha=0.1, dtype=DTYPE)
    dc_ref, dc_tri = triton.testing.make_pair(c_shape, dtype=DTYPE)
    # compute [torch]
    dc_ref = do_mask(dc_ref) if is_sdd else dc_ref
    a_ref = do_mask(a_ref) if is_dsd else a_ref
    b_ref = do_mask(b_ref) if is_dds else b_ref
    a_ref.retain_grad()
    b_ref.retain_grad()
    c_ref = torch.matmul(
        a_ref.transpose(2, 3) if TRANS_A else a_ref,
        b_ref.transpose(2, 3) if TRANS_B else b_ref,
    )
    c_ref.backward(dc_ref)
    c_ref = do_sparsify(c_ref) if is_sdd else c_ref
    da_ref = do_sparsify(a_ref.grad) if is_dsd else a_ref.grad
    db_ref = do_sparsify(b_ref.grad) if is_dds else b_ref.grad
    # triton result
    dc_tri = do_sparsify(dc_tri) if is_sdd else dc_tri
    a_tri = do_sparsify(a_tri) if is_dsd else a_tri
    b_tri = do_sparsify(b_tri) if is_dds else b_tri
    a_tri.retain_grad()
    b_tri.retain_grad()
    op = triton.ops.blocksparse.matmul(
        layout, BLOCK, MODE, trans_a=TRANS_A, trans_b=TRANS_B, device="cuda"
    )
    c_tri = triton.testing.catch_oor(lambda: op(a_tri, b_tri), pytest)
    triton.testing.catch_oor(lambda: c_tri.backward(dc_tri), pytest)
    da_tri = a_tri.grad
    db_tri = b_tri.grad
    # compare
    triton.testing.assert_almost_equal(c_ref, c_tri)
    triton.testing.assert_almost_equal(da_ref, da_tri)
    triton.testing.assert_almost_equal(db_ref, db_tri)


test_matmul("dsd", False, False, 32, torch.float16, Z=1, H=1, M=4096, N=4096, K=4096)
