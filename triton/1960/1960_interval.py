import torch
import triton


@triton.jit
def make_block(inp_sz, blk_sz, blk_off=0):
    blk = triton.language.arange(0, blk_sz) + blk_off
    return blk, blk < inp_sz


@triton.jit
def sum(
    p_out,
    p_inp,
    num_vec,
    vec_sz,
    axis: triton.language.constexpr,
    blk_sz: triton.language.constexpr,
):
    pid = triton.language.program_id(0)

    if axis == 0:
        num_vec, vec_sz = vec_sz, num_vec
        elem_st = num_vec
        p_inp += pid
    else:
        elem_st = 1
        p_inp += pid * vec_sz

    p_out += pid
    acc = 0.0

    for blk_off in range(0, vec_sz, blk_sz):
        blk, msk = make_block(vec_sz, blk_sz, blk_off)
        inp = triton.language.load(p_inp + blk * elem_st, msk, 0)
        acc += triton.language.sum(inp, 0)

    interpreter_builder.record(p_out.handle, acc.handle)
    triton.language.store(p_out, acc)


device = "cuda"
dtype = torch.float16
axis = 0
ctor_args = {"device": device, "dtype": dtype}
inp = torch.tensor(
    [
        [
            30.6875,
            -40.4375,
            -29.1719,
            81.1875,
            23.3125,
            3.6348,
            6.0508,
            -100.5000,
            -6.0273,
            11.6562,
        ],
        [
            21.5469,
            11.3438,
            14.0000,
            33.7188,
            13.4844,
            -18.0938,
            27.5156,
            -29.0625,
            -1.7559,
            20.8594,
        ],
        [
            28.6406,
            -30.1094,
            22.6406,
            -35.8750,
            3.5410,
            -66.1250,
            15.6016,
            -22.4375,
            50.0625,
            39.6562,
        ],
        [
            5.3281,
            -75.1875,
            -13.3828,
            -39.9688,
            -59.9062,
            14.7812,
            -23.0625,
            -3.4336,
            -34.8125,
            32.7812,
        ],
        [
            20.1406,
            -33.4375,
            -50.3438,
            -25.2812,
            69.6250,
            2.2090,
            18.9062,
            16.3750,
            -7.9922,
            27.1562,
        ],
    ],
    **ctor_args,
)
num_vec, vec_sz = inp.shape
out_sz = vec_sz if axis == 0 else num_vec
out = torch.empty(out_sz, **ctor_args)



def grid(meta):
    return (out_sz,)


sum[grid](
    out,
    inp,
    num_vec,
    vec_sz,
    axis,
    4,
)


interpreter_builder.store_interval()
