import torch
import torch.nn.functional as F
from torch import Tensor

from .interval import Interval, lo_hi, torch_where
from .wrapper import tensor_with_interval_round, tensor_with_interval_round_reverse

torch.set_printoptions(precision=10)

def print_free_memory(device=0):
    torch.cuda.set_device(device)
    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocated_memory = torch.cuda.memory_allocated(device)
    reserved_memory = torch.cuda.memory_reserved(device)
    free_memory = total_memory - reserved_memory - allocated_memory
    print(f"Allocated Memory {device}: {allocated_memory / 1e9:.2f} / {total_memory / 1e9:.2f} GB")
    print(f"Free Memory {device}: {free_memory / 1e9:.2f} / {total_memory / 1e9:.2f} GB")

def assert_normal(x):
    assert not torch.any(torch.isnan(x)), "Input contains NaN values"
    assert not torch.any(torch.isposinf(x)), "Input contains inf values"
    # assert not torch.any(torch.isneginf(x)), "Input contains -inf values"

def check(tensor: Tensor):
    lo, hi = tensor.interval.lo, tensor.interval.hi
    unequal_mask = lo <= tensor
    if not torch.all(unequal_mask).item():
        unequal_indices = torch.nonzero(~unequal_mask)  # Find indices where lo > tensor
        for idx in unequal_indices:
            print(f"Element at index {idx} is unequal: {tensor.torch_get_item(tuple(idx))} (lo: {lo.torch_get_item(tuple(idx))})")
            breakpoint()
    unequal_mask = hi >= tensor
    if not torch.all(unequal_mask).item():
        unequal_indices = torch.nonzero(~unequal_mask)  # Find indices where lo > tensor
        for idx in unequal_indices:
            print(f"Element at index {idx} is unequal: {tensor.torch_get_item(tuple(idx))} (hi: {hi.torch_get_item(tuple(idx))})")
            breakpoint()
    assert torch.all(lo <= tensor).item()
    assert torch.all(tensor <= hi).item()

def max_relative_error(tensor: Tensor):
    if not hasattr(tensor, "interval"):
        return 0
    lo, hi = tensor.interval.lo, tensor.interval.hi
    lo_max = torch_abs(torch_div(torch_sub(tensor, lo), lo))
    hi_max = torch_abs(torch_div(torch_sub(tensor, hi), hi))
    lo_max = torch_where( lo_max.isnan(), 0.0, lo_max)
    hi_max = torch_where( hi_max.isnan(), 0.0, hi_max)
    lo_max = torch_max(lo_max).item()
    hi_max = torch_max(hi_max).item()
    max_rel_error = max(lo_max, hi_max)
    return max_rel_error

######################## Wrapper functions Begin ########################
torch_float = Tensor.float
Tensor.float = tensor_with_interval_round(Tensor.float)

torch_add = torch.add
def addwrapper(a: Tensor, b: Tensor) -> Tensor:
    c = torch_add(a, b)
    if isinstance(a, Tensor) and not torch.is_floating_point(a):
        return c
    if isinstance(b, Tensor) and not torch.is_floating_point(b):
        return c
    c.interval = interval_add(a, b)
    c.interval.round()
    check(c)
    return c
torch.add = addwrapper
Tensor.__add__ = addwrapper
Tensor.__radd__ = addwrapper
torch_iadd = Tensor.__iadd__
def iaddwrapper(self: Tensor, other: Tensor):
    torch_iadd(self, other)
    if isinstance(self, Tensor) and not torch.is_floating_point(self):
        return self
    if isinstance(other, Tensor) and not torch.is_floating_point(other):
        return self
    # strange in _dynamo
    if not hasattr(self, "interval"):
        return self
    torch_iadd(self.interval.lo, other.interval.lo)
    torch_iadd(self.interval.hi, other.interval.hi)
    self.interval.round()
    return self
Tensor.__iadd__ = iaddwrapper

torch_sub = torch.sub
def subwrapper(a: Tensor, b: Tensor) -> Tensor:
    c = torch_sub(a, b)
    c.interval = interval_sub(a, b)
    c.interval.round()
    check(c)
    return c
torch_isub = Tensor.__isub__
def isubwrapper(a: Tensor, b: Tensor) -> Tensor:
    torch_isub(a, b)
    if isinstance(a, Tensor) and not torch.is_floating_point(a):
        return a
    if isinstance(b, Tensor) and not torch.is_floating_point(b):
        return a
    # strange in _dynamo
    if not hasattr(a, "interval"):
        return a
    # special case for a -= a
    if a is b:
        a.interval = Interval(torch_sub(a.interval.lo, a.interval.lo), torch_sub(a.interval.hi, a.interval.hi))
    else:
        torch_isub(a.interval.lo, b.interval.hi)
        torch_isub(a.interval.hi, b.interval.lo)
        a.interval.round()
    return a
Tensor.__isub__ = isubwrapper

torch_mul = torch.mul
def mulwrapper(a: Tensor, b: Tensor) -> Tensor:
    c = torch_mul(a, b)
    c.interval = interval_mul_div(a, b, torch_mul)
    c.interval.round()
    check(c)
    return c
torch.mul = mulwrapper
Tensor.__mul__ = mulwrapper
Tensor.__rmul__ = mulwrapper

torch_div = torch.div
def divwrapper(a: Tensor, b: Tensor) -> Tensor:
    c = torch_div(a, b)
    if isinstance(a, Tensor) and not torch.is_floating_point(a):
        return c
    c.interval = interval_mul_div(a, b, torch_div)
    c.interval.round()
    check(c)
    return c
torch.div = divwrapper
Tensor.__truediv__ = divwrapper

torch_matmul = torch.matmul
def matmulwrapper(a: Tensor, b: Tensor, out = None) -> Tensor:
    if hasattr(a, "interval"):
        check(a)
    if hasattr(b, "interval"):
        check(b)
    c = torch_matmul(a, b)
    c.interval = interval_matmul(a, b)
    c.interval.round()
    if out is not None:
        out = c
    check(c)
    # print(f'matmul error from [{max_relative_error(a)}, {max_relative_error(b)}] to [{max_relative_error(c)}]')
    return c
torch.matmul = matmulwrapper

torch_relu = torch.relu
torch.relu = tensor_with_interval_round(torch.relu)
torch_gelu = F.gelu
def geluwrapper(a: Tensor) -> Tensor:
    raise NotImplementedError("gelu is not monotonic")
F.gelu = geluwrapper


torch_silu = F.silu
def siluwrapper(a: Tensor) -> Tensor:
    raise NotImplementedError("silu is not monotonic")
    c = torch_silu(a)
    lo, hi = a.interval.lo, a.interval.hi
    lo = torch_silu(lo)
    hi = torch_silu(hi)
    c_lo = torch_minimum(torch_minimum(lo, hi), c)
    c_hi = torch_maximum(torch_maximum(lo, hi), c)
    c.interval = Interval(c_lo, c_hi)
    c.interval.round()
    return c

torch_log = torch.log
torch.log = tensor_with_interval_round(torch_log)
tensor_log = Tensor.log
Tensor.log = tensor_with_interval_round(tensor_log)
torch_exp = torch.exp
torch.exp = tensor_with_interval_round(torch_exp)
torch_pow = torch.pow
torch.pow = tensor_with_interval_round(torch_pow)
torch_mean = torch.mean
torch.mean = tensor_with_interval_round(torch_mean)
torch_sqrt = Tensor.sqrt
torch.sqrt = tensor_with_interval_round(torch_sqrt)
torch_rsqrt = Tensor.rsqrt
torch.rsqrt = tensor_with_interval_round_reverse(torch_rsqrt)
torch_tanh = torch.tanh
torch.tanh = tensor_with_interval_round(torch.tanh)

# TODO: fix this
torch_sin = torch.sin
def sin_wrapper(a: Tensor) -> Tensor:
    raise NotImplementedError("sin is not monotonic")
torch.sin = sin_wrapper
torch_cos = torch.cos
def cos_wrapper(a: Tensor) -> Tensor:
    raise NotImplementedError("cos is not monotonic")
torch.cos = cos_wrapper

torch_linear = F.linear
def linearwrapper(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    if hasattr(a, "interval"):
        check(a)
    if hasattr(b, "interval"):
        check(b)
    d = torch_linear(a, b, c)
    d.interval = interval_linear(a, b, c)
    d.interval.round()
    check(d)
    # print(f'linear error from [{max_relative_error(a)}, {max_relative_error(b)}] to [{max_relative_error(d)}]')
    return d
F.linear = linearwrapper

torch_scaled_dot_product_attention = F.scaled_dot_product_attention
def scaled_dot_product_attentionwrapper(query: Tensor, key: Tensor, value: Tensor, attn_mask: Tensor | None = None, dropout_p: float = 0.0, is_causal: bool = False, scale: float | None = None, enable_gqa: bool = False) -> Tensor:
    raise NotImplementedError("scaled_dot_product_attention is not implemented")
F.scaled_dot_product_attention = scaled_dot_product_attentionwrapper
F.scaled_dot_product_attention.__doc__ = torch_scaled_dot_product_attention.__doc__

torch_softmax = Tensor.softmax
def softmaxwrapper(self: Tensor, dim: int, *args, **kwargs) -> Tensor:
    check(self)
    c = torch_softmax(self, dim, *args, **kwargs)
    c.interval = interval_softmax(self, dim, c)
    c.interval.round()
    # print(f"softmax error from [{max_relative_error(self)}] to [{max_relative_error(c)}]")
    check(c)
    return c
Tensor.softmax = softmaxwrapper

torch_log_softmax = Tensor.log_softmax
def log_softmaxwrapper(self, dim: int, *args, **kwargs) -> Tensor:
    c = torch_log_softmax(self, dim, *args, **kwargs)
    a = self.softmax(dim)
    a = a.log()
    c.interval = a.interval
    del a
    check(c)
    return c
Tensor.log_softmax = log_softmaxwrapper

torch_conv2d = F.conv2d
def conv2dwrapper(input: Tensor, *args, **kwargs) -> Tensor:
    raise NotImplementedError("conv2d is not implemented")
F.conv2d = conv2dwrapper

torch_max_pool2d = F.max_pool2d
def max_pool2dwrapper(input: Tensor, *args, **kwargs) -> Tensor:
    raise NotImplementedError("max_pool2d is not implemented")
F.max_pool2d = max_pool2dwrapper

torch_layer_norm = torch.layer_norm
# TODO: fix me
def layer_normwrapper(input: Tensor, normalized_shape, weight = None, bias = None, eps: float = 1e-5, cudnn_enabled: bool = True) -> Tensor:
    c = torch_layer_norm(input, normalized_shape, weight, bias, eps, cudnn_enabled)
    lo, hi = input.interval.lo, input.interval.hi
    # mean_lo = torch_mean(lo, dim=-1)
    # mean_hi = torch_mean(hi, dim=-1)
    lo = torch_layer_norm(lo, normalized_shape, weight, bias, eps, cudnn_enabled)
    hi = torch_layer_norm(hi, normalized_shape, weight, bias, eps, cudnn_enabled)
    lo, hi = torch_minimum(lo, hi), torch_maximum(lo, hi)
    lo = torch_minimum(lo, c)
    hi = torch_maximum(hi, c)
    c.interval = Interval(lo, hi)
    check(c)
    # print(f"layer_norm error from [{max_relative_error(input)}] to [{max_relative_error(c)}]")
    return c
torch.layer_norm = layer_normwrapper

torch_min = torch.min
def minwrapper(input, *args, **kwargs):
    raise NotImplementedError("minwrapper is not implemented")
torch.min = minwrapper
torch_max = torch.max
def maxwrapper(input, *args, **kwargs):
    raise NotImplementedError("maxwrapper is not implemented")
torch.max = maxwrapper
torch_sum = torch.sum
def sumwrapper(input, *args, **kwargs):
    raise NotImplementedError("sumwrapper is not implemented")
torch.sum = sumwrapper
torch_abs = torch.abs
def abswrapper(input, *args, **kwargs):
    raise NotImplementedError("abswrapper is not implemented")
torch.abs = abswrapper
torch_neg = torch.neg
def negwrapper(input, *args, **kwargs):
    raise NotImplementedError("negwrapper is not implemented")
torch.neg = negwrapper
torch_minimum = torch.minimum
def minimumwrapper(input, *args, **kwargs):
    raise NotImplementedError("minimumwrapper is not implemented")
torch_maximum = torch.maximum
def maximumwrapper(input, *args, **kwargs):
    raise NotImplementedError("maximumwrapper is not implemented")
torch.maximum = maximumwrapper
torch_T = Tensor.T
def twrapper(input, *args, **kwargs):
    raise NotImplementedError("twrapper is not implemented")
Tensor.T = twrapper
torch_mean = Tensor.mean

#################### Loss functions Begin ####################
torch_nll_loss = F.nll_loss
def nll_losswrapper(input: Tensor, target: Tensor, *args, **kwargs) -> Tensor:
    assert not hasattr(target, "interval")
    c = torch_nll_loss(input, target, *args, **kwargs)
    input_lo, input_hi = input.interval.lo, input.interval.hi
    lo = torch_nll_loss(input_lo, target, *args, **kwargs)
    hi = torch_nll_loss(input_hi, target, *args, **kwargs)
    lo, hi = torch_minimum(lo, hi), torch_maximum(lo, hi)
    c.interval = Interval(lo, hi)
    check(c)
    return c
F.nll_loss = nll_losswrapper

torch_cross_entropy = F.cross_entropy
def cross_entropywrapper(input, target, *args, **kwargs):
    """ softmax + nll"""
    assert hasattr(input, "interval")
    assert not hasattr(target, "interval")
    check(input)
    # print('initial cross_entropy', max_relative_error(input))
    c = torch_cross_entropy(input, target, *args, **kwargs)
    prob = input.softmax(dim=-1)
    # print('relative error of softmax', max_relative_error(prob))
    assert hasattr(prob, "interval")
    log_prob = prob.log()
    # print('relative error of log_softmax', max_relative_error(log_prob))
    assert hasattr(log_prob, "interval")
    loss = F.nll_loss(log_prob, target, *args, **kwargs)
    c.interval = loss.interval
    del loss
    check(c)
    print(f'cross_entropy error from [{max_relative_error(input)}] to [{max_relative_error(c)}]')
    return c
F.cross_entropy = cross_entropywrapper
######################## Wrapper functions End ########################

def interval_add(a: Tensor, b: Tensor) -> Interval:
    a_lo, a_hi = lo_hi(a)
    b_lo, b_hi = lo_hi(b)
    lo = torch_add(a_lo, b_lo)
    hi = torch_add(a_hi, b_hi)
    return Interval(lo, hi)

def interval_sub(a: Tensor, b: Tensor) -> Interval:
    a_lo, a_hi = lo_hi(a)
    b_lo, b_hi = lo_hi(b)
    lo = torch_sub(a_lo, b_hi)
    hi = torch_sub(a_hi, b_lo)
    return Interval(lo, hi)

def interval_mul_div(a: Tensor, b: Tensor, f) -> Interval:
    if not hasattr(b, "interval"):
        a_lo, a_hi = lo_hi(a)
        b_lo, b_hi = lo_hi(b)
        lo = f(a_lo, b_lo)
        hi = f(a_hi, b_hi)
        lo, hi = torch_minimum(lo, hi), torch_maximum(lo, hi)
        return Interval(lo, hi)
    else:
        a_lo, a_hi = lo_hi(a)
        b_lo, b_hi = lo_hi(b)
        ll = f(a_lo, b_lo)
        lh = f(a_lo, b_hi)
        hl = f(a_hi, b_lo)
        hh = f(a_hi, b_hi)
        results = torch.stack([ll, lh, hl, hh])
        lo = torch_min(results, dim=0).values
        hi = torch_max(results, dim=0).values
        return Interval(lo, hi)

def Is2Mr(A):
    """
    M=A_lo +0.5∗(A_hi −A_lo);
    R=M−A_lo;
    """
    lo, hi = lo_hi(A)
    assert torch.all(hi >= lo).item(), "hi < lo"
    M = torch_add(lo, torch_mul(torch_sub(hi, lo), 0.5))
    R = torch_sub(M, lo)
    assert torch.all(R >= 0).item(), "R is negative"
    return M, R

def interval_matmul_impl(A, B, f):
    """
    Implementation of the interval matrix multiplication
    using the midpoint-radius representation.
    f is the function to be used for the multiplication,
    it can be torch.matmul or torch.linear.
    
    function [C]=midrad_mul([A],[B])
        [MA,RA]=Is2Mr([A]);
        [MB,RB]=Is2Mr([B]);
        setround(1);
        R=abs(MA)∗RB+RA∗(abs(MB)+RB);
        C_hi=MA∗MB+R;
        setround(−1);
        C_lo=MA∗MB−R;
        [C]=infsup(C,C);
    end
    """
    MA, RA = Is2Mr(A)
    MB, RB = Is2Mr(B)
    R1 = f(torch_abs(MA), RB)
    R2 = f(RA, torch_add(torch_abs(MB), RB))
    R = torch_add(R1, R2) 
    C = f(MA, MB)
    C_hi = torch_add(C, R)
    C_lo = torch_sub(C, R)
    if not torch.all(C_lo <= C_hi).item():
        breakpoint()
    assert torch.all(C_lo <= C_hi).item(), "C_lo > C_hi"
    return C_lo, C_hi

def interval_matmul(a: Tensor, b: Tensor) -> Interval:
    assert hasattr(a, "interval")
    assert hasattr(b, "interval")
    lo, hi = interval_matmul_impl(a, b, torch_matmul)
    return Interval(lo, hi)

def interval_linear(a: Tensor, b: Tensor, d: Tensor | None = None) -> Interval:
    lo, hi = interval_matmul_impl(a, b, torch_linear)
    if d is not None:
        d_lo, d_hi = lo_hi(d)
        lo = torch_add(lo, d_lo)
        hi = torch_add(hi, d_hi)
    return Interval(lo, hi)

def interval_softmax(a: Tensor, dim: int, c: Tensor) -> Interval:
    lo, hi = a.interval.lo, a.interval.hi
    hi_ratio = torch_where(torch.logical_and(hi.isneginf(), lo.isneginf()), 0.0, torch_sub(hi, lo))
    lo_ratio = torch_where(torch.logical_and(lo.isneginf(), hi.isneginf()), 0.0, torch_sub(lo, hi))
    assert_normal(hi_ratio)
    assert_normal(lo_ratio)
    hi_ratio = torch_exp(hi_ratio)
    lo_ratio = torch_exp(lo_ratio)
    lo, hi = torch_mul(c, lo_ratio), torch_mul(c, hi_ratio)
    # hi_ratio may be too large or inf due to xxx
    hi = torch_where(hi > 1.0, 1.0, hi)
    return Interval(lo, hi)

try:
    import cut_cross_entropy
    from cut_cross_entropy import LinearCrossEntropyImpl
    from cut_cross_entropy.linear_cross_entropy import LCE_IMPL_DEFAULT, linear_cross_entropy
    org_forward = linear_cross_entropy
    def linear_cross_entropy(
        e: torch.Tensor,
        c: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: int = -100,
        softcap: float | None = None,
        reduction: str = "mean",
        shift: bool = False,
        filter_eps: float | str | None = "high",
        impl: str | LinearCrossEntropyImpl = LCE_IMPL_DEFAULT,
    ) -> torch.Tensor:
        assert hasattr (e, "interval")
        assert not hasattr (c, "interval")
        raise NotImplementedError("linear_cross_entropy is not implemented")
except ImportError:
    pass
