import torch
from torch import Tensor
from dataclasses import dataclass

torch_to = Tensor.to
def towrapper(self, *args, **kwargs):
    c = torch_to(self, *args, **kwargs)
    if not hasattr(self, "interval"):
        return c
    c.interval = Interval(
        self.interval.lo.torch_to(*args, **kwargs),
        self.interval.hi.torch_to(*args, **kwargs),
    )
    c.round()
    return c

torch_detach = Tensor.detach
def detachwrapper(self, *args, **kwargs):
    c = torch_detach(self, *args, **kwargs)
    if not hasattr(self, "interval"):
        return c
    c.interval = self.interval
    return c
Tensor.detach = detachwrapper
Tensor.torch_detach = torch_detach

torch_where = torch.where
def wherewrapper(condition: Tensor, input: Tensor, other: Tensor, out=None) -> Tensor:
    if out is not None:
        raise NotImplementedError("out is not supported")
    c = torch_where(condition, input, other)
    i_lo, i_hi = lo_hi(input)
    o_lo, o_hi = lo_hi(other)
    lo = torch_where(condition, i_lo, o_lo)
    hi = torch_where(condition, i_hi, o_hi)
    c.interval = Interval(lo, hi)
    return c
torch.where = wherewrapper


@dataclass
class Interval:
    lo: Tensor
    hi: Tensor

    def round(self) -> "Interval":
        lo, hi = self.lo, self.hi
        minus = torch.tensor(torch.finfo(lo.dtype).min, device=lo.device)
        plus = torch.tensor(torch.finfo(hi.dtype).max, device=hi.device)
        lo = torch_where(
            torch.logical_or(lo.isneginf(), lo.isposinf()), lo, lo.nextafter(minus)
        )
        hi = torch_where(
            torch.logical_or(hi.isposinf(), hi.isneginf()), hi, hi.nextafter(plus)
        )
        self.lo, self.hi = lo, hi
        return self

    def round_(self) -> "Interval":
        lo, hi = self.lo, self.hi
        minus = torch.tensor(torch.finfo(lo.dtype).min, device=lo.device)
        plus = torch.tensor(torch.finfo(hi.dtype).max, device=hi.device)
        lo = torch_where(lo.isneginf(), lo, lo.nextafter(minus))
        hi = torch_where(hi.isposinf(), hi, hi.nextafter(plus))
        self.lo, self.hi = lo, hi
        return self

    @staticmethod
    def from_tensor(x: Tensor) -> "Interval":
        x = x.torch_detach()
        return Interval(x.clone(), x.clone()).round_()


def lo_hi(x: Tensor | int | float) -> Tensor:
    if not hasattr(x, "interval"):
        if isinstance(x, Tensor):
            return x.torch_detach().clone(), x.torch_detach().clone()
        return x, x
    return x.interval.lo, x.interval.hi


def init_interval(self: Tensor):
    self.interval = Interval(self.clone().torch_detach(), self.clone().torch_detach())
    self.interval.round_()
Tensor.init_interval = init_interval
