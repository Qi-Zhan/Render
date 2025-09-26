import torch
import torch.nn.functional as F
from torch import Tensor

from .interval import Interval, torch_where
from .wrapper import tensor_with_interval


torch_view = Tensor.view
Tensor.view = tensor_with_interval(Tensor.view)
torch_reshape = Tensor.reshape
Tensor.reshape = tensor_with_interval(Tensor.reshape)
torch_unsqueeze = Tensor.unsqueeze
Tensor.unsqueeze = tensor_with_interval(Tensor.unsqueeze)
torch_squeeze = Tensor.squeeze
Tensor.squeeze = tensor_with_interval(Tensor.squeeze)
torch_expand = Tensor.expand
Tensor.expand = tensor_with_interval(Tensor.expand)
torch_expand_as = Tensor.expand_as
Tensor.expand_as = tensor_with_interval(Tensor.expand_as)
torch_repeat = Tensor.repeat
Tensor.repeat = tensor_with_interval(Tensor.repeat)
torch_repeat_interleave = Tensor.repeat_interleave
Tensor.repeat_interleave = tensor_with_interval(Tensor.repeat_interleave)
torch_cat = torch.cat
torch.cat = tensor_with_interval(torch.cat)
torch_tranpose = Tensor.transpose
Tensor.transpose = tensor_with_interval(Tensor.transpose)
torch_permute = Tensor.permute
Tensor.permute = tensor_with_interval(Tensor.permute)
torch_get_item = Tensor.__getitem__
Tensor.__getitem__ = tensor_with_interval(Tensor.__getitem__)
Tensor.torch_get_item = torch_get_item
torch_flatten = torch.flatten
torch.flatten = tensor_with_interval(torch.flatten)
torch_contiguous = Tensor.contiguous
Tensor.contiguous = tensor_with_interval(Tensor.contiguous)
torch_masked_fill = Tensor.masked_fill
Tensor.masked_fill = tensor_with_interval(Tensor.masked_fill)

torch_split = Tensor.split
def splitwrapper(self, *args, **kwargs):
    if not hasattr(self, "interval"):
        return torch_split(self, *args, **kwargs)
    results = torch_split(self, *args, **kwargs)
    lo = torch_split(self.interval.lo, *args, **kwargs)
    hi = torch_split(self.interval.hi, *args, **kwargs)
    for i in range(len(results)):
        results[i].interval = Interval(lo[i], hi[i])
    return results
Tensor.split = splitwrapper

torch_embedding = torch.embedding
def embeddingwrapper(input: Tensor, *args, **kwargs) -> Tensor:
    c = torch_embedding(input, *args, **kwargs)
    c.interval = Interval.from_tensor(c)
    return c
torch.embedding = embeddingwrapper


torch_dropout = F.dropout
def dropoutwrapper(input: Tensor, *args, **kwargs) -> Tensor:
    c = torch_dropout(input, *args, **kwargs)
    lo = input.interval.lo
    hi = input.interval.hi
    lo = torch_where(c == 0, 0, lo)
    hi = torch_where(c == 0, 0, hi)
    c.interval = Interval(lo, hi)
    return c
F.dropout = dropoutwrapper