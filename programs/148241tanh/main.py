from dataclasses import dataclass

epsilon = 1.19e-07


@dataclass
class Interval:
    value: float
    lower: float
    upper: float

    def __init__(self, value, lower=None, upper=None):
        self.value = value
        self.lower = lower if lower is not None else value
        self.upper = upper if upper is not None else value

    def up_down(self, n):
        value = self.value
        for _ in range(n):
            lower = (
                self.lower * (1 - epsilon)
                if self.lower > 0
                else self.lower * (1 + epsilon)
            )
            upper = (
                self.upper * (1 + epsilon)
                if self.upper > 0
                else self.upper * (1 - epsilon)
            )
        return Interval(value, lower, upper)

    def __add__(self, other):
        if not isinstance(other, Interval):
            other = Interval(other)
        return Interval(
            self.value + other.value, self.lower + other.lower, self.upper + other.upper
        ).up_down(1)

    def __radd__(self, other):
        if not isinstance(other, Interval):
            other = Interval(other)
        return self + other

    def __sub__(self, other):
        if not isinstance(other, Interval):
            other = Interval(other)
        return Interval(
            self.value - other.value, self.lower - other.upper, self.upper - other.lower
        ).up_down(1)

    def __rsub__(self, other):
        if not isinstance(other, Interval):
            other = Interval(other)
        return other - self

    def __truediv__(self, other):
        if not isinstance(other, Interval):
            other = Interval(other)
        if other.lower <= 0 <= other.upper:
            raise ValueError("Cannot divide by an interval that includes zero.")
        lower = min(
            self.lower / other.lower,
            self.lower / other.upper,
            self.upper / other.lower,
            self.upper / other.upper,
        )
        upper = max(
            self.lower / other.lower,
            self.lower / other.upper,
            self.upper / other.lower,
            self.upper / other.upper,
        )
        return Interval(self.value / other.value, lower, upper).up_down(2)

    def __rtruediv__(self, other):
        if not isinstance(other, Interval):
            other = Interval(other)
        return other / self

    def __mul__(self, other):
        if not isinstance(other, Interval):
            other = Interval(other)
        lower = min(
            self.lower * other.lower,
            self.lower * other.upper,
            self.upper * other.lower,
            self.upper * other.upper,
        )
        upper = max(
            self.lower * other.lower,
            self.lower * other.upper,
            self.upper * other.lower,
            self.upper * other.upper,
        )
        return Interval(self.value * other.value, lower, upper).up_down(1)

    def __rmul__(self, other):
        if not isinstance(other, Interval):
            other = Interval(other)
        return self * other

    def __trunc__(self):
        return self.value.__trunc__()

    def __lt__(self, other):
        if not isinstance(other, Interval):
            return self.value < other
        return self.value < other.value


import math


def log(x):
    if isinstance(x, Interval):
        return Interval(
            math.log(x.value), math.log(x.lower), math.log(x.upper)
        ).up_down(2)
    return math.log(x)


def exp(x):
    if isinstance(x, Interval):
        return Interval(
            math.exp(x.value), math.exp(x.lower), math.exp(x.upper)
        ).up_down(1)
    return math.exp(x)


import torch
import torch.nn as nn
import numpy as np
from torch._inductor import config

config.fallback_random = True
torch.set_grad_enabled(False)
torch.manual_seed(0)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.shrink = nn.Tanhshrink()

    def forward(self, x):
        x = self.shrink(x)
        # x = torch.atan2(x, x)
        return x


model = Model()


x = torch.randn(1, 3, 64, 64)

inputs = [x]


def run_test(model, inputs, backend):
    if backend != "eager":
        model = torch.compile(model, backend=backend)
    torch.manual_seed(0)
    output = model(*inputs)
    return output


output = run_test(model, inputs, "eager")
c_output = run_test(model, inputs, "inductor")

torch.testing.assert_close(output, c_output)
print(output)
print(c_output)
print(torch.max(torch.abs(output - c_output)))
# print the index of the maximum difference
print(torch.argmax(torch.abs(output - c_output)))

# extern "C" void kernel(const float* in_ptr0,
#                        float* out_ptr0)
# {
#     {
#         for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(16L))
#         {
#             auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0), 16);
#             auto tmp1 = decltype(tmp0)(2) / (decltype(tmp0)(1) + (decltype(tmp0)(-2) * tmp0).exp()) - decltype(tmp0)(1);
#             auto tmp2 = tmp0 - tmp1;
#             auto tmp3 = tmp2.atan2(tmp2);
#             tmp3.store(out_ptr0 + static_cast<long>(x0));
#         }
#     }
# }

numpy_x = x.numpy().flatten().tolist()
numpy_x = [Interval(value) for value in numpy_x]
numpy_tanh = [2 / (1 + exp(-2 * x)) - 1 for x in numpy_x]
numpy_y = [x - tanh for x, tanh in zip(numpy_x, numpy_tanh)]
numpy_y = np.array(numpy_y).reshape(1, 3, 64, 64)

for value, interval in zip(output.flatten().tolist(), numpy_y.flatten().tolist()):
    assert (
        interval.lower <= value <= interval.upper
    ), f"Value {value} is not in the interval {interval}"

for value, interval in zip(c_output.flatten().tolist(), numpy_y.flatten().tolist()):
    assert (
        interval.lower <= value <= interval.upper
    ), f"Value {value} is not in the interval {interval}"

np.set_printoptions(precision=10)
torch.set_printoptions(precision=10)

print(numpy_y[0][0][11][47])
print(output[0][0][11][47])
print(c_output[0][0][11][47])
