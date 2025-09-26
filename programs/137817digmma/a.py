from dataclasses import dataclass

epsilon = 9.77e-04


@dataclass
class Interval:
    # we need to maintaine
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


def polevl(x, A):
    result = 0
    for i in range(len(A)):
        result = result * x + A[i]
    return result


def calc_digamma(x):
    x_is_integer = x == math.trunc(x)
    if x < 0:
        if x_is_integer:
            return math.nan
        _, r = math.modf(x)
        pi_over_tan_pi_x = math.pi / math.tan(math.pi * r)
        return calc_digamma(1 - x) - pi_over_tan_pi_x
    result = 0
    while x < 10:
        result = result - 1 / x
        x = x + 1
    if x == 10:
        return result + 2.25175258906672110764
    # Compute asymptotic digamma
    A = [
        8.33333333333333333333e-2,
        -2.10927960927960927961e-2,
        7.57575757575757575758e-3,
        -4.16666666666666666667e-3,
        3.96825396825396825397e-3,
        -8.33333333333333333333e-3,
        8.33333333333333333333e-2,
    ]

    y = 0
    if x < 1.0e17:
        z = 1 / (x * x)
        y = z * polevl(z, A)
    return result + log(x) - (0.5 / x) - y


# Example usage:
x = 0.00615
print("Digamma(", x, ") =", calc_digamma(x))

interval = Interval(x)
print(f"Digamma({x}) = {calc_digamma(interval)}")
