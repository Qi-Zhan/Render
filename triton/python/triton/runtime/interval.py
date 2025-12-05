import math
import numpy as np
import numba

eps_64 = np.finfo(np.float64).eps
eps_32 = np.finfo(np.float32).eps


def exp(x):
    if isinstance(x, Interval):
        return x.exp()
    return math.exp(x)


def exp2(x):
    if isinstance(x, Interval):
        return x.exp2()
    return math.exp2(x)


def log(x):
    if isinstance(x, Interval):
        return x.log()
    return math.log(x)


def log2(x):
    if isinstance(x, Interval):
        return x.log2()
    return math.log2(x)


def sqrt(x):
    if isinstance(x, Interval):
        return x.sqrt()
    return math.sqrt(x)


def sin(x):
    if isinstance(x, Interval):
        return x.sin()
    return math.sin(x)


def cos(x):
    if isinstance(x, Interval):
        return x.cos()
    return math.cos(x)


def up(v, eps):
    if v >= 0:
        return v * (1 + eps)
    else:
        return v * (1 - eps)


def down(v, eps):
    if v >= 0:
        return v * (1 - eps)
    else:
        return v * (1 + eps)


def down_up(left, right, eps):
    return down(left, eps), up(right, eps)


def cast_eps(interval, eps):
    """Casts the epsilon of the interval to the given value
    - If eps is smaller than the current epsilon (precison, lost), then we need to down_up the interval
    - Else, we just need to update the epsilon
    """
    if isinstance(interval, Interval):
        return interval.cast_eps(eps)
    return interval


class Interval:
    def __init__(self, left, right, eps):
        assert isinstance(left, (int, float))
        assert isinstance(right, (int, float))
        assert isinstance(eps, (int, float))
        assert left <= right
        self.left = left
        self.right = right
        self.eps = eps

    def from_float(f, eps):
        eps = float(eps)
        f = float(f)
        # we assume the input is precise, not from real casting
        left, right = down_up(f, f, 0)
        return Interval(left, right, eps)

    def from_constant(c):
        c = float(c)
        return Interval(c, c, 0)

    def __repr__(self):
        return f"[{self.left}, {self.right}]"

    def cast_eps(self, eps) -> "Interval":
        left, right = self.left, self.right
        if eps > self.eps:  # cast to lower precision
            # round to the nearest value or chunk
            # left, right = down_up(left, right, 1 / 2 * eps)
            left, right = down_up(left, right, eps)
        return Interval(left, right, eps)

    def __add__(self, other):
        if not isinstance(other, Interval):
            other = Interval.from_constant(other)
        left = self.left + other.left
        right = self.right + other.right
        eps = max(self.eps, other.eps)
        left, right = down_up(left, right, eps)
        return Interval(left, right, eps)

    def __radd__(self, other):
        if not isinstance(other, Interval):
            other = Interval.from_constant(other)
        return other + self

    def __sub__(self, other):
        if not isinstance(other, Interval):
            other = Interval.from_constant(other)
        left = self.left - other.right
        right = self.right - other.left
        eps = max(self.eps, other.eps)
        left, right = down_up(left, right, eps)
        return Interval(left, right, eps)

    def __rsub__(self, other):
        if not isinstance(other, Interval):
            other = Interval.from_constant(other)
        return other - self

    def __mul__(self, other):
        if not isinstance(other, Interval):
            other = Interval.from_constant(other)
        left = min(
            self.left * other.left,
            self.left * other.right,
            self.right * other.left,
            self.right * other.right,
        )
        right = max(
            self.left * other.left,
            self.left * other.right,
            self.right * other.left,
            self.right * other.right,
        )
        eps = max(self.eps, other.eps)
        left, right = down_up(left, right, eps)
        return Interval(left, right, eps)

    def __rmul__(self, other):
        if not isinstance(other, Interval):
            other = Interval.from_constant(other)
        return other * self

    def __contains__(self, other):
        if isinstance(other, Interval):
            return self.left <= other.left and other.right <= self.right
        return self.left <= other <= self.right

    def __truediv__(self, other):
        if not isinstance(other, Interval):
            other = Interval.from_constant(other)
        if 0 in other:
            raise ZeroDivisionError(f"Division by zero in {other}")
        interval = self * Interval(1 / other.right, 1 / other.left, eps=0)
        left, right = down_up(interval.left, interval.right, interval.eps)
        return Interval(left, right, interval.eps)

    def __rtruediv__(self, other):
        if not isinstance(other, Interval):
            other = Interval.from_constant(other)
        return other / self

    def __mod__(self, other):
        raise NotImplementedError("Modulo not implemented for intervals")

    def __rmod__(self, other):
        raise NotImplementedError("Modulo not implemented for intervals")

    def exp(self):
        left = math.exp(self.left)
        right = math.exp(self.right)
        eps = self.eps
        left, right = down_up(left, right, 2 * eps)
        return Interval(left, right, eps)

    def exp2(self):
        left = math.exp2(self.left)
        right = math.exp2(self.right)
        eps = self.eps
        left, right = down_up(left, right, 2 * eps)
        return Interval(left, right, eps)

    def log(self):
        if self.left <= 0:
            raise ValueError(f"Logarithm of non-positive number in {self}")
        left = math.log(self.left)
        right = math.log(self.right)
        eps = self.eps
        left, right = down_up(left, right, eps)
        return Interval(left, right, eps)

    def log2(self):
        if self.left <= 0:
            raise ValueError(f"Logarithm of non-positive number in {self}")
        left = math.log2(self.left)
        right = math.log2(self.right)
        eps = self.eps
        left, right = down_up(left, right, eps)
        return Interval(left, right, eps)

    def sqrt(self):
        if self.left < 0:
            raise ValueError(f"Square root of negative number in {self}")
        left = math.sqrt(self.left)
        right = math.sqrt(self.right)
        eps = self.eps
        left, right = down_up(left, right, eps)
        return Interval(left, right, eps)

    def __le__(self, other):
        """for triton overflow detection"""
        return True

    def __ge__(self, other):
        """for triton underflow detection"""
        return True

make_interval = np.frompyfunc(lambda x, eps: Interval.from_float(x, eps), 2, 1)

def adjust_down(v, eps):
    return np.where(v >= 0, v * (1 - eps), v * (1 + eps))


def adjust_up(v, eps):
    return np.where(v >= 0, v * (1 + eps), v * (1 - eps))


def adjust_bounds(lo, hi, eps):
    return adjust_down(lo, eps), adjust_up(hi, eps)


class IntervalArray:
    def __init__(self, lo, hi, eps):
        self.lo = lo
        self.hi = hi
        self.eps = float(eps)

    @staticmethod
    def from_float(f, eps):
        f = np.asarray(f, dtype=np.float64)
        return IntervalArray(f, f, eps)

    @staticmethod
    def from_constant(mat):
        if not isinstance(mat, np.ndarray):
            mat = np.array(mat, dtype=np.float64)
        mat = np.asarray(mat, dtype=np.float64)
        return IntervalArray(mat, mat, 0)

    def __contains__(self, other):
        if isinstance(other, IntervalArray):
            return np.all(self.lo <= other.lo) and np.all(other.hi <= self.hi)
        return np.all(self.lo <= other) and np.all(other <= self.hi)

    def __add__(self, other):
        eps = max(self.eps, other.eps)
        lo_a, lo_b, hi_a, hi_b = self.lo, other.lo, self.hi, other.hi
        lo = lo_a + lo_b
        hi = hi_a + hi_b
        lo_adj, hi_adj = adjust_bounds(lo, hi, eps)
        return IntervalArray(lo_adj, hi_adj, max(self.eps, other.eps))

    def __radd__(self, other):
        if not isinstance(other, IntervalArray):
            other = IntervalArray.from_constant(other)
        return other + self

    def __sub__(self, other):
        eps = max(self.eps, other.eps)
        lo_a, lo_b, hi_a, hi_b = self.lo, other.lo, self.hi, other.hi
        lo = lo_a - hi_b
        hi = hi_a - lo_b
        lo_adj, hi_adj = adjust_bounds(lo, hi, eps)
        return IntervalArray(lo_adj, hi_adj, max(self.eps, other.eps))

    def __rsub__(self, other):
        if not isinstance(other, IntervalArray):
            other = IntervalArray.from_constant(other)
        return other - self

    def __mul__(self, other):
        eps = max(self.eps, other.eps)
        ll = self.lo * other.lo
        lh = self.lo * other.hi
        hl = self.hi * other.lo
        hh = self.hi * other.hi
        lo = np.minimum.reduce([ll, lh, hl, hh])
        hi = np.maximum.reduce([ll, lh, hl, hh])
        lo, hi = adjust_bounds(lo, hi, eps)
        return IntervalArray(lo, hi, eps)

    def __rmul__(self, other):
        if not isinstance(other, IntervalArray):
            other = IntervalArray.from_constant(other, 0)
        return other * self

    def __truediv__(self, other):
        if np.any(other.lo <= 0) and np.any(other.hi >= 0):
            raise ZeroDivisionError("Interval contains zero in division")
        inv_lo = 1.0 / other.hi
        inv_hi = 1.0 / other.lo
        inv = IntervalArray(inv_lo, inv_hi, other.eps)
        return self * inv

    def select(self, other, condition):
        lo = np.where(condition, self.lo, other.lo)
        hi = np.where(condition, self.hi, other.hi)
        eps = max(self.eps, other.eps)
        return IntervalArray(lo, hi, eps)

    def cast_eps(self, new_eps):
        if new_eps > self.eps:
            lo, hi = adjust_bounds(self.lo, self.hi, new_eps)
            return IntervalArray(lo, hi, new_eps)
        return IntervalArray(self.lo, self.hi, new_eps)

    def exp(self):
        lo = np.exp(self.lo)
        hi = np.exp(self.hi)
        lo_adj, hi_adj = adjust_bounds(lo, hi, 2 * self.eps)
        return IntervalArray(lo_adj, hi_adj, self.eps)

    def exp2(self):
        lo = np.exp2(self.lo)
        hi = np.exp2(self.hi)
        lo_adj, hi_adj = adjust_bounds(lo, hi, 2 * self.eps)
        return IntervalArray(lo_adj, hi_adj, self.eps)

    def log(self):
        if np.any(self.lo <= 0):
            raise ValueError("Logarithm of non-positive interval")
        lo = np.log(self.lo)
        hi = np.log(self.hi)
        lo_adj, hi_adj = adjust_bounds(lo, hi, 2 * self.eps)
        return IntervalArray(lo_adj, hi_adj, self.eps)

    def log2(self):
        if np.any(self.lo <= 0):
            raise ValueError("Logarithm of non-positive interval")
        lo = np.log2(self.lo)
        hi = np.log2(self.hi)
        lo_adj, hi_adj = adjust_bounds(lo, hi, 2 * self.eps)
        return IntervalArray(lo_adj, hi_adj, self.eps)

    def sqrt(self):
        if np.any(self.lo < 0):
            raise ValueError("Square root of negative interval")
        lo = np.sqrt(self.lo)
        hi = np.sqrt(self.hi)
        lo_adj, hi_adj = adjust_bounds(lo, hi, self.eps)
        return IntervalArray(lo_adj, hi_adj, self.eps)
    
    def matmul_without_mixed_precision(self, b: "IntervalArray", d, mul_eps, add_eps, d_org):
        """self @ b + d"""
        eps = max(self.eps, b.eps)
        a_low, a_high = self.lo, self.hi
        b_low, b_high = b.lo, b.hi
        c_low, c_high = _matmul(a_low, a_high, b_low, b_high, eps, eps)
        c = IntervalArray(c_low, c_high, eps)
        if np.all(d_org != 0):
            c = c + d
        return c


    def matmul(self, b: "IntervalArray", d, mul_eps, add_eps, d_org):
        """self @ b + d"""
        self_cast = self.cast_eps(mul_eps)
        b_cast = b.cast_eps(mul_eps)
        a_low, a_high = self_cast.lo, self_cast.hi
        b_low, b_high = b_cast.lo, b_cast.hi
        c_low, c_high = _matmul(a_low, a_high, b_low, b_high, mul_eps, add_eps)
        c = IntervalArray(c_low, c_high, add_eps)
        if np.all(d_org != 0):
            d = d.cast_eps(add_eps)
            c = c + d
        return c

    def transpose(self, perm):
        return IntervalArray(self.lo.transpose(perm), self.hi.transpose(perm), self.eps)

    def expand_dims(self, axis):
        return IntervalArray(
            np.expand_dims(self.lo, axis), np.expand_dims(self.hi, axis), self.eps
        )

    def broadcast_to(self, shape):
        return IntervalArray(
            np.broadcast_to(self.lo, shape), np.broadcast_to(self.hi, shape), self.eps
        )

    def sin(self):
        lo = np.sin(self.lo)
        hi = np.sin(self.hi)
        lo_adj, hi_adj = adjust_bounds(lo, hi, 2 * self.eps)
        return IntervalArray(lo_adj, hi_adj, self.eps)

    def cos(self):
        lo = np.cos(self.lo)
        hi = np.cos(self.hi)
        lo_adj, hi_adj = adjust_bounds(lo, hi, 2 * self.eps)
        return IntervalArray(lo_adj, hi_adj, self.eps)

    @property
    def shape(self):
        return self.lo.shape

    def argmax(self, axis, keepdims):
        lo = np.argmax(self.lo, axis=axis, keepdims=keepdims)
        hi = np.argmax(self.hi, axis=axis, keepdims=keepdims)
        return IntervalArray(lo, hi, self.eps)

    def take_along_axis(self, indices, axis):
        lo = np.take_along_axis(self.lo, indices, axis=axis)
        hi = np.take_along_axis(self.hi, indices, axis=axis)
        return IntervalArray(lo, hi, self.eps)

    def sum(self, axis, keepdims):
        # sum is operated on float32
        add_eps = (self.lo.shape[axis] - 1) * eps_32
        lo = np.sum(self.lo, axis=axis, keepdims=keepdims)
        hi = np.sum(self.hi, axis=axis, keepdims=keepdims)
        lo = np.atleast_1d(lo)
        hi = np.atleast_1d(hi)

        lo_abs = np.sum(np.abs(self.lo), axis=axis, keepdims=keepdims)
        hi_abs = np.sum(np.abs(self.hi), axis=axis, keepdims=keepdims)
        lo_abs = np.atleast_1d(lo_abs)
        hi_abs = np.atleast_1d(hi_abs)

        lo -= lo_abs * add_eps
        hi += hi_abs * add_eps
        # cast back to original precision 
        if self.eps > eps_32:
            lo, hi = adjust_bounds(lo, hi, self.eps)

        return IntervalArray(lo, hi, self.eps)

    def squeeze(self, axis):
        lo = np.squeeze(self.lo, axis=axis)
        hi = np.squeeze(self.hi, axis=axis)
        return IntervalArray(lo, hi, self.eps)

    def cumprod(self, reduce):
        lo = np.cumprod(self.lo, axis=reduce.axis)
        hi = np.cumprod(self.hi, axis=reduce.axis)
        return IntervalArray(lo, hi, self.eps)

    def cumsum(self, axis):
        lo = np.cumsum(self.lo, axis=axis)
        hi = np.cumsum(self.hi, axis=axis)
        return IntervalArray(lo, hi, self.eps)

    def fmod(self, other):
        lo = np.fmod(self.lo, other.lo)
        hi = np.fmod(self.hi, other.hi)
        return IntervalArray(lo, hi, self.eps)

    def index(self, mask):
        return IntervalArray(self.lo[mask], self.hi[mask], self.eps)

    def __str__(self):
        return f"IntervalArray({self.lo}, {self.hi}, {self.eps})"

    ########## We do not need these, just for triton compatibility ##########

    def __and__(self, other):
        return self

    def __rand__(self, other):
        return self

    def __or__(self, other):
        return self

    def __ror__(self, other):
        return self

    def __gt__(self, other):
        return self

    def __lt__(self, other):
        return self

    def __le__(self, other):
        return self

    def __ge__(self, other):
        return self


@numba.njit()
def _matmul(
    a_low: np.ndarray,
    a_high: np.ndarray,
    b_low: np.ndarray,
    b_high: np.ndarray,
    mul_eps,
    add_eps,
):
    n, m = a_low.shape
    m, k = b_low.shape
    c_low = np.zeros((n, k))
    c_high = np.zeros((n, k))
    add_eps = (m - 1) * add_eps
    for i in range(n):
        for j in range(k):
            a_low_i = a_low[i, :]
            a_high_i = a_high[i, :]
            b_low_j = b_low[:, j]
            b_high_j = b_high[:, j]

            a_low_b_low = np.multiply(a_low_i, b_low_j)
            a_low_b_high = np.multiply(a_low_i, b_high_j)
            a_high_b_low = np.multiply(a_high_i, b_low_j)
            a_high_b_high = np.multiply(a_high_i, b_high_j)
            # minimum of four values
            lo = np.minimum(a_low_b_low, a_low_b_high)
            lo = np.minimum(lo, a_high_b_low)
            lo = np.minimum(lo, a_high_b_high)
            # maximum of four values
            hi = np.maximum(a_low_b_low, a_low_b_high)
            hi = np.maximum(hi, a_high_b_low)
            hi = np.maximum(hi, a_high_b_high)

            lo = np.where(lo >= 0, lo * (1 - mul_eps), lo * (1 + mul_eps))
            hi = np.where(hi >= 0, hi * (1 + mul_eps), hi * (1 - mul_eps))
            ### \Delta <= \sum|a| * \epsilon
            lo_abs = np.sum(np.abs(lo))
            hi_abs = np.sum(np.abs(hi))
            lo = np.sum(lo)
            hi = np.sum(hi)
            lo -= lo_abs * add_eps
            hi += hi_abs * add_eps

            c_low[i, j] = lo
            c_high[i, j] = hi
    return c_low, c_high
