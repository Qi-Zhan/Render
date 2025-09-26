from .interface import Interval


def tensor_with_interval(op):
    """Decorator to add interval support to tensor operations
    by simply doing the same operation on the interval tensor.
    It can be used in non-arithmetic operations (e.g. view, reshape, etc.)
    """

    def wrapper(self, *args, **kwargs):
        c = op(self, *args, **kwargs)
        if not hasattr(self, "interval"):
            return c
        c.interval = Interval(
            op(self.interval.lo, *args, **kwargs),
            op(self.interval.hi, *args, **kwargs),
        )
        # if c is not self:
        #     del self.interval
        # torch.cuda.empty_cache
        return c

    return wrapper


def tensor_with_interval_round(op):
    """Decorator to add interval support to tensor operations
    by simply doing the same operation on the interval tensor and **round**.
    It can be used in both monotonic and elementwise operations (e.g. exp, log, etc.)
    """

    def wrapper(self, *args, **kwargs):
        c = op(self, *args, **kwargs)
        if not hasattr(self, "interval"):
            return c
        c.interval = Interval(
            op(self.interval.lo, *args, **kwargs),
            op(self.interval.hi, *args, **kwargs),
        )
        c.interval.round()
        return c

    return wrapper


def tensor_with_interval_round_reverse(op):
    """Decorator to add interval support to tensor operations
    by simply doing the same operation on the interval tensor and **round**.
    It can be used in both monotonic and elementwise operations (e.g. exp, log, etc.)
    """

    def wrapper(self, *args, **kwargs):
        c = op(self, *args, **kwargs)
        if not hasattr(self, "interval"):
            return c
        c.interval = Interval(
            op(self.interval.hi, *args, **kwargs),
            op(self.interval.lo, *args, **kwargs),
        )
        c.interval.round()
        return c

    return wrapper
