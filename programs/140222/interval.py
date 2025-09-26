import torch
import torch.nn.functional as F

input = torch.tensor(
    [
        [
            [
                [[-1.6641, -0.9219], [2.5156, -0.4648]],
                [[1.0469, 1.4531], [0.0908, 0.1514]],
            ],
            [
                [[-1.2422, 1.2656], [-1.0859, -0.0801]],
                [[2.3750, -0.4238], [-0.9023, -0.4570]],
            ],
        ],
        [
            [
                [[-1.3125, 0.1099], [-0.2129, -0.6641]],
                [[-0.3867, -0.2617], [1.1641, 0.4043]],
            ],
            [
                [[-0.5352 - 0.5352, -0.4785], [0.2061, 0.2734]],
                [[-1.3672, 0.0544], [-1.3203, -1.5469]],
            ],
        ],
    ],
    dtype=torch.float16,
)
result_cpu = F.log_softmax(input, -1)
input_cuda = input.cuda()
result_cuda = F.log_softmax(input_cuda, -1)
print("result_cpu", result_cpu)
print("result_cuda", result_cuda.cpu())
assert torch.allclose(result_cpu, result_cuda.cpu(), atol=1e-2, rtol=1e-3)


from tensor_interval import *

input_cuda = interval_init_inputs(input_cuda)
result_interval = F.log_softmax(input_cuda, -1)
lower, upper = result_interval.interval.lo, result_interval.interval.hi
for lo, hi in zip(lower.flatten(), upper.flatten()):
    print(f"lo: {lo}, hi: {hi}")
assert torch.all(lower <= result_cuda)
assert torch.all(result_cuda <= upper)

assert torch.all(lower <= result_cpu.cuda())
assert torch.all(result_cpu.cuda() <= upper)

print("In the interval, the result is correct.")
