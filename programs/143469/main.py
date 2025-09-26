import torch
from torch import nn
torch.manual_seed(0)

NUM_INPUT=50
INPUT_SIZE=500
NUM_LINEAR=2
DEVICE="cuda"

class SimpleModel(nn.Module):
    def __init__(self, device=DEVICE):
        super().__init__()
        self.device = device
        self.weights = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(NUM_INPUT, INPUT_SIZE//NUM_LINEAR)) for _ in range(NUM_LINEAR)]
        ).to(self.device)
        self.biases = torch.nn.ParameterList(
            [torch.randn(NUM_INPUT).to(self.device) for _ in range(NUM_LINEAR)]
        ).to(self.device)

    def to_float(self):
        for layer in [self.weights, self.biases]:
            layer = layer.cpu().float().to(self.device)

    def to_double(self):
        for layer in [self.weights, self.biases]:
            layer = layer.cpu().double().to(self.device)

    def forward(self, x):
        l1_out = torch.split(x.to(self.device), INPUT_SIZE//NUM_LINEAR, dim=1)
        l1_linear = []
        for i in range(len(l1_out)):
            l1_linear.append(
                torch.nn.functional.linear(
                    l1_out[i], self.weights[i], self.biases[i])
            )
        l1_out = torch.cat(l1_linear, dim=1)
        return l1_out

arg = torch.randn(NUM_INPUT, INPUT_SIZE, device=DEVICE)
arg = arg + torch.randn(NUM_INPUT, INPUT_SIZE, device=DEVICE) + torch.tensor(100, dtype=torch.float32, device=DEVICE)
low_input = arg.to(torch.float32)
model = SimpleModel()
fp32_origin = model(low_input)

optimized_model = torch.compile(model).to(DEVICE)
optimized_model.to_float()
fp32_compiled = optimized_model(low_input)

high_input = arg.to(torch.float64)
model.to_double()
fp64_ref = model(high_input)
print("Eager divergence", torch.max(torch.abs(fp32_origin - fp64_ref)))
print("Compile divergence", torch.max(torch.abs(fp32_compiled - fp64_ref)))
