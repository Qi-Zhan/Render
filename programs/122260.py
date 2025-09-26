# OK
import torch
torch.set_default_device('cuda')

scale = torch.tensor(0.180336877703666687)
x = torch.tensor(1134139801600.000000)

def f(x, scale):
    max_scaled = x * scale
    return torch.exp(max_scaled - x * scale)

print(f(x, scale))
print(torch.compile(f)(x, scale))