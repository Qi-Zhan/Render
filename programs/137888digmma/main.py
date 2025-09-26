import numpy as np
import torch
def test_digamma():
    input_data = np.array([0.10000000149011612], dtype=np.float32)
    pytorch_result = torch.digamma(torch.tensor(input_data, dtype=torch.float32)).numpy().astype(np.float32)
    print(f"PyTorch digamma result: {pytorch_result}")
    pytorch_result_64 = torch.digamma(torch.tensor(input_data, dtype=torch.float32).double()).numpy().astype(np.float32)
    print(f"PyTorch digamma64 result: {pytorch_result_64}")
    from scipy.special import psi
    psi_result = psi(input_data).astype(np.float32)
    print(f"Scipy digamma result: {psi_result}")

if __name__ == "__main__":
    test_digamma()