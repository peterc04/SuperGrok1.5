import pytest
import torch


@pytest.fixture(autouse=True)
def _deterministic():
    torch.manual_seed(0)
    torch.set_num_threads(1)
    yield


def tiny_mlp(seed=0):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.GELU(), torch.nn.Linear(16, 4))


def batch(seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(32, 8, generator=g), torch.randint(0, 4, (32,), generator=g)


def max_diff(m1, m2):
    return max((a - b).abs().max().item() for a, b in zip(m1.parameters(), m2.parameters()))
