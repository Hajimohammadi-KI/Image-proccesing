import torch

from xai_proj_b.utils.seed import set_seed


def test_reproducibility_helper():
    set_seed(42, deterministic=True)
    a = torch.randn(3)
    set_seed(42, deterministic=True)
    b = torch.randn(3)
    assert torch.allclose(a, b)
