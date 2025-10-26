import torch
from PIL import Image

from xai_proj_b.data.transforms import AugmentationParams, build_train_transforms


def test_transforms_deterministic(tmp_path):
    aug = AugmentationParams(name="test", random_crop=False, random_flip=False, color_jitter=None)
    transform = build_train_transforms("cifar10", 32, aug)
    img = Image.new("RGB", (32, 32), color=(123, 50, 200))
    torch.manual_seed(0)
    out1 = transform(img)
    torch.manual_seed(0)
    out2 = transform(img)
    assert torch.allclose(out1, out2)
