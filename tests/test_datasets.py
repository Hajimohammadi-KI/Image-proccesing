import shutil
from pathlib import Path

from xai_proj_b.data.datasets import _validate_own_dataset, OWN_DATASET_CLASSES


def _create_fake_image(path: Path):
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (10, 10), color=(255, 0, 0)).save(path)


def test_validate_own_dataset_pass(tmp_path):
    class_dir = tmp_path / OWN_DATASET_CLASSES[0]
    img_path = class_dir / "student1_phone1_coffee-mug_0001.jpg"
    _create_fake_image(img_path)
    _validate_own_dataset(tmp_path)


def test_validate_own_dataset_fail(tmp_path):
    class_dir = tmp_path / OWN_DATASET_CLASSES[0]
    img_path = class_dir / "badname.jpg"
    _create_fake_image(img_path)
    try:
        _validate_own_dataset(tmp_path)
    except ValueError:
        return
    assert False, "Expected ValueError for invalid filename"
