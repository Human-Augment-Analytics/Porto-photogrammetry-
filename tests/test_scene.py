"""Scene path contract, validation guards, and the COLMAP mask-symlink trick."""
import pytest

from augenblick.core.errors import SceneError
from augenblick.core.scene import Scene


def test_path_properties(tmp_path):
    scene = Scene(tmp_path)
    assert scene.images_dir == tmp_path / "images"
    assert scene.masks_dir == tmp_path / "masks"
    assert scene.sparse_dir == tmp_path / "sparse" / "0"


def test_require_images_missing(tmp_path):
    with pytest.raises(SceneError):
        Scene(tmp_path).require_images()


def test_require_images_empty(tmp_path):
    (tmp_path / "images").mkdir()
    with pytest.raises(SceneError):
        Scene(tmp_path).require_images()


def test_require_images_ok(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "a.jpg").touch()
    Scene(tmp_path).require_images()


def test_require_reconstruction_missing(tmp_path):
    with pytest.raises(SceneError):
        Scene(tmp_path).require_reconstruction()


def test_require_reconstruction_empty_dir(tmp_path):
    (tmp_path / "sparse" / "0").mkdir(parents=True)
    scene = Scene(tmp_path)
    assert not scene.has_reconstruction()
    with pytest.raises(SceneError):
        scene.require_reconstruction()


def test_has_reconstruction_non_empty(tmp_path):
    sparse = tmp_path / "sparse" / "0"
    sparse.mkdir(parents=True)
    (sparse / "cameras.bin").touch()
    scene = Scene(tmp_path)
    assert scene.has_reconstruction()
    scene.require_reconstruction()


def test_link_colmap_masks_none_without_masks(tmp_path):
    assert Scene(tmp_path).link_colmap_masks(tmp_path / "masks_colmap") is None


def test_link_colmap_masks_names_and_idempotent(tmp_path):
    masks = tmp_path / "masks"
    masks.mkdir()
    (masks / "foo.png").touch()
    scene = Scene(tmp_path)
    dest = tmp_path / "out" / "masks_colmap"

    assert scene.link_colmap_masks(dest) == dest
    assert (dest / "foo.jpg.png").is_symlink()

    scene.link_colmap_masks(dest)
    assert sorted(p.name for p in dest.iterdir()) == ["foo.jpg.png"]
