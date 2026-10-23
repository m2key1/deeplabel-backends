import pytest

albumentations = pytest.importorskip('albumentations')

import augmentation
from pydantic.errors import PydanticInvalidForJsonSchema


def test_the_catalog_leaves_out_a_transform_it_cannot_describe(monkeypatch):
    def undescribable():
        raise PydanticInvalidForJsonSchema('Cannot generate a JsonSchema for numpy.ndarray')

    monkeypatch.setattr(albumentations.Rotate.InitSchema, 'model_json_schema', undescribable)
    names = [t['name'] for t in augmentation.catalog()['transforms']]
    assert 'Rotate' not in names
    assert 'Affine' in names
