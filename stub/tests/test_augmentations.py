import pytest
from PIL import Image

from deeplabel_backend import errors
from deeplabel_stub.model import StubModel


def stack(*steps: dict) -> dict:
    return {'transform': {'__class_fullname__': 'Compose', 'p': 1.0, 'transforms': list(steps)}}


def preview(spec: dict):
    return StubModel().augmentation_preview(spec, [Image.new('RGB', (40, 10), 'white')], 0, 1)


def refusal(spec: dict) -> str:
    with pytest.raises(errors.BackendError) as raised:
        preview(spec)
    return raised.value.message


def test_the_catalogue_has_every_field_kind_and_a_default_stack():
    catalogue = StubModel().augmentations()
    assert [t['name'] for t in catalogue['transforms']] == [
        'Affine', 'GaussNoise', 'Morphological', 'InvertImg', 'BBoxSafeRandomCrop', 'RandomCrop']
    steps = catalogue['default']['transform']['transforms']
    assert [s['__class_fullname__'] for s in steps] == ['Affine', 'GaussNoise', 'OneOf', 'Normalize', 'ToTensorV2']


def test_a_sound_stack_inverts_each_image():
    assert preview(stack({'__class_fullname__': 'InvertImg', 'p': 1.0}))[0].image.getpixel((0, 0)) == (0, 0, 0)


def test_a_range_whose_low_end_exceeds_its_high_end_is_refused_naming_the_step():
    message = refusal(stack({'__class_fullname__': 'GaussNoise', 'p': 1.0, 'std_range': [0.9, 0.1]}))
    assert message == 'GaussNoise: std_range: the low end must not exceed the high end'


def test_an_unknown_step_is_refused():
    assert refusal(stack({'__class_fullname__': 'Wobble', 'p': 1.0})) == 'Wobble: unknown transform Wobble'


def test_a_box_step_is_refused_where_the_trainer_supplies_no_boxes(monkeypatch):
    monkeypatch.delenv('STUB_TRAIN_SNAPSHOT', raising=False)
    message = refusal(stack({'__class_fullname__': 'OneOf', 'p': 1.0, 'transforms': [
        {'__class_fullname__': 'BBoxSafeRandomCrop', 'p': 1.0}]}))
    assert message == 'OneOf[0].BBoxSafeRandomCrop: applying it to an image failed: this step needs bounding boxes'


def test_a_box_step_is_fine_where_the_trainer_supplies_boxes(monkeypatch):
    monkeypatch.setenv('STUB_TRAIN_SNAPSHOT', 'page-boxes/1')
    assert len(preview(stack({'__class_fullname__': 'BBoxSafeRandomCrop', 'p': 1.0}))) == 1


def test_the_harshest_of_many_draws_is_more_mangled_than_one():
    image = Image.new('RGB', (40, 10), (100, 150, 200))
    spec = stack({'__class_fullname__': 'InvertImg', 'p': 1.0})
    [one] = StubModel().augmentation_preview(spec, [image], 0, 1)
    [harshest] = StubModel().augmentation_preview(spec, [image], 0, 12)
    assert one.image.getpixel((0, 0)) == (155, 105, 55)
    assert harshest.image.getpixel((0, 0)) == (128, 0, 0)


def test_boxes_come_back_where_they_went_in():
    page = Image.new('RGB', (40, 10), 'white')
    [shown] = StubModel().augmentation_preview(stack(), [page], 0, 1, boxes=[[[1, 2, 30, 8]]])
    assert shown.boxes == [[1, 2, 30, 8]]


def test_line_outlines_come_back_where_they_went_in():
    page = Image.new('RGB', (40, 10), 'white')
    outline = [[1, 2], [30, 2], [30, 8], [1, 8]]
    [shown] = StubModel().augmentation_preview(stack(), [page], 0, 1, polygons=[[outline]])
    assert shown.polygons == [outline]
