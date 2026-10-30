import io
import json
import sys
import warnings
from pathlib import Path

import pytest

albumentations = pytest.importorskip('albumentations')

from deeplabel_backend import Model, augmentation, create_app
from deeplabel_backend.errors import BackendError
from fastapi.testclient import TestClient
from PIL import Image
from pydantic.errors import PydanticInvalidForJsonSchema

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'contract_tests'))
from fixture_server import SECRET_HEADER

SECRET = 'unit-secret'


def spec(*steps: dict) -> dict:
    return {'transform': {'__class_fullname__': 'Compose', 'p': 1.0, 'transforms': list(steps)}}


FLIP = {'__class_fullname__': 'HorizontalFlip', 'p': 1.0}


def page(width: int = 200, height: int = 100) -> Image.Image:
    return Image.new('RGB', (width, height), 'white')


def test_the_catalog_leaves_out_a_transform_it_cannot_describe(monkeypatch):
    def undescribable():
        raise PydanticInvalidForJsonSchema('Cannot generate a JsonSchema for numpy.ndarray')

    monkeypatch.setattr(albumentations.Rotate.InitSchema, 'model_json_schema', undescribable)
    names = [t['name'] for t in augmentation.catalog()['transforms']]
    assert 'Rotate' not in names
    assert 'Affine' in names


def test_the_catalog_carries_the_trainers_own_stack_as_its_default():
    stack = augmentation.to_spec([albumentations.Rotate(limit=2, p=0.5)])
    assert augmentation.catalog(stack)['default'] == stack
    assert stack['transform']['transforms'][0]['limit'] == [-2, 2]
    assert 'default' not in augmentation.catalog()


def test_a_line_is_shown_at_the_height_the_model_reads_and_has_no_boxes():
    [shown] = augmentation.preview(spec(FLIP), [page(400, 50)], seed=1, draws=1, height=100)
    assert shown.image.size == (800, 100)
    assert shown.boxes is None


def test_a_box_goes_through_the_stack_with_its_page():
    [shown] = augmentation.preview(spec(FLIP), [page()], seed=1, draws=1, boxes=[[[10, 20, 50, 60]]])
    assert shown.image.size == (200, 100)
    assert shown.boxes == [[150.0, 20.0, 190.0, 60.0]]


def test_a_page_is_scaled_to_max_side_and_its_boxes_with_it():
    [shown] = augmentation.preview(spec(), [page(400, 200)], seed=1, draws=1, max_side=200,
                                   boxes=[[[40, 40, 100, 80]]])
    assert shown.image.size == (200, 100)
    assert shown.boxes == [[20.0, 20.0, 50.0, 40.0]]


def test_a_box_the_draw_cuts_away_is_none_and_the_others_keep_their_place():
    crop = {'__class_fullname__': 'Crop', 'p': 1.0, 'x_min': 100, 'y_min': 0, 'x_max': 200, 'y_max': 100}
    [shown] = augmentation.preview(spec(crop), [page()], seed=1, draws=1,
                                   boxes=[[[10, 20, 50, 60], [120, 10, 160, 90]]])
    assert shown.image.size == (100, 100)
    assert shown.boxes == [None, [20.0, 10.0, 60.0, 90.0]]


def test_a_line_outline_goes_through_the_stack_with_its_page():
    outline = [[10, 20], [50, 20], [50, 30], [10, 30]]
    [shown] = augmentation.preview(spec(FLIP), [page()], seed=1, draws=1, polygons=[[outline]])
    xs = sorted({x for x, _ in shown.polygons[0]})
    ys = sorted({y for _, y in shown.polygons[0]})
    assert (xs[0], xs[-1], ys[0], ys[-1]) == (149.0, 189.0, 20.0, 30.0)
    assert shown.boxes is None


def test_a_page_is_scaled_to_max_side_and_its_outlines_with_it():
    outline = [[40, 40], [100, 40], [100, 80], [40, 80]]
    [shown] = augmentation.preview(spec(), [page(400, 200)], seed=1, draws=1, max_side=200, polygons=[[outline]])
    assert sorted(map(tuple, shown.polygons[0])) == [(20.0, 20.0), (20.0, 40.0), (50.0, 20.0), (50.0, 40.0)]


def test_an_outline_the_draw_cuts_away_is_none_and_the_others_keep_their_place():
    crop = {'__class_fullname__': 'Crop', 'p': 1.0, 'x_min': 100, 'y_min': 0, 'x_max': 200, 'y_max': 100}
    gone = [[10, 20], [50, 20], [50, 60], [10, 60]]
    kept = [[120, 10], [160, 10], [160, 90], [120, 90]]
    [shown] = augmentation.preview(spec(crop), [page()], seed=1, draws=1, polygons=[[gone, kept]])
    assert shown.polygons[0] is None
    assert sorted(map(tuple, shown.polygons[1])) == [(20.0, 10.0), (20.0, 90.0), (60.0, 10.0), (60.0, 90.0)]


class Blots(albumentations.ImageOnlyTransform):
    """Blots of ink across the page.

    Args:
        count (tuple[int, int]): How many blots, at least and at most.
    """

    class InitSchema(albumentations.ImageOnlyTransform.InitSchema):
        count: tuple[int, int] = (1, 3)

    def __init__(self, count=(1, 3), p=0.5):
        super().__init__(p=p)
        self.count = count

    def apply(self, img, **params):
        return img

    def get_transform_init_args_names(self):
        return ('count',)


def test_a_trainers_own_step_is_offered_under_the_name_a_spec_carries_it_by():
    offered = {t['name']: t for t in augmentation.catalog(extra=(Blots,))['transforms']}
    name = Blots.get_class_fullname()
    assert name.endswith('.Blots') and name in offered
    assert offered[name]['category'] == 'other'
    assert [(p['name'], p['kind'], p['default']) for p in offered[name]['params']] == [('count', 'range', [1, 3])]
    assert augmentation.to_spec([Blots(count=(2, 4))])['transform']['transforms'][0]['__class_fullname__'] == name


def test_a_box_step_on_a_trainer_without_boxes_is_refused_naming_what_it_lacks():
    safe_crop = {'__class_fullname__': 'BBoxSafeRandomCrop', 'p': 1.0}
    with pytest.raises(BackendError) as refused:
        augmentation.preview(spec(safe_crop), [page()], seed=1, draws=1, height=50)
    assert 'which this task does not supply' in str(refused.value.message)


def test_an_argument_the_step_does_not_take_is_refused_without_leaving_warnings_raised():
    wrong = {'__class_fullname__': 'GaussianBlur', 'p': 1.0, 'blurriness': 3}
    with pytest.raises(BackendError) as refused:
        augmentation.preview(spec(FLIP, wrong), [page()], seed=1, draws=1, height=50)
    assert refused.value.message == "Compose[1].GaussianBlur: Argument(s) 'blurriness' are not valid for transform GaussianBlur"
    assert not [f for f in warnings.filters if f[0] == 'error' and f[1] is not None and 'not valid' in f[1].pattern]


class Trainer(Model):
    name = 'trainer'
    model_type = 'trainer'
    tasks = ('layout',)

    def train_info(self):
        return {'tasks': ['layout'], 'snapshot': 'page-boxes/1', 'entrypoint': ['train']}

    def augmentation_preview(self, spec, images, seed, draws, boxes=None, polygons=None):
        return augmentation.preview(spec, images, seed, draws, max_side=1000, boxes=boxes, polygons=polygons)


def post_preview(client, body: dict, count: int = 1):
    out = io.BytesIO()
    page().save(out, format='PNG')
    return client.post('/v1/train/augmentations/preview', headers={SECRET_HEADER: SECRET},
                       files=[('request', ('r.json', json.dumps(body), 'application/json')),
                              *[(f'img{i}', (f'{i}.png', out.getvalue(), 'image/png')) for i in range(count)]])


def test_the_preview_endpoint_answers_where_the_boxes_went(tmp_path):
    client = TestClient(create_app(Trainer(), secret=SECRET, cache_dir=tmp_path))
    answer = post_preview(client, {'spec': spec(FLIP), 'boxes': [[[10, 20, 50, 60]]]})
    assert answer.status_code == 200
    assert answer.json()['boxes'] == [[[150.0, 20.0, 190.0, 60.0]]]
    assert answer.json()['images'][0].startswith('data:image/png;base64,')

    without = post_preview(client, {'spec': spec(FLIP)})
    assert 'boxes' not in without.json()
    assert len(without.json()['images']) == 1


def test_boxes_for_fewer_images_than_sent_are_refused(tmp_path):
    client = TestClient(create_app(Trainer(), secret=SECRET, cache_dir=tmp_path))
    answer = post_preview(client, {'spec': spec(FLIP), 'boxes': [[[10, 20, 50, 60]]]}, count=2)
    assert answer.status_code == 400
    assert 'boxes are given for 1 images, not the 2 sent' in answer.text


def test_the_preview_endpoint_answers_where_the_outlines_went(tmp_path):
    client = TestClient(create_app(Trainer(), secret=SECRET, cache_dir=tmp_path))
    answer = post_preview(client, {'spec': spec(), 'polygons': [[[[10, 20], [50, 20], [50, 30], [10, 30]]]]})
    assert answer.status_code == 200
    assert sorted(map(tuple, answer.json()['polygons'][0][0])) == [(10, 20), (10, 30), (50, 20), (50, 30)]
    assert 'boxes' not in answer.json()


def test_outlines_for_fewer_images_than_sent_are_refused(tmp_path):
    client = TestClient(create_app(Trainer(), secret=SECRET, cache_dir=tmp_path))
    answer = post_preview(client, {'spec': spec(), 'polygons': [[]]}, count=2)
    assert answer.status_code == 400
    assert 'polygons are given for 1 images, not the 2 sent' in answer.text
