import pytest

pytest.importorskip('albumentations')

from deeplabel_backend import rfdetr_augmentation

LISTED = frozenset({'Rotate'})


def spec(*steps: dict) -> dict:
    return {'transform': {'__class_fullname__': 'Compose', 'p': 1.0, 'transforms': list(steps)}}


def test_a_spec_becomes_the_dict_of_name_and_params_rfdetr_trains_with_in_its_order():
    group = {'__class_fullname__': 'OneOf', 'p': 1.0, 'transforms': [
        {'__class_fullname__': 'Rotate', 'p': 0.4, 'limit': [-3, 3]}, {'__class_fullname__': 'GaussianBlur', 'p': 0.6}]}
    config = rfdetr_augmentation.to_aug_config(spec({'__class_fullname__': 'ToGray', 'p': 1.0}, group), 'lines', LISTED)
    assert config == {'ToGray': {'p': 1.0}, 'OneOf': {'p': 1.0, 'transforms': [
        {'Rotate': {'p': 0.4, 'limit': [-3, 3]}}, {'GaussianBlur': {'p': 0.6}}]}}
    assert list(config) == ['ToGray', 'OneOf']


def test_a_step_chosen_twice_is_refused_as_rfdetr_keys_its_steps_by_name():
    with pytest.raises(ValueError, match=r'^Compose\[2\]\.GaussianBlur: RF-DETR takes each step once - '
                                         r'put the copies in a group$'):
        rfdetr_augmentation.to_aug_config(spec({'__class_fullname__': 'GaussianBlur', 'p': 0.5},
                                               {'__class_fullname__': 'ToGray', 'p': 1.0},
                                               {'__class_fullname__': 'GaussianBlur', 'p': 0.2}), 'lines', LISTED)


def test_an_aug_config_reads_back_as_the_stack_rfdetr_builds_with_its_groups_always_run():
    stack = rfdetr_augmentation.to_spec({'OneOf': {'p': 0.3, 'transforms': [{'GaussianBlur': {'p': 0.5}}]}})
    [group] = stack['transform']['transforms']
    assert (group['__class_fullname__'], group['p']) == ('OneOf', 1.0)


def test_a_group_rfdetr_would_run_every_time_is_refused_below_p_1():
    group = {'__class_fullname__': 'Sequential', 'p': 0.3, 'transforms': [{'__class_fullname__': 'GaussianBlur', 'p': 1.0}]}
    with pytest.raises(ValueError, match=r'^Compose\[0\]\.Sequential: RF-DETR runs a Sequential group every time'):
        rfdetr_augmentation.to_aug_config(spec(group), 'lines', LISTED)


def test_a_step_that_moves_the_page_but_not_what_is_on_it_is_refused_naming_what_stays():
    nested = {'__class_fullname__': 'OneOf', 'p': 1.0, 'transforms': [{'__class_fullname__': 'Perspective', 'p': 1.0}]}
    with pytest.raises(ValueError, match=r'^Compose\[0\]\.OneOf\[0\]\.Perspective: RF-DETR would move the page '
                                         r'but not its lines with it$'):
        rfdetr_augmentation.to_aug_config(spec(nested), 'lines', LISTED)


def test_a_page_moving_step_rfdetr_lists_is_taken():
    assert rfdetr_augmentation.to_aug_config(spec({'__class_fullname__': 'Rotate', 'p': 1.0}), 'boxes', LISTED) == {'Rotate': {'p': 1.0}}
