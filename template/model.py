from pathlib import Path

from deeplabel_backend import Loaded, Model


class MyModel(Model):
    """Rename, then fill in load() and predict(). Section 4 of docs/MODEL-BACKENDS.md has
    the result schema of each task; coordinates are fractions of the image sent."""

    name = 'my-layout'
    version = '0.1.0'
    model_type = 'yolo'
    tasks = ('layout',)
    weight_roles = {'weights': 'required'}

    def tasks_detail(self) -> dict[str, dict]:
        return {'layout': {'batch': {'max_items': 16}, 'labels': ['Header', 'Paragraph'], 'max_side': 3000,
                           'params': {'threshold': {'type': 'number', 'default': 0.3}}}}

    def load(self, files: dict[str, Path], device: str) -> Loaded:
        # e.g. network = YOLO(files['weights']).to('cuda' if device == 'auto' else device)
        network = None
        return Loaded(handle=network, facts={})

    def unload(self, loaded: Loaded) -> None:
        # e.g. del loaded.handle; torch.cuda.empty_cache()
        pass

    def predict(self, loaded, task, images, params, contexts):
        threshold = params.get('threshold', 0.3)
        results = []
        for image in images:
            detections = []
            # for box in loaded.handle(image, conf=threshold)[0].boxes:
            #     x0, y0, x1, y1 = box.xyxyn[0].tolist()
            #     detections.append({'label': names[int(box.cls)], 'bbox': [x0, y0, x1, y1],
            #                        'confidence': float(box.conf)})
            results.append({'detections': detections})
        return results

    def batch_size(self, task: str, params: dict) -> int | None:
        return 8

    def train_info(self) -> dict | None:
        """Return None if the model does not train here. Otherwise section 6.2's block."""
        return {
            'tasks': ['layout'],
            'snapshot': 'page-boxes/1',
            'entrypoint': ['python', '/srv/train_entrypoint.py'],
            'image': 'my-layout-backend:0.1.0',
            'resources': {'gpus': {'min': 1, 'max': 1, 'multi_gpu': False}, 'cpus_per_gpu': 8,
                          'memory_per_gpu_gb': 32, 'default_batch': {'local': 8, 'cluster': 16}},
            'tunables': [{'key': 'learning_rate', 'label': 'Learning rate', 'group': 'Schedule', 'kind': 'float',
                          'help': 'Peak learning rate.', 'above': 0, 'default': 0.001}],
            'outputs': ['weights'],
            'resume': False,
        }
