from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass
class Loaded:
    """What load() hands back: the model's own handle, and the facts DeepLabel stores on
    the model row (OCR target_height and max_width, a map checkpoint's class count)."""
    handle: Any
    facts: dict = field(default_factory=dict)


@dataclass
class Prediction:
    """predict() may return this instead of a plain list when the response carries more
    than the results, such as the decode an OCR request actually ran."""
    results: list[dict]
    extra: dict = field(default_factory=dict)


class Model:
    """The hooks a backend implements. Everything else (HTTP, auth, the weights cache,
    batching, validation, training orchestration) is the SDK's."""

    name = 'model'
    version = '0.0.0'
    model_type = ''
    tasks: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    concurrency = 1
    max_request_mb = 64
    weight_roles: dict[str, str] = {'weights': 'required'}
    builtin = False

    def tasks_detail(self) -> dict[str, dict]:
        return {task: {} for task in self.tasks}

    def load(self, files: dict[str, Path], device: str) -> Loaded:
        raise NotImplementedError

    def unload(self, loaded: Loaded) -> None:
        pass

    def predict(self, loaded: Loaded, task: str, images: list[Image.Image], params: dict,
                contexts: list[dict]) -> list[dict] | Prediction:
        raise NotImplementedError

    def batch_size(self, task: str, params: dict) -> int | None:
        """How many images one predict() call takes; None passes the whole request. The
        SDK halves it after an out-of-memory error."""
        return None

    def train_info(self) -> dict | None:
        """The /info train block (section 6.2), or None for a backend that does not train."""
        return None

    def train_command(self) -> list[str]:
        return list((self.train_info() or {}).get('entrypoint') or [])

    def tunables(self) -> list[dict]:
        return list((self.train_info() or {}).get('tunables') or [])

    def augmentations(self) -> dict:
        return {}

    def augmentation_preview(self, spec: dict, images: list[Image.Image], seed: int,
                             draws: int) -> list[Image.Image]:
        raise NotImplementedError
