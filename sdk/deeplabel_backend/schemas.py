from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

CONTRACT = '1.4'
TASKS = ('layout', 'lines', 'ocr', 'map_segmentation', 'edges')
WEIGHT_ROLES = ('weights', 'lm', 'lm_dict', 'config')
SNAPSHOT_FORMAT = 'deeplabel-snapshot/1'
PROGRESS_FORMAT = 'deeplabel-progress/1'


class Message(BaseModel):
    model_config = ConfigDict(extra='allow')


class FileRef(Message):
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    size: int | None = Field(default=None, ge=0)
    url: str


class LoadRequest(Message):
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    model_type: str | None = None
    files: dict[str, FileRef]


class UnloadRequest(Message):
    sha256: str | None = None


class WeightsRef(Message):
    sha256: str
    files: dict[str, str] = {}


class Item(Message):
    id: str
    image: str
    context: dict[str, Any] = {}


class PredictRequest(Message):
    weights: WeightsRef | None = None
    params: dict[str, Any] = {}
    items: list[Item]


class RunRef(Message):
    id: int | str
    attempt: int = 1
    name: str = ''
    owner: str = ''


class SnapshotRef(Message):
    manifest_url: str


class BaseWeights(Message):
    """The version a run starts from: stored files by role, or the backend's builtin."""
    sha256: str | None = None
    files: dict[str, FileRef] = {}
    builtin: bool = False


class TrainParams(Message):
    epochs: int | None = None
    # Per GPU. The effective batch is batch_size x grad_accum_steps x gpus (6.9).
    batch_size: int | None = None
    gpus: int = 1
    grad_accum_steps: int | None = None
    # A factor on the trainer's own learning rates, for an effective batch above the one
    # they were tuned at; None leaves them as they are.
    lr_scale: float | None = None
    hyperparams: dict[str, Any] = {}
    augmentation: dict[str, Any] | None = None
    build_lm: bool = False


class TrainRequest(Message):
    run: RunRef
    task: str
    snapshot: SnapshotRef
    base_weights: BaseWeights | None = None
    params: TrainParams = TrainParams()


class PreviewRequest(Message):
    spec: dict[str, Any]
    seed: int = 0
    draws: int = 1
    # Per image, its boxes as [x0, y0, x1, y1] in the image's pixels, for a trainer handed boxes.
    boxes: list[list[list[float]]] | None = None


class Health(Message):
    status: Literal['ok', 'starting', 'error']
    contract: str
    busy: int = Field(ge=0)
    loaded: list[str]


class TaskDetail(Message):
    batch: dict[str, int] = {}
    params: dict[str, Any] = {}


class Info(Message):
    contract: str = Field(pattern=r'^\d+\.\d+$')
    name: str
    version: str
    model_type: str
    tasks: list[Literal['layout', 'lines', 'ocr', 'map_segmentation', 'edges']] = Field(min_length=1)
    capabilities: list[str]
    concurrency: int = Field(ge=1)
    max_request_mb: int = Field(ge=1)
    tasks_detail: dict[str, TaskDetail]
    weights: dict[str, Any]
    train: dict[str, Any] | None = None


class ErrorDetail(BaseModel):
    code: str
    message: str
    retryable: bool


class ErrorBody(BaseModel):
    error: ErrorDetail
