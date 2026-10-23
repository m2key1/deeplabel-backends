import json
import os
from pathlib import Path

from deeplabel_backend.train import ProgressLog, write_outputs


def main() -> None:
    """Runs locally inside the backend's container and on the cluster inside the same
    image, with no network: everything comes from the four TRAIN_* paths."""
    snapshot = Path(os.environ['TRAIN_SNAPSHOT'])
    output = Path(os.environ['TRAIN_OUTPUT'])
    base = Path(os.environ['TRAIN_BASE']) / 'weights'
    params = json.loads(Path(os.environ['TRAIN_PARAMS']).read_text(encoding='utf-8'))
    epochs = params.get('epochs') or 10
    learning_rate = params['hyperparams'].get('learning_rate', 0.001)

    progress = ProgressLog(output)
    for epoch in range(1, epochs + 1):
        # train one epoch on snapshot / 'train', validate on snapshot / 'valid'
        # starting from base if base.is_file(), at learning_rate
        progress.write(epoch=epoch, loss=0.0, map50=0.0)
    progress.close()

    (output / 'model.pt').write_bytes(b'')
    write_outputs(output, {'val_metric': 0.0, 'val_metric_name': 'map50', 'higher_is_better': True},
                  [{'role': 'weights', 'path': 'model.pt'}])


if __name__ == '__main__':
    main()
