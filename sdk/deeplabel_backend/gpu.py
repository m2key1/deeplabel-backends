"""The card this backend runs on, as /v1/info reports it (6.9): DeepLabel picks a local
run's batch from the backend's batch_by_gpu by this card's id."""
import functools
import subprocess

# Card ids, as DeepLabel and batch_by_gpu name them, by a part of the driver's card name.
CARDS = (('h200', 'H200'), ('rtx6000bb', 'RTX PRO 6000'), ('rtx6000bb', 'RTX 6000 Blackwell'),
         ('rtx5090', 'RTX 5090'), ('l40s', 'L40S'))


def card_id(name: str) -> str | None:
    return next((card for card, part in CARDS if part.lower() in name.lower()), None)


def parse(smi_output: str) -> dict | None:
    """nvidia-smi's `name, memory.total` CSV lines, one per card, into the first card's
    description and how many there are."""
    rows = [[part.strip() for part in line.split(',')] for line in smi_output.strip().splitlines() if line.strip()]
    if not rows:
        return None
    name, memory = rows[0][0], rows[0][1] if len(rows[0]) > 1 else ''
    digits = ''.join(ch for ch in memory if ch.isdigit())
    return {'name': name, 'card': card_id(name), 'count': len(rows),
            'memory_gb': round(int(digits) / 1024) if digits else None}


@functools.cache
def detect() -> dict | None:
    try:
        found = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
                               capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return parse(found.stdout) if found.returncode == 0 else None
