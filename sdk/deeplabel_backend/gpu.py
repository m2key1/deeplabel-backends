"""The card this backend runs on: what /v1/info reports of it (6.9), from which DeepLabel
picks a local run's batch, and its memory for /v1/memory (3.9), from which DeepLabel decides
which backends of the card stay loaded."""
import functools
import gc
import subprocess
import sys

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


def parse_memory(smi_output: str) -> dict | None:
    """nvidia-smi's `memory.total, memory.free` CSV lines in MiB, one per card, into the
    first card's in bytes."""
    rows = [[part.strip() for part in line.split(',')] for line in smi_output.strip().splitlines() if line.strip()]
    if not rows or len(rows[0]) < 2:
        return None
    total, free = rows[0][:2]
    return {'total': int(total) << 20, 'free': int(free) << 20}


def card_memory() -> dict | None:
    """The whole card's memory, every process on it counted. Asked of the driver, so a
    backend that has not touched CUDA does not start a context just to answer."""
    try:
        found = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.free', '--format=csv,noheader,nounits'],
                               capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return parse_memory(found.stdout) if found.returncode == 0 else None


def _cuda():
    """torch, once the model code has imported it and started CUDA in this process; the
    SDK itself never imports it, so a backend without torch measures nothing."""
    torch = sys.modules.get('torch')
    return torch if torch is not None and torch.cuda.is_initialized() else None


def start_measuring() -> None:
    torch = _cuda()
    if torch is not None:
        torch.cuda.reset_peak_memory_stats()


def peak() -> int | None:
    """The most card memory this process's allocator held since start_measuring."""
    torch = _cuda()
    return torch.cuda.max_memory_reserved() if torch is not None else None


def release() -> None:
    """Hand what an unloaded model left in the allocator's cache back to the card, so other
    backends and the card's free memory see it."""
    gc.collect()
    torch = _cuda()
    if torch is not None:
        torch.cuda.empty_cache()
