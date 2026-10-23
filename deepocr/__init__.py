import os
import sys
from pathlib import Path

# The DeepOCR checkout this wrapper runs: in the image, the pinned revision at /srv/model.
MODEL_REPO = Path(os.environ.get('DEEPOCR_REPO', '/srv/model')).resolve()
if str(MODEL_REPO) not in sys.path:
    sys.path.insert(0, str(MODEL_REPO))

# The builtin weights, DeepOCR v3, built into the image from Hugging Face (Dockerfile).
BUILTIN_DIR = Path(os.environ.get('DEEPOCR_BUILTIN_DIR', '/srv/builtin'))
BUILTIN_FILES = {'weights': 'deepocr-v3.pt', 'lm': 'union_bytes_v3.bin', 'lm_dict': 'union_bytes_v3.dict'}
