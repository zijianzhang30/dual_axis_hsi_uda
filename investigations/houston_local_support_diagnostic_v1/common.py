import hashlib
import json
from pathlib import Path
import numpy as np

PROJECT = Path('/home/zhangzj26/dual_axis_hsi_uda')
HERE = Path(__file__).resolve().parent
OUT = PROJECT / 'results/houston_local_support_diagnostic_v1'
CACHE = PROJECT / 'results/houston_class_geometry_diagnostic_v1'
BASE = PROJECT / 'results/tokenizer_mechanism_v2'
DATA = PROJECT / 'results/tokenizer_stem_value_diagnostic_v1/data'
SEEDS = (2100, 2101, 2102)
STAGES = ('T', 'Z')
KS = (5, 15)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def ahash(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
