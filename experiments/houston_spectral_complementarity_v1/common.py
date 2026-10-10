import json,hashlib
from pathlib import Path
import numpy as np
BASE=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/tokenizer_stem_value_diagnostic_v1')
OUT=BASE.parent/'houston_spectral_complementarity_v1'
SEEDS=(2100,2101,2102)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def verify():
 for p,h in json.loads((OUT/'immutable_manifest.json').read_text()).items():assert sha(p)==h,p
