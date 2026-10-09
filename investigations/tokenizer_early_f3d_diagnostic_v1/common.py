import sys,json,hashlib,importlib.util
from pathlib import Path
import numpy as np,torch
sys.dont_write_bytecode=True
OUT=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/tokenizer_early_f3d_diagnostic_v1')
BASE=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/tokenizer_mechanism_v2')
PREVIOUS=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/tokenizer_stem_value_diagnostic_v1')
SEEDS=[2100,2101,2102]
sys.path.insert(0,str(BASE))
spec=importlib.util.spec_from_file_location('frozen_model_common',BASE/'common.py');frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return frozen.sha(p)
def hasharray(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
