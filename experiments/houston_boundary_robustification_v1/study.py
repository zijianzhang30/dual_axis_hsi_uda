import hashlib,importlib.util,json
from pathlib import Path
ROOT=Path('/home/zhangzj26/dual_axis_hsi_uda')
OUT=ROOT/'results/houston_boundary_robustification_v1'
HERE=Path(__file__).resolve().parent
BASE=ROOT/'results/tokenizer_mechanism_v2'
GEOM=ROOT/'results/houston_boundary_geometric_stress_diagnostic_v1'
DATA=ROOT/'results/tokenizer_stem_value_diagnostic_v1/data'
SEEDS=(2100,2101,2102)
ARMS=('B0','B1','B2')
PAIRS=[(a,b) for a in range(7) for b in range(a+1,7)]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def module(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def environment():
 import sys
 sys.path.insert(0,str(BASE))
 return module('boundary_frozen_C_protocol',BASE/'common.py')
def verify():
 for rel,h in json.loads((OUT/'PRETRAIN_WEIGHTS_CODE_FROZEN.json').read_text())['files_sha256'].items():assert sha(rel)==h,rel
def state_hash(m):return environment().state_hash(m)
