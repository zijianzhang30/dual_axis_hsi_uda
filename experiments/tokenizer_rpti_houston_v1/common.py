import hashlib
import importlib.util
import json
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np
import torch

PROJECT=Path('/home/zhangzj26/dual_axis_hsi_uda')
BASE=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/tokenizer_mechanism_v2')
OUT=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/tokenizer_rpti_houston_v1')
SEEDS=(2100,2101,2102)
ARMS=('J1',)

def dump(path,obj):Path(path).write_text(json.dumps(obj,indent=2)+'\n')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def state_hash(model):
 h=hashlib.sha256()
 for value in model.state_dict().values():h.update(value.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj

def environment():
 snapshot=OUT/'snapshot/BiDA'
 sys.path.insert(0,str(snapshot/'experiments/bida_memory_v1'));sys.path.insert(0,str(snapshot))
 from experiments.bida_memory_v1.run import make_loaders as houston_loaders
 from models.get_model import get_model
 from models.BiDA import BiDAnet
 from utils.scheduler import load_scheduler
 from utils.utils_HSI import seed_worker
 from utils.dataset import HSIDataset,sample_gt
 joint=module('frozen_joint_normalization',OUT/'snapshot/normalization.py')
 helper=module('frozen_metric_helpers',OUT/'snapshot/helpers.py')
 pavia=module('frozen_pavia_loaders',OUT/'snapshot/pavia_loaders.py')
 return houston_loaders,get_model,BiDAnet,load_scheduler,seed_worker,joint,helper,pavia

def loaders(args):
 env=environment()
 if args.dataset=='houston':return env[0](args)[:4]
 return env[7].make_loaders(args)

def build(seed,arm,dataset):
 from argparse import Namespace
 import scorer
 env=environment();bands=48 if dataset=='houston' else 102
 args=Namespace(model='BiDA',lr=.01,num_tokens=4,dim=64,depth=3,loss_type='softmax',labelsmooth='off')
 if dataset=='houston':
  model=env[1]('BiDA','Houston13',13,args);unused=env[1]('BiDA','Houston13',13,args,ema=True)
 else:
  model=env[2](n_bands=102,num_classes=7,num_tokens=4,dim=64,depth=3);unused=env[2](n_bands=102,num_classes=7,num_tokens=4,dim=64,depth=3)
 del unused
 original=state_hash(model);before={k:v.clone() for k,v in model.state_dict().items()}
 env[5].install(model,'fixed_mixture')
 cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state_all()
 scorer.install(model,'C',seed,bands)
 if arm=='J1':__import__('rpti').install(model)
 assert torch.equal(cpu,torch.get_rng_state())
 assert all(torch.equal(x,y) for x,y in zip(cuda,torch.cuda.get_rng_state_all()))
 assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items() if arm=='A' or not k.startswith('conv_a.'))
 return model,original
