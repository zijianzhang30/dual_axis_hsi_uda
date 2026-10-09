"""Independent immutable source/data provenance. Does not modify existing jobs."""
import ast
import importlib.metadata
import shutil
from pathlib import Path
from common import OUT,PROJECT,dump,sha

def extract(path,names,imports):
 source=path.read_text();tree=ast.parse(source)
 functions=[ast.get_source_segment(source,n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names]
 # Decorators must be retained for inference functions; extraction starts at def.
 body=[]
 for text in functions:
  if text.startswith('def source_evaluate') or text.startswith('def predict'):text='@torch.inference_mode()\n'+text
  body.append(text)
 return imports+'\n\n'+'\n\n'.join(body)+'\n'

def main():
 OUT.mkdir(exist_ok=False)
 for p in Path(__file__).parent.iterdir():
  if p.is_file():shutil.copy2(p,OUT/p.name)
 (PROJECT/'results/tokenizer_mechanism_v2').symlink_to(OUT,target_is_directory=True)
 snapshot=OUT/'snapshot';(snapshot/'BiDA').mkdir(parents=True)
 original=Path('/home/zhangzj26/IEEE_TCSVT_BiDA');hashes={}
 for p in original.rglob('*.py'):
  if any(x in p.parts for x in ['.git','.venv','__pycache__']):continue
  relative=p.relative_to(original);target=snapshot/'BiDA'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);hashes[str(relative)]=sha(p)
 joint=PROJECT/'experiments/bn_stabilization_v1/normalization.py';shutil.copy2(joint,snapshot/'normalization.py')
 helper=PROJECT/'experiments/bida_self_dropout/train.py'
 (snapshot/'helpers.py').write_text(extract(helper,{'state_hash','file_hash','source_evaluate','predict','metrics'},'import hashlib\nfrom pathlib import Path\nimport numpy as np\nimport torch\nimport torch.nn.functional as F\nfrom sklearn.metrics import confusion_matrix,cohen_kappa_score'))
 pavia=PROJECT/'experiments/pavia_joint_v1/train.py'
 (snapshot/'pavia_loaders.py').write_text(extract(pavia,{'load_scene','make_loaders'},'from pathlib import Path\nimport numpy as np\nimport torch\nfrom scipy.io import loadmat\nfrom utils.dataset import HSIDataset,sample_gt'))
 provenance={'original_BiDA_sha256':hashes,'joint_sha256':sha(joint),'metric_source_sha256':sha(helper),'pavia_loader_source_sha256':sha(pavia),
  'snapshot_sha256':{str(p.relative_to(snapshot)):sha(p) for p in snapshot.rglob('*.py')},
  'study_sha256':{p.name:sha(p) for p in OUT.glob('*.py')},'protocol_sha256':sha(OUT/'PROTOCOL.md'),
  'packages':{n:importlib.metadata.version(n) for n in ['torch','numpy','scipy','scikit-learn','einops','spectral']},
  'dataset_sha256':{d:{p.name:sha(p) for p in (Path('/home/zhangzj26/TGRS_MLUDA-2024/datasets')/d).glob('*.mat')} for d in ['Houston','Pavia']}}
 dump(OUT/'provenance.json',provenance);print('Prepared',OUT,flush=True)
if __name__=='__main__':main()
