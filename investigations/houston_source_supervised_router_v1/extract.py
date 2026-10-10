"""Fresh frozen source extraction; never reads target labels or imports scoring."""
import importlib.util
import json
import sys
import numpy as np
import torch
from common import HERE,OUT,BASE,DATA,GEOM,SEEDS,STAGES,AUGS,CONFIG,sha,ahash,dump

sys.dont_write_bytecode=True
sys.path.insert(0,str(BASE))
spec=importlib.util.spec_from_file_location('original_C_common',BASE/'common.py')
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)

class Patches(torch.utils.data.Dataset):
    def __init__(self,view):
        self.cube=np.load(DATA/'source_cube.npy',mmap_mode='r');self.coords=np.load(DATA/'source_coords.npy')
        self.view=view;cfg=AUGS[view];band=np.linspace(0,2*np.pi,self.cube.shape[-1])
        self.gain=cfg['scale']*(1+cfg['gain_sin']*np.sin(band)+cfg['gain_cos2']*np.cos(2*band))
        self.offset=cfg['offset_cos']*np.cos(band)+cfg['offset_sin']*np.sin(band)
    def __len__(self):return len(self.coords)
    def __getitem__(self,i):
        x,y=self.coords[i];p=np.array(self.cube[x-6:x+7,y-6:y+7],dtype=np.float32)
        if self.view:
            rng=np.random.default_rng(20261011+10000*self.view+i)
            p=(p*self.gain+self.offset+rng.normal(0,AUGS[self.view]['noise'],p.shape)).astype(np.float32)
        return torch.from_numpy(p.transpose(2,0,1)).unsqueeze(0)

@torch.inference_mode()
def extract(model,folder,seed):
    latest={};hook=model.conv2d_features.register_forward_hook(lambda m,args,out:latest.update(V=out))
    method=model._forward_semantic_tokens
    def tokenize(v):
        t=method(v);latest['T']=t;return t
    model._forward_semantic_tokens=tokenize
    audit={}
    try:
        for view,cfg in enumerate(AUGS):
            loader=torch.utils.data.DataLoader(Patches(view),batch_size=128,shuffle=False,num_workers=0)
            arrays={};pos=0
            for patch in loader:
                assert not model.training and all(not m.training for m in model.modules())
                assert all(not p.requires_grad for p in model.parameters())
                patch=patch.cuda();result=model(patch,patch,return_feat_prob=True);logits,z=result[1],result[3]
                values={s:torch.nn.functional.normalize(v.flatten(1),dim=1).cpu().numpy() for s,v in [('V',latest['V']),('T',latest['T']),('Z',z)]}
                values.update(rawZ=z.cpu().numpy(),logits=logits.cpu().numpy())
                for key,value in values.items():
                    assert np.isfinite(value).all()
                    if key not in arrays:arrays[key]=np.lib.format.open_memmap(folder/f'view{view}_{key}.npy',mode='w+',dtype=value.dtype,shape=(2520,*value.shape[1:]))
                    arrays[key][pos:pos+len(patch)]=value
                pos+=len(patch)
            assert pos==2520
            for a in arrays.values():a.flush()
            if view==0:
                errors={}
                for key in ['V','T','Z','logits']:
                    old=np.load(GEOM/f'seed_{seed}/source_{key}.npy');err=float(abs(arrays[key]-old).max())
                    errors[key]=err;assert err<2e-5,(key,err)
                assert np.array_equal(arrays['logits'].argmax(1),np.load(GEOM/f'seed_{seed}/source_predictions.npy'))
                audit['identity_replay_max_errors']=errors
            print(seed,'source',cfg['name'],'2520 frozen features extracted',flush=True)
    finally:hook.remove();model._forward_semantic_tokens=method
    return audit

def main():
    OUT.mkdir(exist_ok=False);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    metadata=json.loads((DATA.parent/'fixed_inputs.json').read_text())
    gate=json.loads((GEOM/'ALL_FEATURES_PREDICTIONS_FROZEN.json').read_text())
    prov={'historical_metadata':metadata,'fixed_config':CONFIG,'source_only_extraction':True,
          'input_hashes':{str(DATA/f'source_{key}.npy'):sha(DATA/f'source_{key}.npy') for key in ['cube','coords','labels','probe_train_mask']},'seeds':{}}
    coords=np.load(DATA/'source_coords.npy');train=np.load(DATA/'source_probe_train_mask.npy');sy=np.load(DATA/'source_labels.npy')
    assert train.sum()==2393 and (~train).sum()==127
    for name,value in [('source_coords',coords),('source_train_mask',train),('source_labels',sy)]:np.save(OUT/f'{name}.npy',value)
    for seed in SEEDS:
        folder=OUT/f'seed_{seed}';folder.mkdir();cp=BASE/'houston'/f'C_{seed}/checkpoints/epoch_200.pth'
        assert sha(cp)==metadata['checkpoints'][str(seed)]['sha256']
        cfg=json.loads((cp.parent.parent/'config.json').read_text());assert cfg['arm']=='C' and cfg['epochs']==200
        assert ahash(coords[train])==cfg['indices_sha256']['source'] and ahash(coords[~train])==cfg['indices_sha256']['val']
        for key in ['V','T','Z','logits','predictions']:
            rel=f'seed_{seed}/source_{key}.npy';assert sha(GEOM/rel)==gate['files_sha256'][rel]
        original.environment()[4](seed);model,_=original.build(seed,'C','houston')
        model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True)['model'],strict=True)
        model.cuda().eval().requires_grad_(False);state=original.state_hash(model)
        audit=extract(model,folder,seed)
        assert original.state_hash(model)==state and sha(cp)==metadata['checkpoints'][str(seed)]['sha256']
        prov['seeds'][str(seed)]={'checkpoint':str(cp),'checkpoint_sha256':sha(cp),'state_sha256':state,'main_model_unchanged':True,'audit':audit}
        del model;torch.cuda.empty_cache()
    dump(OUT/'extraction_provenance.json',prov)
    dump(OUT/'ALL_SOURCE_FEATURES_FROZEN.json',{'target_GT_opened':False,'all_seeds_completed':True,
        'files_sha256':{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*.npy')}})
    print('All source identity/shift features frozen; main model unchanged; target GT unopened.',flush=True)

if __name__=='__main__':main()
