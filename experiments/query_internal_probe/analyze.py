"""Non-invasive internal spatial reader hooks with offset-aware geometry."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('factorial_probe_helpers',ROOT/'experiments/tokenizer_readout_v1/train.py')
# This helper imports its local pipeline.
import sys
sys.path.insert(0,str(ROOT/'experiments/tokenizer_readout_v1'))
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)

class Stage:
    def __init__(self):self.n=0;self.unit_sum=None;self.sum=None;self.sq=None;self.pooled=[];self.selected=[]
    def add(self,value):
        x=value.detach().float().cpu()
        if x.ndim==2:x=x[:,None,:]
        unit=torch.nn.functional.normalize(x,dim=-1)
        total=x.double().sum(0);sq=x.double().square().sum(0)
        self.sum=total if self.sum is None else self.sum+total
        self.sq=sq if self.sq is None else self.sq+sq
        u=unit.double().sum(0);self.unit_sum=u if self.unit_sum is None else self.unit_sum+u
        self.n+=len(x);self.pooled.append(x.mean(1))
        points=torch.linspace(0,x.shape[1]-1,min(5,x.shape[1])).round().long().unique()
        self.selected.append(x[:,points])
    def finish(self):
        p=torch.cat(self.pooled).double();selected=torch.cat(self.selected).double()
        mean=self.sum/self.n;var=(self.sq/self.n-mean.square()).clamp_min(0)
        raw=float((self.sq/self.n).sum(-1).mean().sqrt());center=float(var.sum(-1).mean().sqrt())
        centered=p-p.mean(0);cov=centered.T@centered/(len(p)-1)
        trace=float(cov.trace());denom=float(cov.square().sum())
        unit=torch.nn.functional.normalize(p,dim=-1)
        return {'positions':self.sum.shape[0],'channels':p.shape[1],
                'pooled_mean_norm':float(p.norm(dim=-1).mean()),
                'pooled_direction_concentration':float(unit.mean(0).norm()),
                'mean_position_concentration':float((self.unit_sum/self.n).norm(dim=-1).mean()),
                'centered_rms':center,'raw_rms':raw,'centered_fraction':center/max(raw,1e-12),
                'pooled_centered_rms':float(centered.square().sum(-1).mean().sqrt()),
                'covariance_participation_rank':trace*trace/denom if denom>1e-18 else 0.0},p,selected

def distribution(x):
    x=x.flatten().double()
    return {'mean':float(x.mean()),'sample_sd':float(x.std()),
            'quantiles':torch.quantile(x,torch.tensor([.05,.25,.5,.75,.95],dtype=torch.float64)).tolist(),
            'histogram':torch.histc(x.float(),bins=20,min=-1,max=1).long().tolist()}

class Capture:
    def __init__(self,model):
        self.values={};self.handles=[]
        def image(key,flatten_bands=False):
            def hook(module,inputs,output):
                x=output
                if flatten_bands:x=x.flatten(1,2)
                self.values[key]=x.flatten(2).transpose(1,2)
            return hook
        def pre(key,index=0):
            def hook(module,inputs):self.values[key]=inputs[index]
            return hook
        def output(key,tuple_output=False):
            def hook(module,inputs,out):self.values[key]=out[0] if tuple_output else out
            return hook
        reader=model.spatial_reader
        self.handles=[model.conv3d_features[2].register_forward_hook(image('stem3d',True)),
                      model.conv2d_features[0].register_forward_hook(image('projection_pre_bn')),
                      model.conv2d_features[1].register_forward_hook(image('bn2d')),
                      model.conv2d_features[2].register_forward_hook(image('stem_relu')),
                      reader.register_forward_pre_hook(pre('plus_pos',1)),
                      reader.q_norm.register_forward_hook(output('fixed_query_norm')),
                      reader.kv_norm.register_forward_hook(output('kv_norm')),
                      reader.attn.register_forward_hook(output('attention_delta',True)),
                      reader.ff_norm.register_forward_pre_hook(pre('attention_residual')),
                      reader.ff_norm.register_forward_hook(output('ffn_ln')),
                      reader.ff.register_forward_hook(output('ffn_delta')),
                      reader.register_forward_hook(output('ffn_residual',True)),
                      model.nn1.register_forward_pre_hook(pre('classifier_z'))]
    def close(self):
        for h in self.handles:h.remove()

@torch.no_grad()
def measure(model,images,device):
    model.eval();stages={d:{} for d in images};capture=Capture(model)
    try:
        for domain,batches in images.items():
            for start in range(0,len(batches),128):
                capture.values={};x=batches[start:start+128].to(device);model(x,x)
                for key,value in capture.values.items():stages[domain].setdefault(key,Stage()).add(value)
    finally:capture.close()
    result={d:{} for d in images};cosines={};rng=np.random.default_rng(7319)
    pairs={key:(torch.from_numpy(rng.integers(0,2048,4096)),torch.from_numpy(rng.integers(0,2048,4096))) for key in ('source_source','target_target','source_target')}
    # Exclude within-domain self-pairs.
    for key in ('source_source','target_target'):
        i,j=pairs[key];pairs[key]=(i,torch.where(j==i,(j+1)%2048,j))
    weight=torch.nn.functional.normalize(model.nn1.weight.detach().cpu().double(),dim=-1)
    for key in stages['source']:
        vectors={};selected={}
        for domain in images:
            report,v,s=stages[domain][key].finish();vectors[domain]=v;selected[domain]=s
            if v.shape[-1]==64:report['cos_w6_mean']=float((torch.nn.functional.normalize(v,dim=-1)@weight.T)[:,5].mean())
            result[domain][key]=report
        cosines[key]={}
        for name,(i,j) in pairs.items():
            left,right=name.split('_')
            a,b=vectors[left][i],vectors[right][j]
            cosines[key][name]={'pooled':distribution(torch.nn.functional.cosine_similarity(a,b,dim=-1)),
                                'same_coordinate':distribution(torch.nn.functional.cosine_similarity(selected[left][i],selected[right][j],dim=-1))}
        for domain in images:
            before=result[domain]['stem_relu']['centered_rms'] if 'stem_relu' in result[domain] else None
            if key=='plus_pos':
                assert abs(result[domain][key]['centered_rms']-before)<1e-4
    return {'domains':result,'cosine_distributions':cosines}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',required=True)
    parser.add_argument('--arms',nargs='+',choices=('query_class','query_mean'),required=True)
    args=parser.parse_args();torch.set_num_threads(2);device=torch.device('cuda:'+args.device)
    opts=SimpleNamespace(seed=2100,num_workers=0,dataset_dir='/home/zhangzj26/TGRS_MLUDA-2024/datasets/Houston/',num_tokens=4,dim=64,depth=3)
    _,_,_,target,source,_=base.make_loaders(opts)
    sample=json.loads((ROOT/'results/joint_tokenizer_probe/seed_2101.json').read_text())
    images={'source':torch.stack([source.dataset[i][0] for i in sample['source_indices']]),'target':torch.stack([target.dataset[i][0] for i in sample['target_indices']])}
    import hashlib
    assert hashlib.sha256(images['source'].numpy().tobytes()+images['target'].numpy().tobytes()).hexdigest()==sample['sample_sha256']
    out=ROOT/'results/query_internal_probe';out.mkdir(parents=True,exist_ok=True)
    for arm in args.arms:
        dest=out/f'{arm}_2101.json'
        if dest.exists():raise RuntimeError(f'Existing probe requires inspection: {dest}')
        folder=ROOT/'results/query_class_readout_v1/formal_2101' if arm=='query_class' else ROOT/'results/tokenizer_readout_v1/query_mean_2101'
        manifest=json.loads((folder/'frozen_manifest.json').read_text());result=json.loads((folder/'results.json').read_text())
        assert base.helper.file_hash(folder/'frozen_manifest.json')==result['frozen_manifest_sha256']
        model=base.get_model('BiDA','Houston13',13,opts)
        conv=copy.deepcopy(model.conv_a);semantic=model._forward_semantic_tokens.__func__
        base.transplant(model);base.install_readout(model,base.a1);base.configure_cell(model,arm,conv,semantic);base.joint.install(model,'fixed_mixture');model.to(device)
        # Verify hooks on a small unmodified forward before any stage measurement.
        model.eval();x=images['target'][:128].to(device)
        with torch.no_grad():
            plain=model(x,x)[1];capture=Capture(model);hooked=model(x,x)[1];capture.close();assert torch.equal(plain,hooked)
        rows=[]
        for item in manifest['checkpoints']:
            assert base.helper.file_hash(item['path'])==item['sha256']
            model.load_state_dict(torch.load(item['path'],map_location=device,weights_only=True)['model'],strict=True)
            before=base.helper.state_hash(model);report=measure(model,images,device);assert base.helper.state_hash(model)==before
            rows.append({'epoch':item['epoch'],'checkpoint_sha256':item['sha256'],'report':report})
            print(json.dumps({'arm':arm,'epoch':item['epoch'],'stages':{k:round(v['pooled_direction_concentration'],3) for k,v in report['domains']['target'].items()}}),flush=True)
        dest.write_text(json.dumps({'arm':arm,'seed':2101,'sample_sha256':sample['sample_sha256'],'protocol_sha256':base.helper.file_hash(Path(__file__).with_name('PROTOCOL.md')),'script_sha256':base.helper.file_hash(Path(__file__)),'rows':rows},indent=2)+'\n');del model

if __name__=='__main__':main()
