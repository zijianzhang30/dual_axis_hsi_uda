"""Compare angular representations using frozen eval and current joint moments."""
import argparse
import copy
import json
from pathlib import Path
from types import MethodType,SimpleNamespace
import torch
import train

ROOT=Path(__file__).resolve().parents[2]

def folder(arm):
    return ROOT/'results/query_class_readout_v1/formal_2101' if arm=='query_class' else ROOT/f'results/tokenizer_readout_v1/{arm}_2101'

def vector_report(values,model):
    x=torch.cat(values).double();norm=x.norm(dim=-1)
    unit=torch.nn.functional.normalize(x,dim=-1)
    w=torch.nn.functional.normalize(model.nn1.weight.detach().cpu().double(),dim=-1)
    return {'mean_norm':float(norm.mean()),'direction_concentration':float(unit.mean(0).norm()),'cos_w6_mean':float((unit@w.T)[:,5].mean())}

@torch.no_grad()
def measure(model,images,mode,device):
    model.eval() if mode=='eval' else model.train()
    for module in model.modules():
        if isinstance(module,torch.nn.Dropout):module.eval()
    values={d:{k:[] for k in ('token_mean','pre_norm','z','logits')} for d in images}
    current={'domain':'source','calls':0}
    original=model._classify_tokens
    def classify(self,tokens):
        if mode=='joint_train':current['domain']='source' if current['calls']==0 else 'target'
        current['calls']+=1
        values[current['domain']]['token_mean'].append(tokens.mean(1).cpu())
        return original(tokens)
    def norm_hook(module,inputs):values[current['domain']]['pre_norm'].append(inputs[0].cpu())
    def head_hook(module,inputs,output):
        values[current['domain']]['z'].append(inputs[0].cpu())
        values[current['domain']]['logits'].append(output.cpu())
    model._classify_tokens=MethodType(classify,model)
    handles=[model.norm.register_forward_pre_hook(norm_hook),model.nn1.register_forward_hook(head_hook)]
    try:
        for start in range(0,len(images['source']),128):
            xs,xt=(images[d][start:start+128].to(device) for d in ('source','target'))
            current['calls']=0
            if mode=='joint_train':model(xs,xt)
            else:
                for domain,x in (('source',xs),('target',xt)):
                    current['domain']=domain;model(x,x)
    finally:
        model._classify_tokens=original
        for handle in handles:handle.remove()
    result={}
    for domain,v in values.items():
        result[domain]={key:vector_report(v[key],model) for key in ('token_mean','pre_norm','z')}
        logits=torch.cat(v['logits']).double()
        margin=logits[:,5]-torch.cat((logits[:,:5],logits[:,6:]),1).max(1).values
        result[domain]['margin6_mean']=float(margin.mean())
        result[domain]['prediction_shares']=(torch.bincount(logits.argmax(1),minlength=7)/len(logits)).tolist()
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',required=True)
    parser.add_argument('--arms',nargs='+',choices=('query_class','bida_class','query_mean','bida_mean'),required=True)
    args=parser.parse_args();torch.set_num_threads(2);device=torch.device('cuda:'+args.device)
    opts=SimpleNamespace(seed=2100,num_workers=0,dataset_dir='/home/zhangzj26/TGRS_MLUDA-2024/datasets/Houston/',num_tokens=4,dim=64,depth=3)
    _,_,_,target,source,_=train.make_loaders(opts)
    sample=json.loads((ROOT/'results/joint_tokenizer_probe/seed_2101.json').read_text())
    images={'source':torch.stack([source.dataset[i][0] for i in sample['source_indices']]),'target':torch.stack([target.dataset[i][0] for i in sample['target_indices']])}
    import hashlib
    assert hashlib.sha256(images['source'].numpy().tobytes()+images['target'].numpy().tobytes()).hexdigest()==sample['sample_sha256']
    out=ROOT/'results/tokenizer_readout_v1/probes';out.mkdir(parents=True,exist_ok=True)
    for arm in args.arms:
        dest=out/f'{arm}.json'
        if dest.exists():raise RuntimeError(f'Existing probe requires inspection: {dest}')
        path=folder(arm);manifest=json.loads((path/'frozen_manifest.json').read_text());res=json.loads((path/'results.json').read_text())
        assert train.helper.file_hash(path/'frozen_manifest.json')==res['frozen_manifest_sha256']
        model=train.get_model('BiDA','Houston13',13,opts)
        conv=copy.deepcopy(model.conv_a);semantic=model._forward_semantic_tokens.__func__
        train.transplant(model);train.install_readout(model,train.a1);train.configure_cell(model,arm,conv,semantic);train.joint.install(model,'fixed_mixture');model.to(device)
        rows=[]
        for item in manifest['checkpoints']:
            assert train.helper.file_hash(item['path'])==item['sha256']
            state=torch.load(item['path'],map_location=device,weights_only=True)['model'];model.load_state_dict(state,strict=True)
            before=train.helper.state_hash(model);modes={}
            for mode in ('eval','joint_train'):
                model.load_state_dict(state,strict=True);modes[mode]=measure(model,images,mode,device)
                model.load_state_dict(state,strict=True);assert train.helper.state_hash(model)==before
            rows.append({'epoch':item['epoch'],'checkpoint_sha256':item['sha256'],'modes':modes})
            print(json.dumps({'arm':arm,'epoch':item['epoch'],'target_concentration':modes['eval']['target']['z']['direction_concentration'],'target_cos6':modes['eval']['target']['z']['cos_w6_mean']}),flush=True)
        dest.write_text(json.dumps({'arm':arm,'seed':2101,'sample_sha256':sample['sample_sha256'],'script_sha256':train.helper.file_hash(Path(__file__)),'rows':rows},indent=2)+'\n');del model

if __name__=='__main__':main()
