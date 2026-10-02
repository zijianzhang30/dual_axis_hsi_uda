"""Frozen target angular probes before/after the A1 class reader."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import torch
import train

ROOT=Path(__file__).resolve().parents[2]

def stats(values,model):
    a=torch.cat(values).double()
    norms=a.norm(dim=-1)
    unit=torch.nn.functional.normalize(a,dim=-1)
    weights=torch.nn.functional.normalize(model.nn1.weight.detach().cpu().double(),dim=-1)
    return {'mean_norm':float(norms.mean()),'norm_sd':float(norms.std()),
            'direction_concentration':float(unit.mean(0).norm()),
            'classifier_cosine_mean':(unit@weights.T).mean(0).tolist()}

@torch.no_grad()
def measure(model,images,device):
    model.eval();values={};logits=[]
    def hook(key):
        def record(module,inputs,output):
            if key=='spatial_tokens':
                vector=output[0].mean(1)
                values.setdefault('spatial_all_tokens',[]).append(output[0].flatten(0,1).cpu())
            elif key=='class_reader':vector=output[0][:,0]
            else:vector=inputs[0]
            values.setdefault(key,[]).append(vector.cpu())
        return record
    handles=[model.spatial_reader.register_forward_hook(hook('spatial_tokens')),
             model.class_reader.register_forward_hook(hook('class_reader')),
             model.nn1.register_forward_hook(hook('classifier_z'))]
    try:
        for start in range(0,len(images),128):
            x=images[start:start+128].to(device)
            logits.append(model(x,x)[1].cpu())
    finally:
        for handle in handles:handle.remove()
    report={key:stats(v,model) for key,v in values.items()}
    z=torch.cat(logits).double()
    margin=z[:,5]-torch.cat((z[:,:5],z[:,6:]),1).max(1).values
    report['mean_logit']=z.mean(0).tolist()
    report['class6_margin_mean']=float(margin.mean())
    report['prediction_shares']=(torch.bincount(z.argmax(1),minlength=7)/len(z)).tolist()
    return report

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',required=True)
    parser.add_argument('--seeds',type=int,nargs='+',choices=(2100,2101,2102),required=True)
    args=parser.parse_args();torch.set_num_threads(2)
    device=torch.device('cuda:'+args.device)
    opts=SimpleNamespace(seed=2100,num_workers=0,dataset_dir='/home/zhangzj26/TGRS_MLUDA-2024/datasets/Houston/',num_tokens=4,dim=64,depth=3)
    _,_,_,target,source,_=train.make_loaders(opts)
    sample=json.loads((ROOT/'results/joint_tokenizer_probe/seed_2101.json').read_text())
    images=torch.stack([target.dataset[i][0] for i in sample['target_indices']])
    source_images=torch.stack([source.dataset[i][0] for i in sample['source_indices']])
    import hashlib
    assert hashlib.sha256(source_images.numpy().tobytes()+images.numpy().tobytes()).hexdigest()==sample['sample_sha256']
    out=ROOT/'results/query_class_readout_v1/probes';out.mkdir(parents=True,exist_ok=True)
    for seed in args.seeds:
        path=ROOT/f'results/query_class_readout_v1/formal_{seed}'
        dest=out/f'seed_{seed}.json'
        if dest.exists():raise RuntimeError(f'Existing probe requires inspection: {dest}')
        manifest=json.loads((path/'frozen_manifest.json').read_text())
        result=json.loads((path/'results.json').read_text())
        assert train.helper.file_hash(path/'frozen_manifest.json')==result['frozen_manifest_sha256']
        model=train.get_model('BiDA','Houston13',13,opts)
        train.transplant(model);train.install_readout(model,train.a1);train.joint.install(model,'fixed_mixture');model.to(device)
        rows=[]
        for item in manifest['checkpoints']:
            assert train.helper.file_hash(item['path'])==item['sha256']
            model.load_state_dict(torch.load(item['path'],map_location=device,weights_only=True)['model'],strict=True)
            before=train.helper.state_hash(model)
            row=measure(model,images,device)
            assert train.helper.state_hash(model)==before
            rows.append({'epoch':item['epoch'],'checkpoint_sha256':item['sha256'],'target':row})
            print(json.dumps({'seed':seed,'epoch':item['epoch'],'z_concentration':row['classifier_z']['direction_concentration'],'cos6':row['classifier_z']['classifier_cosine_mean'][5]}),flush=True)
        dest.write_text(json.dumps({'seed':seed,'sample_sha256':sample['sample_sha256'],'probe_script_sha256':train.helper.file_hash(Path(__file__)),'rows':rows},indent=2)+'\n')
        del model

if __name__=='__main__':main()
