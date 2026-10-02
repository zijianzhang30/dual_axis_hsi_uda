"""HyRANK fixed200 A/native full, B/joint-self, C/joint center relations."""
import argparse
import hashlib
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
import common as c


class Trace:
    def reset(self):
        self.batches=hashlib.sha256();self.rng=hashlib.sha256()
        self.ce=self.count=self.steps=0
    def pre(self,model,inputs):
        if not model.training:return
        for value in (*self.source,self.target):self.batches.update(value.numpy().tobytes())
        self.rng.update(torch.get_rng_state().numpy().tobytes())
        self.rng.update(torch.cuda.get_rng_state(self.device).numpy().tobytes())
    def post(self,model,inputs,outputs):
        if not model.training:return
        y=self.source[1].to(self.device)
        self.ce+=F.cross_entropy(outputs[0],y).item()*len(y)
        self.count+=len(y);self.steps+=1


class TrainingLoader:
    def __init__(self,loader,trace,target=False):self.loader,self.trace,self.target=loader,trace,target
    def __iter__(self):
        for x,y in self.loader:
            if self.target:
                self.trace.target=x
                yield x,torch.zeros_like(y) # Native A sees no actual target-label values.
            else:
                self.trace.source=(x,y)
                yield x,y


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=('A','B','C'),required=True)
    p.add_argument('--seed',type=int,choices=(2100,2101,2102),required=True)
    p.add_argument('--device',type=int,required=True);a=p.parse_args()
    args=c.options(a.seed)
    out=c.OUT/f'hyrank_{a.arm}_{a.seed}'
    if out.exists():raise RuntimeError('Refuse overwrite')
    assert json.loads((c.OUT/'implementation_checks.json').read_text())['all_passed']
    torch.set_num_threads(2)
    device=torch.device(f'cuda:{a.device}');torch.cuda.set_device(device)
    c.seed_worker(a.seed)
    loaders,data=c.loaders(args);source,val,target,target_eval=loaders
    model,ema,initial=c.build(args,a.arm)
    config={'benchmark':'Dioni -> Loukia (author out68)','arm':a.arm,'seed':a.seed,
            'primary':'fixed_epoch_200','settings':vars(args),'data_protocol':data,
            'initial_common_model_sha256':initial,'initial_complete_state_sha256':c.helper.state_hash(model),
            'parameters':sum(p.numel() for p in model.parameters()),'device':str(device),
            'bn':'native sequential' if a.arm=='A' else 'Full Joint',
            'objective':'released full BiDA loss expressions' if a.arm=='A' else 'source-self CE only',
            'bn_momentum':.1 if a.arm=='A' else .19,
            'normalization_provenance':'actual released main.py normband call and Dioni/Loukia loader branches',
            'configuration_provenance':{'explicit_generic_released_defaults':['ratio','bs','lr','patch_size','dim','depth','num_tokens','lambda1','lambda2','ema_decay','re_ratio','epoch'],
                  'uniform_inherited':['dropout0.1','source-val epoch1 then every10','fixed200 no target selection'],
                  'HyRANK_specific_paper_overrides':'not documented; not assumed'},
            'code_sha256':{str(path):c.helper.file_hash(path) for path in
              [c.BIDA/'main.py',c.BIDA/'train_pipeline.py',c.BIDA/'models/BiDA.py',c.BIDA/'utils/dataset.py',
               c.BIDA/'utils/scheduler.py',c.ROOT/'experiments/relation_tokenizer_v1/relation.py',
               c.ROOT/'experiments/bn_stabilization_v1/normalization.py',c.ROOT/'experiments/official_bench_v1/PROTOCOL.md',
               c.ROOT/'experiments/official_bench_v1/common.py',c.ROOT/'experiments/official_bench_v1/train.py']}}
    out.mkdir();(out/'checkpoints').mkdir()
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    print(json.dumps(config),flush=True)
    model.to(device)
    if ema is not None:ema.to(device)
    optimizer,scheduler=c.load_scheduler('BiDA',model,args);assert scheduler is None
    criterion,_=c.make_loss(args,num_classes=12)
    trace=Trace();trace.device=device;trace.reset()
    hooks=[model.register_forward_pre_hook(trace.pre),model.register_forward_hook(trace.post)]
    training_source=TrainingLoader(source,trace);training_target=TrainingLoader(target,trace,True)
    checkpoints=[]
    torch.cuda.synchronize(device);start=time.monotonic()
    with (out/'history.jsonl').open('w') as history:
        def callback(epoch,network,teacher,opt,global_step):
            assert trace.count>0 and trace.steps>0
            row={'epoch':epoch,'steps':trace.steps,'source_ce':trace.ce/trace.count,
                 'batch_stream_sha256':trace.batches.hexdigest(),'rng_stream_sha256':trace.rng.hexdigest(),
                 'global_step':global_step}
            if epoch==1 or epoch%10==0:
                row['source_val']=c.helper.source_evaluate(network,val,device)
                print(json.dumps(row),flush=True)
            if epoch%10==0:
                path=out/'checkpoints'/f'epoch_{epoch:03d}.pth'
                torch.save({'model':network.state_dict(),'epoch':epoch,'config':config},path)
                checkpoints.append({'epoch':epoch,'path':str(path),'sha256':c.helper.file_hash(path)})
            history.write(json.dumps(row)+'\n');history.flush();trace.reset()
            return row
        if a.arm=='A':
            native,diff=c.native_train()
            (out/'official_loop_diff.patch').write_text(diff)
            native(model,ema,optimizer,criterion,12,training_source,val,training_target,
                   target_eval,args,str(out),device,scheduler,callback)
        else:
            global_step=0
            for epoch in range(1,201):
                model.train()
                for (x,y),(t,_) in zip(training_source,training_target):
                    if len(x)!=len(t):continue
                    x,y,t=x.to(device),y.to(device),t.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss=F.cross_entropy(model(x,t)[0],y);assert torch.isfinite(loss)
                    loss.backward();optimizer.step();global_step+=1
                callback(epoch,model,None,optimizer,global_step)
    torch.cuda.synchronize(device);seconds=time.monotonic()-start
    for hook in hooks:hook.remove()
    start=time.monotonic();logits,predictions=c.helper.predict(model,target_eval,device)
    pred_path=out/'predictions_epoch_200.npz';np.savez_compressed(pred_path,logits=logits,predictions=predictions)
    h=[json.loads(s) for s in (out/'history.jsonl').read_text().splitlines()]
    manifest={'config':config,'checkpoints':checkpoints,
              'prediction':{'path':str(pred_path),'sha256':c.helper.file_hash(pred_path)},
              'train_seconds':seconds,'inference_seconds':time.monotonic()-start,
              'source_val_at_final':h[-1]['source_val']}
    (out/'frozen_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'frozen_manifest_sha256':c.helper.file_hash(out/'frozen_manifest.json')}),flush=True)


if __name__=='__main__':main()
