"""Instrumentation only: retain the native update loop and target validation call."""
import ast
import hashlib
import json
import pickle
import random
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/official_bench_v1'))
import common as c

OUT=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/hyrank_A_no_extra_val_v1')
SEEDS=(2100,2101,2102)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def dump(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def state(model):
    def digest(values):
        h=hashlib.sha256()
        for name,v in values:
            h.update(name.encode());h.update(v.detach().cpu().numpy().tobytes())
        return h.hexdigest()
    return {'parameters':digest(model.named_parameters()),'buffers':digest(model.named_buffers()),
            'modes':{name:module.training for name,module in model.named_modules()}}


class IndexedDataset(c.HSIDataset):
    def __getitem__(self,index):
        x,y=super().__getitem__(index)
        return x,y,index


def loaders(args):
    # The original loader factory still creates precisely one shared generator.
    # Metadata is returned by already-requested dataset reads, not another iterator.
    previous=c.HSIDataset;c.HSIDataset=IndexedDataset
    try:return c.loaders(args)
    finally:c.HSIDataset=previous


class Loader:
    def __init__(self,base,trace,kind):self.base,self.trace,self.kind=base,trace,kind
    def __iter__(self):
        for x,y,index in self.base:
            if self.kind=='source':self.trace.source=(x,y,index)
            elif self.kind=='target':self.trace.target=(x,index)
            else:self.trace.eval_labels=y
            yield x,torch.zeros_like(y) if self.kind=='target' else y


class Trace:
    def __init__(self,model,ema,generator,device,out,source_ds,target_ds):
        self.model,self.ema,self.generator,self.device,self.out=model,ema,generator,device,out
        self.source_ds,self.target_ds=source_ds,target_ds
        self.previous_end=None;self.phase='train';self.epoch=1;self.predictions=[];self.reset()
    def rng(self):
        return {'shared_loader':sha(self.generator.get_state().numpy().tobytes()),
                'torch_cpu':sha(torch.get_rng_state().numpy().tobytes()),
                'torch_cuda':sha(torch.cuda.get_rng_state(self.device).numpy().tobytes()),
                'python':sha(pickle.dumps(random.getstate())),
                'numpy':sha(pickle.dumps(np.random.get_state()))}
    def reset(self):
        self.batch=hashlib.sha256();self.old_rng=hashlib.sha256();self.rng_digest=hashlib.sha256()
        self.indices=hashlib.sha256();self.ce=0.;self.steps=0;self.count=0;self.teacher_calls=0
        self.first_forward=None;self.last_forward=None
    def pre(self,model,inputs):
        if self.phase=='validation':
            assert not model.training
            return
        assert self.phase=='train' and all(m.training for m in model.modules())
        if self.steps==0:
            current=state(model)
            if self.previous_end is not None:
                assert current['parameters']==self.previous_end['student']['parameters']
                assert current['buffers']==self.previous_end['student']['buffers']
                assert state(self.ema)==self.previous_end['teacher']
            self.first_forward={'epoch':self.epoch,'student_all_modules_train':True,
                                'teacher_train':self.ema.training,'rng':self.rng(),
                                'state_unchanged_since_previous_epoch_end':self.previous_end is not None}
        x,y,si=self.source;t,ti=self.target
        for v in (x,y,t):self.batch.update(v.numpy().tobytes())
        self.old_rng.update(torch.get_rng_state().numpy().tobytes())
        self.old_rng.update(torch.cuda.get_rng_state(self.device).numpy().tobytes())
        rng=self.rng();self.rng_digest.update(json.dumps(rng,sort_keys=True).encode())
        src=self.source_ds.indices[si.numpy()];tar=self.target_ds.indices[ti.numpy()]
        self.indices.update(src.tobytes());self.indices.update(tar.tobytes())
        record={'epoch':self.epoch,'step_in_epoch':self.steps+1,'source_dataset_indices':si.tolist(),
                'target_dataset_indices':ti.tolist(),'source_centers_sha256':sha(src.tobytes()),
                'target_centers_sha256':sha(tar.tobytes()),'rng':rng,'student_train':model.training,
                'teacher_train':self.ema.training}
        self.batch_log.write(json.dumps(record)+'\n')
        self.last_forward=rng
    def post(self,model,inputs,outputs):
        if self.phase=='validation':
            self.logits.append(outputs[1].detach().cpu().numpy());self.labels.append(self.eval_labels.numpy())
            return
        y=self.source[1].to(self.device)
        ce=F.cross_entropy(outputs[0],y)
        assert torch.isfinite(ce)
        self.ce+=ce.item()*len(y);self.count+=len(y);self.steps+=1
    def teacher_pre(self,model,inputs):
        assert self.phase=='train' and model.training and self.model.training
        self.teacher_calls+=1
    def before_validation(self,epoch):
        assert self.steps==80 and self.teacher_calls==80
        self.phase='validation';self.logits=[];self.labels=[]
        self.eval_before={'student':state(self.model),'teacher':state(self.ema),'rng':self.rng()}
    def after_validation(self,epoch,native_acc,native_results):
        after={'student':state(self.model),'teacher':state(self.ema),'rng':self.rng()}
        for key in ('parameters','buffers'):
            assert self.eval_before['student'][key]==after['student'][key]
        assert self.eval_before['teacher']==after['teacher']
        logits=np.concatenate(self.logits).astype(np.float32,copy=False)
        labels=np.concatenate(self.labels);pred=logits.argmax(1).astype(np.int64)
        path=self.out/'predictions'/f'epoch_{epoch:03d}.npz'
        np.savez_compressed(path,logits=logits,predictions=pred)
        metrics=c.helper.metrics(labels,pred,logits,12)
        assert abs(metrics['oa_percent']/100-native_acc)<1e-10
        checkpoint=self.out/'checkpoints'/f'epoch_{epoch:03d}.pth'
        torch.save({'model':self.model.state_dict(),'epoch':epoch},checkpoint)
        self.predictions.append({'epoch':epoch,'metrics':metrics,'prediction_path':str(path),
                                 'prediction_sha256':c.helper.file_hash(path),
                                 'checkpoint_path':str(checkpoint),'checkpoint_sha256':c.helper.file_hash(checkpoint),
                                 'validation_before':self.eval_before,'validation_after':after})
        self.phase='train'
    def end_epoch(self,epoch,global_step):
        assert self.steps==80 and self.teacher_calls==80 and global_step==epoch*80
        row={'epoch':epoch,'global_step':global_step,'steps':self.steps,'source_ce':self.ce/self.count,
             'batch_stream_sha256':self.batch.hexdigest(),'rng_stream_sha256':self.old_rng.hexdigest(),
             'actual_centers_sha256':self.indices.hexdigest(),'all_rng_stream_sha256':self.rng_digest.hexdigest(),
             'first_training_forward':self.first_forward,'last_training_forward_rng':self.last_forward,
             'epoch_end_rng':self.rng(),'student_train_at_epoch_end':self.model.training,
             'teacher_train_at_epoch_end':self.ema.training,'target_validation_calls':int(epoch%10==0),
             'source_validation_calls':0,'model_calls':{'student_train':self.steps,'teacher_train':self.teacher_calls,
                                                       'student_eval':len(self.logits) if epoch%10==0 else 0}}
        if epoch%10==0:row['target']=self.predictions[-1]['metrics']
        self.history.write(json.dumps(row)+'\n');self.history.flush();self.batch_log.flush()
        if epoch==1 or epoch%10==0:print(json.dumps(row),flush=True)
        self.previous_end={'student':state(self.model),'teacher':state(self.ema)}
        self.epoch=epoch+1;self.reset()


def native(trace):
    path=c.BIDA/'train_pipeline.py'
    original=path.read_text();tree=ast.parse(original)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train')
    fn.args.args.append(ast.arg(arg='audit_trace'))
    loop=next(n for n in fn.body if isinstance(n,ast.For))
    block=next(n for n in loop.body if isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
               and isinstance(n.test.left,ast.BinOp) and isinstance(n.test.left.right,ast.Attribute)
               and n.test.left.right.attr=='log_interval')
    original_call=ast.dump(block.body[0])
    assert isinstance(block.body[0],ast.Assign) and block.body[0].value.func.id=='validation'
    # Keep the EXACT native validation assignment in the EXACT native if location.
    # Only diagnostic logging brackets it; target-best saving/control is removed.
    block.body=[ast.parse('audit_trace.before_validation(e)').body[0],block.body[0],
                ast.parse('audit_trace.after_validation(e, ts_acc, results)').body[0]]
    assert ast.dump(block.body[1])==original_call
    loop.body.append(ast.parse('audit_trace.end_epoch(e, global_step)').body[0])
    ast.fix_missing_locations(fn)
    namespace=dict(c.official.__dict__)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),namespace)
    # ast.unparse is unavailable in the original Python 3.8 environment; record
    # exact AST plus hashes rather than making a hand-written optimization loop.
    dump(trace.out/'native_loop_audit.json',{'native_source_sha256':c.helper.file_hash(path),
         'native_validation_assignment_AST':original_call,'compiled_function_AST':ast.dump(fn),
         'training_inner_loop_unchanged':True,'scheduler_then_target_validation_location_preserved':True,
         'epoch1_source_validation':False,'target_best_saving_removed_for_diagnostic_only':True})
    return namespace['train']
