"""All saved checkpoints for one seed; predictions first, metrics second."""
import argparse
import json
from pathlib import Path
from types import MethodType
import time
import numpy as np
import torch
from scipy.io import loadmat
import common as t


def probe(model,values,device):
    captured={}
    def feature_hook(module,args,result):
        captured['stem_center']=result[:,:,6,6]
        captured['stem_mean']=result.mean((-2,-1))
    handle=model.conv2d_features[2].register_forward_hook(feature_hook)
    old_semantic=model._forward_semantic_tokens
    semantic=old_semantic.__func__
    def tokens(self,features):
        result=semantic(self,features)
        captured['token_mean']=result.mean(1)
        return result
    model._forward_semantic_tokens=MethodType(tokens,model)
    names=('stem_center','stem_mean','token_mean','classifier_z')
    stats={};target_predictions=[]
    before=t.c.helper.state_hash(model)
    try:
        for domain,loader in (('source_val',values[1]),('target_eval',values[3])):
            sums={key:np.zeros((7,64),dtype=np.float64) for key in names}
            counts=np.zeros(7,dtype=np.int64)
            norm_sums={key:np.zeros(7,dtype=np.float64) for key in names}
            with torch.inference_mode():
                for images,labels in loader:
                    images=images.to(device);labels=labels.numpy()
                    output=model(images,images,return_feat_prob=True)
                    captured['classifier_z']=output[3]
                    counts+=np.bincount(labels,minlength=7)
                    for key in names:
                        f=captured[key].cpu().numpy().astype(np.float64)
                        np.add.at(sums[key],labels,f)
                        np.add.at(norm_sums[key],labels,np.linalg.norm(f,axis=1))
                    if domain=='target_eval':target_predictions.append(output[1].argmax(1).cpu().numpy())
            stats[domain]={'counts':counts.tolist(),
                          'mean_features':{key:(sums[key]/counts[:,None]).tolist() for key in names},
                          'mean_norms':{key:(norm_sums[key]/counts).tolist() for key in names}}
    finally:
        handle.remove();model._forward_semantic_tokens=old_semantic
    assert before==t.c.helper.state_hash(model)
    similarities={}
    for key in names:
        s=np.asarray(stats['source_val']['mean_features'][key]);q=np.asarray(stats['target_eval']['mean_features'][key])
        s/=np.maximum(np.linalg.norm(s,axis=1,keepdims=True),1e-12)
        q/=np.maximum(np.linalg.norm(q,axis=1,keepdims=True),1e-12)
        cosine=q@s.T
        similarities[key]={'target_vs_source_class_mean_cosine':cosine.tolist(),
                          'same_class_cosine':np.diag(cosine).tolist(),
                          'nearest_source_class':(cosine.argmax(1)+1).tolist()}
    return {'class_conditional_statistics':stats,'stage_similarities':similarities,
            'all_parameters_buffers_unchanged':True,
            'definitions':'existing stages.py: center(6,6), CNN mean, token mean, final normalized CLS'},np.concatenate(target_predictions)


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=t.SEEDS,required=True)
    p.add_argument('--device',type=int,default=0);args=p.parse_args()
    folder=t.OUT/('seed_%d'%args.seed);folder.mkdir(exist_ok=False)
    torch.set_num_threads(2);torch.cuda.set_device(args.device)
    device=torch.device('cuda:%d'%args.device)
    inventory=json.loads((t.OUT/'inventory.json').read_text())
    assert all(t.c.helper.file_hash(Path(path))==value for path,value in inventory['source_sha256'].items())
    t.c.pavia.seed_worker(args.seed)
    values=t.c.loaders('pavia',args.seed,4)
    audit=t.c.inputs_audit('pavia',values)
    reference=inventory['runs']['B_%d'%args.seed]['config']
    assert audit['indices_sha256']==reference['indices_sha256']
    assert audit['data_sha256']==reference['data_sha256']
    assert audit['sample_counts']==list(reference['sample_counts'].values())
    target=values[3].dataset
    labels=np.asarray(target.labels,dtype=np.int64)
    raw=loadmat(t.c.DATA/'Pavia/pavia_gt_7.mat')['pavia_gt_7'].astype(int)-1
    assert np.array_equal(labels,raw[target.indices[:,0]-6,target.indices[:,1]-6])
    assert len(labels)==39348
    # Independent fixed pixel-order check and real normalized center check.
    centers=target.data[target.indices[:128,0],target.indices[:128,1]]
    lengths=np.linalg.norm(centers,axis=1)
    assert np.all((lengths==0)|np.isclose(lengths,1,atol=1e-6))
    t.c.dump(folder/'input_audit.json',dict(audit,fixed_eval_n=len(labels),
                  physical_gpu=__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),
                  checkpoint_analysis_only=True))
    predictions={};missing=[];started=time.monotonic()
    for arm in t.ARMS:
        run=inventory['runs']['%s_%d'%(arm,args.seed)]
        t.c.pavia.seed_worker(args.seed)
        model,_=t.c.build('pavia',arm,args.seed);model.to(device).eval()
        for record in run['checkpoints']:
            epoch=record['epoch']
            if record['status']=='missing':missing.append(dict(record,arm=arm));continue
            cp=Path(record['path'])
            if not cp.exists():missing.append(dict(record,arm=arm,status='missing_since_inventory'));continue
            assert t.c.helper.file_hash(cp)==record['sha256']
            checkpoint=torch.load(cp,map_location='cpu')
            assert checkpoint['config']==run['config'] and checkpoint['epoch']==epoch
            model.load_state_dict(checkpoint['model']);model.eval();del checkpoint
            state=t.c.helper.state_hash(model)
            begin=time.monotonic()
            logits,pred=t.c.helper.predict(model,values[3],device)
            assert state==t.c.helper.state_hash(model)
            name='%s_epoch_%03d'%(arm,epoch);path=folder/(name+'.npz')
            np.savez_compressed(path,logits=logits,predictions=pred)
            row={'arm':arm,'epoch':epoch,'checkpoint_sha256':record['sha256'],
                 'prediction_path':str(path),'prediction_sha256':t.c.helper.file_hash(path),
                 'inference_seconds':time.monotonic()-begin,'state_unchanged':True}
            if epoch==200:
                original=t.c.OUT/('%s_%d'%(arm,args.seed))/'predictions_epoch_200.npz'
                assert t.c.helper.file_hash(original)==run['final_prediction_sha256']
                with np.load(original) as old:assert np.array_equal(pred,old['predictions'])
                row['matches_original_fixed200_predictions']=True
            predictions[name]=row
            t.c.dump(folder/'progress.json',{'seed':args.seed,'completed_prediction_n':len(predictions),
                       'current':name,'missing':missing,'elapsed_seconds':time.monotonic()-started})
            print('FROZEN',args.seed,name,'seconds',round(row['inference_seconds'],2),flush=True)
        del model;torch.cuda.empty_cache()
    t.c.dump(folder/'frozen_prediction_manifest.json',{'seed':args.seed,
                'pixel_order_sha256':audit['indices_sha256']['3'],'predictions':predictions,'missing':missing})
    rows={}
    for name,record in predictions.items():
        assert t.c.helper.file_hash(Path(record['prediction_path']))==record['prediction_sha256']
        with np.load(record['prediction_path']) as p:
            rows[name]=dict(record,metrics=t.metrics(labels,p['predictions'],p['logits']))
    t.c.dump(folder/'trajectory.json',{'seed':args.seed,'records':rows,'missing':missing,
                'all_predictions_frozen_before_metric_computation':True})
    stages={}
    for arm in t.ARMS:
        name='%s_epoch_200'%arm
        if name not in predictions:stages[arm]={'status':'missing_epoch200'};continue
        record=next(x for x in inventory['runs']['%s_%d'%(arm,args.seed)]['checkpoints'] if x['epoch']==200)
        t.c.pavia.seed_worker(args.seed)
        model,_=t.c.build('pavia',arm,args.seed)
        model.load_state_dict(torch.load(record['path'],map_location='cpu')['model']);model.to(device).eval()
        result,pred=probe(model,values,device)
        with np.load(predictions[name]['prediction_path']) as saved:assert np.array_equal(pred,saved['predictions'])
        result.update(arm=arm,seed=args.seed,epoch=200,checkpoint_sha256=record['sha256'],
                      predictions_match_frozen_trajectory=True)
        if arm=='B' and args.seed==2100:
            old=json.loads((t.OUT.parent/'pavia_l2_fixed_diagnostic_v1/stage_diagnosis.json').read_text())
            assert old['checkpoint_sha256']==record['sha256']
            error=max(float(np.max(np.abs(np.asarray(old['stage_similarities'][key]['target_vs_source_class_mean_cosine'])-
                       np.asarray(result['stage_similarities'][key]['target_vs_source_class_mean_cosine']))))
                       for key in result['stage_similarities'])
            assert error<=5e-5, ('B2100 probe does not reproduce',error)
            result['existing_B2100_probe_max_cosine_error']=error
        stages[arm]=result
        del model;torch.cuda.empty_cache()
    t.c.dump(folder/'stages_epoch200.json',stages)
    t.c.dump(folder/'completion.json',{'seed':args.seed,'checkpoint_count':len(rows),
               'missing':missing,'elapsed_seconds':time.monotonic()-started,
               'no_training_or_model_selection':True})
    print('COMPLETED_SEED',args.seed,flush=True)


if __name__=='__main__':main()
