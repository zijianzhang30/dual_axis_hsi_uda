"""Source-only relation reference calibration, followed by offline target GT audit.
No optimizer, no model updates, no target-selected thresholds or representations.
"""
import csv
import importlib.util
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path('/home/zhangzj26/dual_axis_hsi_uda')
OUT = Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/pavia_relation_reference_diagnosis_v1')
spec = importlib.util.spec_from_file_location('frozen_c_lmmd', ROOT/'experiments/pavia_lmmd_tokenizer_v1/common.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
PLAN = dict(seeds=[2100,2101,2102], checkpoint_epoch=100, high_confidence=0.9,
    source_per_class=180, reference_per_class=120, calibration_per_class=60,
    split_rng='private numpy RandomState(seed+110000), within each source class',
    reference='five-nearest-source-reference mean squared distance, per class',
    coverage=0.95, calibration_rank=58, reference_neighbors=5,
    representations={'W': 'primary: sqrt(spatial assignment W), flattened and L2 normalized; Hellinger geometry',
                     'relation_hidden': 'secondary predeclared control: W-weighted existing ReLU hidden32, flattened and L2 normalized; no CNN V/Z'},
    preprocessing='unchanged P2 z-score+official ILDA; 102 bands; identity 11x11 patches; anchor (5,5)',
    target_labels='loaded only after all three source references/thresholds and all target scores are saved',
    limitations=['Frozen epoch100 models only; not historical training gradients.',
                 'Calibration split is independent of reference construction, but checkpoint training saw all 180 source samples per class.',
                 'Source patches can spatially overlap; nominal 95% source calibration is not a target coverage guarantee.',
                 'Target evaluation mask inherits original protocol; GT class values never enter references or thresholds.',
                 'No alignment loss, threshold tuning, new model training or automatic follow-up.'])

def unit(a):
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)

@torch.no_grad()
def extract(module, dataset):
    values = {'W': [], 'relation_hidden': []}
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    maximum = 0.
    for patch, unused in loader:
        assert patch.shape[1:] == (1,102,11,11)
        patch = patch.cuda()
        relation = module.relations(patch)
        assert float(relation[:,:,5,5].abs().max()) == 0.
        h = module.scorer[1](module.scorer[0](relation.permute(0,2,3,1)))
        w = module.scorer[2](h).permute(0,3,1,2).flatten(2).softmax(-1)
        error = float((w-module(patch)).abs().max())
        maximum = max(maximum, error)
        assert error < 1e-6 and w.shape[1:] == (4,121)
        rh = torch.einsum('bkn,bnd->bkd', w, h.reshape(-1,121,32))
        assert torch.isfinite(rh).all() and torch.isfinite(w).all()
        values['W'].append(w.sqrt().flatten(1).cpu().numpy())
        values['relation_hidden'].append(rh.flatten(1).cpu().numpy())
    return {name:unit(np.concatenate(rows)) for name,rows in values.items()}, maximum

def distances(x, bank, labels):
    result = np.empty((len(x),7),np.float32)
    for start in range(0,len(x),1024):
        xx=x[start:start+1024]
        for k in range(7):
            ref=bank[labels==k]
            squared=np.maximum((xx*xx).sum(1)[:,None]+(ref*ref).sum(1)[None,:]-2*xx@ref.T,0)
            result[start:start+len(xx),k]=np.partition(squared,4,axis=1)[:,:5].mean(1)
    return result

def rate(mask, values):
    return float(np.mean(values[mask])*100) if mask.any() else None

def main():
    start=time.time(); OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'frozen_plan.json').exists(), 'Never overwrite an existing diagnostic'
    c.dump(OUT/'frozen_plan.json',PLAN)
    torch.set_num_threads(2)
    b=c.base(); source_records={}; paths=[]; model_audits={}
    for seed in PLAN['seeds']:
        folder=c.run_folder('C_LMMD',seed)
        manifest=json.loads((folder/'frozen_manifest.json').read_text())
        ckpt=next(v for v in manifest['checkpoints'] if v['epoch']==100)
        assert c.sha(ckpt['path'])==ckpt['sha256']
        paths.extend([Path(ckpt['path']),folder/'frozen_manifest.json',folder/'predictions_epoch100.npz',
                      c.P2/'inputs'/('source_selection_%d.npz'%seed)])
    paths.extend([c.P2/'inputs/P2_source.npy',c.P2/'inputs/P2_target.npy',c.P2/'inputs/target_training_coords.npy'])
    originals={str(p):c.sha(p) for p in paths}
    c.dump(OUT/'provenance.json',originals)
    # Hard barrier: target GT array cannot be loaded during references or scoring.
    np_load=np.load
    def guarded_load(path,*args,**kwargs):
        assert 'evaluation_labels_OFFLINE_ONLY' not in str(path), 'Premature target GT access'
        return np_load(path,*args,**kwargs)
    np.load=guarded_load
    for seed in PLAN['seeds']:
        folder=c.run_folder('C_LMMD',seed); destination=OUT/str(seed);destination.mkdir()
        checkpoint=folder/'checkpoints/epoch_100.pth'
        model,_=c.build(seed,'C')
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert saved['epoch']==100
        model.load_state_dict(saved['model'],strict=True); model=model.cuda().eval()
        before=c.state_hash(model)
        selection=np.load(c.P2/'inputs'/('source_selection_%d.npz'%seed))
        y=selection['labels'];rng=np.random.RandomState(seed+110000)
        fit=[];cal=[]
        for k in range(7):
            ix=np.flatnonzero(y==k);assert len(ix)==180;ix=rng.permutation(ix)
            fit.extend(ix[:120]);cal.extend(ix[120:])
        fit=np.asarray(fit);cal=np.asarray(cal)
        source,err=extract(model.backbone.relation_scores,b.PatchSet('P2','source',selection['coords']))
        refs={}; metrics={}
        for name,x in source.items():
            dist=distances(x[cal],x[fit],y[fit])
            threshold=np.array([np.sort(dist[y[cal]==k,k])[57] for k in range(7)])
            assert np.isfinite(threshold).all()
            refs[name]={'bank':x[fit], 'threshold':threshold}
            metrics[name]={'calibration_nearest_class_accuracy':float((dist.argmin(1)==y[cal]).mean()*100),
                'calibration_own_class_acceptance':[float((dist[y[cal]==k,k]<=threshold[k]).mean()*100) for k in range(7)],
                'calibration_wrong_C3_acceptance_C4_C7':float((dist[np.isin(y[cal],[3,6]),2]<=threshold[2]).mean()*100),
                'thresholds':threshold.tolist()}
        np.savez(destination/'source_references.npz',reference_indices=fit,calibration_indices=cal,
                 source_coords=selection['coords'],source_labels=y,reference_labels=y[fit],
                 **{name+'_bank':v['bank'] for name,v in refs.items()},
                 **{name+'_threshold':v['threshold'] for name,v in refs.items()})
        c.dump(destination/'SOURCE_ONLY_CALIBRATION_LOCK.json',{'seed':seed,'source_metrics':metrics,
            'source_references_sha256':c.sha(destination/'source_references.npz'),'direct_W_max_error':err})
        assert c.state_hash(model)==before
        source_records[seed]={'refs':refs,'labels':y[fit],'metrics':metrics,'model':model,'state_hash':before}
        print('SOURCE LOCKED seed=%d metrics=%s'%(seed,json.dumps(metrics)),flush=True)
    c.dump(OUT/'ALL_SOURCE_REFERENCES_LOCKED.json',{str(s):c.sha(OUT/str(s)/'SOURCE_ONLY_CALIBRATION_LOCK.json') for s in PLAN['seeds']})
    target_coords=np.load(c.P2/'inputs/target_training_coords.npy')
    target=b.PatchSet('P2','target',target_coords)
    for seed,item in source_records.items():
        x,err=extract(item['model'].backbone.relation_scores,target)
        scores={name:distances(v,item['refs'][name]['bank'],item['labels']) for name,v in x.items()}
        destination=OUT/str(seed)
        np.savez(destination/'target_relation_scores_NO_GT.npz',**scores)
        assert c.state_hash(item['model'])==item['state_hash']
        assert all(p.grad is None for p in item['model'].parameters())
        model_audits[str(seed)]={'state_unchanged':True,'parameter_gradients_none':True,'direct_W_max_error':err,
                               'CNN_BN_calls':0,'Transformer_calls':0}
        del item['model']
        print('TARGET SCORES SAVED without GT seed=%d'%seed,flush=True)
    c.dump(OUT/'ALL_TARGET_SCORES_SAVED_NO_GT.json',{str(s):c.sha(OUT/str(s)/'target_relation_scores_NO_GT.npz') for s in PLAN['seeds']})
    # Offline evaluation only; all source references/thresholds irrevocably locked above.
    np.load=np_load
    truth=np.load(c.P2/'inputs/evaluation_labels_OFFLINE_ONLY.npy')
    rows=[]
    for seed,item in source_records.items():
        logits=np.load(c.run_folder('C_LMMD',seed)/'predictions_epoch100.npz')['logits']
        prob=torch.from_numpy(logits).softmax(1).numpy();pred=prob.argmax(1);conf=prob.max(1)
        assert len(truth)==len(pred)==len(target_coords)==39348
        scored=np.load(OUT/str(seed)/'target_relation_scores_NO_GT.npz')
        for name,refs in item['refs'].items():
            score=scored[name];threshold=refs['threshold'];accepted=score<=threshold[None,:]
            bad=(pred==2)&np.isin(truth,[3,6])&(conf>=0.9)
            good3=(pred==2)&(truth==2)&(conf>=0.9)
            combined=bad|good3
            auc=float(roc_auc_score(bad[combined],score[combined,2]))
            for cl in [3,6,None]:
                subgroup=bad if cl is None else bad&(truth==cl)
                corrected=accepted[np.arange(len(truth)),truth]
                row={'seed':seed,'representation':name,'true_class':'C4+C7' if cl is None else 'C%d'%(cl+1),
                    'high_conf_wrong_to_C3_count':int(subgroup.sum()),
                    'wrong_C3_rejection_percent':rate(subgroup,~accepted[:,2]),
                    'reject_C3_and_accept_true_class_percent':rate(subgroup,(~accepted[:,2])&corrected),
                    'relation_nearest_class_is_true_percent':rate(subgroup,score.argmin(1)==truth),
                    'correct_C3_high_conf_count':int(good3.sum()),
                    'correct_C3_high_conf_retention_percent':rate(good3,accepted[:,2]),
                    'error_vs_correct_C3_AUROC':auc}
                correct=(pred==truth)&(np.isin(truth,[3,6]) if cl is None else truth==cl)
                row.update(correct_C4_C7_count=int(correct.sum()),
                    correct_C4_C7_own_class_retention_percent=rate(correct,corrected),
                    correct_C4_C7_high_conf_retention_percent=rate(correct&(conf>=.9),corrected),
                    wrong_true_class_support_percent=rate(subgroup,corrected))
                rows.append(row)
    with (OUT/'offline_target_metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    means={}
    for name in PLAN['representations']:
        rr=[r for r in rows if r['representation']==name and r['true_class']=='C4+C7']
        means[name]={key:{'mean':float(np.mean([r[key] for r in rr])),
                         'sample_std':float(np.std([r[key] for r in rr],ddof=1)),
                         'per_seed':[r[key] for r in rr]}
            for key in ['wrong_C3_rejection_percent','reject_C3_and_accept_true_class_percent',
                        'correct_C3_high_conf_retention_percent','correct_C4_C7_own_class_retention_percent',
                        'relation_nearest_class_is_true_percent','error_vs_correct_C3_AUROC']}
    assert all(c.sha(path)==h for path,h in originals.items())
    c.dump(OUT/'summary.json',{'plan':PLAN,'rows':rows,'mean_std':means,'source_calibration':{
        str(s):i['metrics'] for s,i in source_records.items()},'model_audits':model_audits,
        'all_original_files_unchanged':True,'elapsed_seconds':time.time()-start,'no_followups_launched':True})
    print('COMPLETED '+json.dumps(means),flush=True)

if __name__=='__main__':main()
