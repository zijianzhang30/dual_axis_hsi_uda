"""Read-only low-accuracy diagnosis from frozen predictions and one checkpoint."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix

ROOT=Path('/home/zhangzj26/dual_axis_hsi_uda')
OUT=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/pavia_l2_fixed_diagnostic_v1')
spec=importlib.util.spec_from_file_location('diagnostic_fixed_common',ROOT/'experiments/pavia_l2_fixed_v1/common.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def main():
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(2);torch.cuda.set_device(2)
    source,sg=c.load_scene(c.DATA/'Pavia','paviaU')
    target,tg=c.load_scene(c.DATA/'Pavia','pavia')
    c.pavia.seed_worker(2100)
    train_gt,val_gt=c.pavia.sample_gt(sg,.95,2100,mode='random')
    smask=train_gt>=0;smask[0,:]=False;smask[:,0]=False
    tmask=tg>=0;tmask[0,:]=False;tmask[:,0]=False
    source_labels=train_gt[smask];target_labels=tg[tmask]
    svalues=source[smask];tvalues=target[tmask]
    scount=np.bincount(source_labels,minlength=7);tcount=np.bincount(target_labels,minlength=7)
    smean=np.stack([svalues[source_labels==i].mean(0,dtype=np.float64) for i in range(7)])
    tmean=np.stack([tvalues[target_labels==i].mean(0,dtype=np.float64) for i in range(7)])
    su=smean/np.linalg.norm(smean,axis=1,keepdims=True)
    tu=tmean/np.linalg.norm(tmean,axis=1,keepdims=True)
    cosine=tu@su.T
    # Spectral prototypes are input diagnostics only, not trained feature probes.
    prototype=tvalues@su.T
    proto_pred=prototype.argmax(1)
    proto_cm=confusion_matrix(target_labels,proto_pred,labels=np.arange(7))
    observations={}
    for folder in sorted(x for x in c.OUT.glob('[ABC]_210*') if x.is_dir()):
        if not (folder/'results.json').exists():continue
        r=json.loads((folder/'results.json').read_text())
        cm=np.asarray(r['target_at_final']['confusion_matrix'])
        rowshares=cm/cm.sum(1,keepdims=True)
        with np.load(folder/'predictions_epoch_200.npz') as p:
            logits=p['logits'].astype(np.float64);pred=p['predictions']
        assert len(logits)==len(target_labels)
        margins=[]
        for i in range(7):
            z=logits[target_labels==i]
            competitor=np.max(np.delete(z,i,axis=1),axis=1)
            margins.append(float(np.mean(z[:,i]-competitor)))
        observations[folder.name]={'oa_percent':r['target_at_final']['oa_percent'],
            'aa_percent':r['target_at_final']['aa_percent'],
            'source_val_oa':r['source_val_at_final']['oa_percent'],
            'recall':r['target_at_final']['per_class_percent'],
            'true_class_to_predicted_class_percent':(rowshares*100).tolist(),
            'mean_true_class_logit_margin':margins,
            'prediction_share':r['target_at_final']['prediction_shares']}
    # Same source-val evaluation output branch as native helper, no BN update.
    values=c.loaders('pavia',2100,0)
    source_val=values[1]
    device=torch.device('cuda:2')
    c.pavia.seed_worker(2100)
    model,_=c.build('pavia','B',2100)
    cp=c.OUT/'B_2100/checkpoints/epoch_200.pth'
    model.load_state_dict(torch.load(cp,map_location='cpu')['model']);model.to(device).eval()
    state_before={k:v.clone() for k,v in model.state_dict().items()}
    ys=[];preds=[]
    with torch.inference_mode():
        for images,labels in source_val:
            images=images.to(device)
            preds.append(model(images,images)[1].argmax(1).cpu().numpy())
            ys.append(labels.numpy())
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in state_before.items())
    scm=confusion_matrix(np.concatenate(ys),np.concatenate(preds),labels=np.arange(7))
    source_val_report={'arm':'B','seed':2100,'checkpoint_sha256':c.helper.file_hash(cp),
            'per_class_n':scm.sum(1).tolist(),
            'recall_percent':(np.diag(scm)/scm.sum(1)*100).tolist(),
            'oa_percent':float(np.trace(scm)/scm.sum()*100),
            'all_model_parameters_and_buffers_unchanged':True}
    native=json.loads((c.OUT.parent/'mluda_pavia_native_v1/results.json').read_text())
    native_recall=np.asarray([r['per_class_recall_percent'] for r in native['runs']])
    result={'scope':'read-only post-hoc diagnosis, no new training/BN calibration/model selection',
        'source_train_class_counts':scount.tolist(),'source_train_class_shares_percent':(scount/scount.sum()*100).tolist(),
        'target_eval_class_counts':tcount.tolist(),'target_eval_class_shares_percent':(tcount/tcount.sum()*100).tolist(),
        'input_spectral_target_vs_source_class_mean_cosine':cosine.tolist(),
        'input_prototype_per_class_recall_percent':(np.diag(proto_cm)/proto_cm.sum(1)*100).tolist(),
        'input_prototype_confusion':proto_cm.tolist(),'observations':observations,
        'source_val':source_val_report,
        'native_mluda_mean_class_recall_percent':native_recall.mean(0).tolist(),
        'caveat':'target GT used post-hoc only, spectral center means are not classifier features; causes not identified by correlation'}
    c.dump(OUT/'diagnosis.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
