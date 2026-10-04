"""Locate class-conditional feature shifts in a frozen B2100 model; no training."""
import importlib.util
import json
from pathlib import Path
from types import MethodType
import numpy as np
import torch

ROOT=Path('/home/zhangzj26/dual_axis_hsi_uda')
OUT=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/pavia_l2_fixed_diagnostic_v1')
spec=importlib.util.spec_from_file_location('stage_fixed_common',ROOT/'experiments/pavia_l2_fixed_v1/common.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def main():
    assert not (OUT/'stage_diagnosis.json').exists()
    torch.set_num_threads(2);torch.cuda.set_device(2)
    c.pavia.seed_worker(2100)
    values=c.loaders('pavia',2100,4)
    model,_=c.build('pavia','B',2100)
    checkpoint=c.OUT/'B_2100/checkpoints/epoch_200.pth'
    model.load_state_dict(torch.load(checkpoint,map_location='cpu')['model'])
    model.to('cuda:2').eval()
    state_before=c.helper.state_hash(model)
    captured={}
    def feature_hook(module,args,result):
        captured['stem_center']=result[:,:,6,6]
        captured['stem_mean']=result.mean((-2,-1))
    handle=model.conv2d_features[2].register_forward_hook(feature_hook)
    semantic=model._forward_semantic_tokens.__func__
    def tokens(self,features):
        result=semantic(self,features)
        captured['token_mean']=result.mean(1)
        return result
    model._forward_semantic_tokens=MethodType(tokens,model)
    names=('stem_center','stem_mean','token_mean','classifier_z')
    stats={};target_predictions=[]
    for domain,loader in (('source_val',values[1]),('target_eval',values[3])):
        sums={key:np.zeros((7,64),dtype=np.float64) for key in names}
        counts=np.zeros(7,dtype=np.int64)
        norm_sums={key:np.zeros(7,dtype=np.float64) for key in names}
        with torch.inference_mode():
            for images,labels in loader:
                images=images.to('cuda:2');labels=labels.numpy()
                output=model(images,images,return_feat_prob=True)
                captured['classifier_z']=output[3]
                counts+=np.bincount(labels,minlength=7)
                for key in names:
                    features=captured[key].cpu().numpy().astype(np.float64)
                    np.add.at(sums[key],labels,features)
                    np.add.at(norm_sums[key],labels,np.linalg.norm(features,axis=1))
                if domain=='target_eval':target_predictions.append(output[1].argmax(1).cpu().numpy())
        stats[domain]={'counts':counts.tolist(),'mean_features':{key:(sums[key]/counts[:,None]).tolist() for key in names},
                       'mean_norms':{key:(norm_sums[key]/counts).tolist() for key in names}}
        print('finished',domain,'n',int(counts.sum()),flush=True)
    assert c.helper.state_hash(model)==state_before, 'Model buffers or tensors unexpectedly mutated'
    handle.remove()
    with np.load(c.OUT/'B_2100/predictions_epoch_200.npz') as original:
        assert np.array_equal(np.concatenate(target_predictions),original['predictions'])
    similarities={}
    for key in names:
        s=np.asarray(stats['source_val']['mean_features'][key]);t=np.asarray(stats['target_eval']['mean_features'][key])
        s/=np.maximum(np.linalg.norm(s,axis=1,keepdims=True),1e-12)
        t/=np.maximum(np.linalg.norm(t,axis=1,keepdims=True),1e-12)
        cosine=t@s.T
        similarities[key]={'target_vs_source_class_mean_cosine':cosine.tolist(),
                           'same_class_cosine':np.diag(cosine).tolist(),
                           'nearest_source_class':(cosine.argmax(1)+1).tolist()}
    result={'arm':'B','seed':2100,'epoch':200,'checkpoint_sha256':c.helper.file_hash(checkpoint),
            'all_parameters_buffers_unchanged':True,'target_predictions_identical_to_frozen_run':True,
            'class_conditional_statistics':stats,'stage_similarities':similarities,
            'caveat':'post-hoc class centroid similarity is descriptive, not causal proof or a selected model'}
    c.dump(OUT/'stage_diagnosis.json',result)
    print(json.dumps(similarities,indent=2),flush=True)


if __name__=='__main__':main()
