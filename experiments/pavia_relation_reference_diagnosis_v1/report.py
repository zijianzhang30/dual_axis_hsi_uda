"""Offline reporting only; no reference, score or threshold is changed."""
import csv
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/pavia_relation_reference_diagnosis_v1')
P2=OUT.parent/'pavia_protocol_transfer_v1'
study=OUT.parent/'pavia_lmmd_tokenizer_v1'
s=json.loads((OUT/'summary.json').read_text())
y=np.load(P2/'inputs/evaluation_labels_OFFLINE_ONLY.npy')
extra=[]
fig,axes=plt.subplots(2,3,figsize=(13,7),sharey=True)
for col,seed in enumerate((2100,2101,2102)):
    logits=np.load(study/('C_LMMD_%d'%seed)/'predictions_epoch100.npz')['logits']
    p=torch.from_numpy(logits).softmax(1).numpy();pred=p.argmax(1);high=p.max(1)>=.9
    scores=np.load(OUT/str(seed)/'target_relation_scores_NO_GT.npz')
    references=np.load(OUT/str(seed)/'source_references.npz')
    for row,name in enumerate(('W','relation_hidden')):
        good=(y==2)&(pred==2)&high
        groups=[good,(y==3)&(pred==2)&high,(y==6)&(pred==2)&high]
        for k,bad in zip((3,6),groups[1:]):
            cohort=good|bad
            extra.append({'seed':seed,'representation':name,'error_class':'C%d'%(k+1),
                'error_count':int(bad.sum()),'correct_C3_count':int(good.sum()),
                'AUROC_error_vs_correct_C3':float(roc_auc_score(bad[cohort],scores[name][cohort,2]))})
        ax=axes[row,col]
        for group,label,color in zip(groups,['Correct C3','C4 wrongly predicted C3','C7 wrongly predicted C3'],['#228833','#cc3311','#4477aa']):
            ratio=scores[name][group,2]/references[name+'_threshold'][2]
            xx=np.sort(ratio)
            ax.plot(xx,np.arange(1,len(xx)+1)/len(xx),label=label,color=color)
        ax.axvline(1,color='black',linestyle='--',linewidth=1)
        ax.set_xlim(0,5);ax.set_ylim(0,1);ax.grid(alpha=.2)
        ax.set_title('%s | seed %d'%(name,seed));ax.set_xlabel('C3 distance / source-only threshold')
        if col==0:ax.set_ylabel('Fraction at or below distance')
axes[0,0].legend(fontsize=8,loc='lower right')
fig.suptitle('Frozen epoch100 relation support: confidence >= 0.9; rejection is to the right of x=1')
fig.tight_layout();fig.savefig(OUT/'relation_support_ECDF.png',dpi=180);fig.savefig(OUT/'relation_support_ECDF.svg');plt.close(fig)
with (OUT/'per_class_discrimination.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(extra[0]));w.writeheader();w.writerows(extra)

lines=['Pavia frozen C+CE+LMMD relation reference diagnosis — completed',
 'Checkpoints: seeds 2100/2101/2102, fixed epoch100. No training or new loss.',
 'High confidence: classifier max probability >= 0.9, fixed before target GT evaluation.',
 'Source-only reference construction: each actual 180/class source selection split 120 reference + 60 calibration.',
 'Class support distance: mean squared distance to five closest unit-normalized class reference descriptors.',
 'Class threshold: rank58 of 60 held-out source calibration distances (nominal95%; observed96.67%).',
 'All three source references and thresholds locked, then all target scores saved, BEFORE loading target class labels.',
 'Primary W: sqrt assignment-map Hellinger descriptor. Secondary: existing hidden32 features pooled by W.',
 'No CNN V/Z in descriptors; no new trained projection; same source calibration procedure for both.',
 '', 'Combined C4/C7 -> C3 high-confidence errors: counts 1772 / 1748 / 2237.',
 'Correct high-confidence C3 controls: counts 2684 / 2616 / 2677.', '']
for name,m in s['mean_std'].items():
    lines.append(name)
    for key,v in m.items():
        lines.append('  %s: %.4f +/- %.4f; seeds=%s'%(key,v['mean'],v['sample_std'],v['per_seed']))
    for cls in ('C4','C7'):
        rr=[r for r in s['rows'] if r['representation']==name and r['true_class']==cls]
        auc=[r['AUROC_error_vs_correct_C3'] for r in extra if r['representation']==name and r['error_class']==cls]
        lines.append('  '+cls+' rejection / correct-own-class retention / AUROC by seed:')
        for r,a in zip(rr,auc):
            lines.append('    %d: %.2f%% / %.2f%% / %.4f'%(r['seed'],r['wrong_C3_rejection_percent'],r['correct_C4_C7_own_class_retention_percent'],a))
    lines.append('')
lines += ['Interpretation:',
 'Relation-hidden references preserve most correct C4/C7 (96.01% mean), but do not reliably screen incorrect C3 correspondences as a whole.',
 'C4 shows a limited repeatable signal: AUROC 0.6104/0.6351/0.6525; rejection44.62/43.05/53.91%.',
 'C7 shows the opposite ordering: AUROC0.2939/0.2995/0.3318; rejection12.18/9.99/13.59%.',
 'Correct C3 is falsely rejected32.30/25.84/31.57%, comparable to the combined-error rejection28.84/28.09/29.50%.',
 'Therefore, a blanket relation-support filter is No-Go on current evidence; source-calibrated relation support does not transfer reliably enough to screen C4 and C7 together.',
 'The C4-specific signal is descriptive partial evidence, not a target-free deployable true-class selector; true class is unknown at inference.',
 'No alignment mechanism/loss is designed or trained automatically; no threshold or descriptor was chosen using target performance.',
 '', 'Limitations:',
 'Source calibration samples were held out from reference construction, but not from original model training; neighboring source patches may overlap.',
 'Nominal source coverage does not imply target coverage; no conformal generalization guarantee under domain shift.',
 'Analysis describes only the three frozen final checkpoints; prior epoch-end gradient evidence cannot establish corresponding errors throughout actual training.',
 'This negative result applies to these two explicit source-reference tests, not a proof that every possible use of relations is impossible.',
 'All original checkpoint/config/data/prediction file hashes and all model state hashes unchanged; parameter gradients None.',
 'CNN BN and Transformer calls: zero. Target GT only offline evaluation. No target-best used.',
 'Raw evidence: frozen_plan.json, provenance.json, source calibration locks, source_references.npz, target_relation_scores_NO_GT.npz, offline_target_metrics.csv, per_class_discrimination.csv.']
(OUT/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
for file in Path(__file__).parent.glob('*.py'):
    shutil.copy2(file,OUT/file.name)
print((OUT/'RESULTS.txt').read_text())
