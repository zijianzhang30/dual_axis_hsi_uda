"""Offline report of all predeclared tests; no target-chosen thresholds/variants."""
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/pavia_subprototype_diagnosis_v1')
s=json.loads((OUT/'summary.json').read_text());seeds=s['plan']['seeds']

def aggregate(stage,count,rule,cls):
    return next(a['statistics'] for a in s['aggregates'] if (a['stage'],a['prototype_count'],a['rule'],a['true_class'])==(stage,count,rule,cls))

def fmt(v):return '%.2f +/- %.2f; seeds=%s'%(v['mean'],v['sample_std'],', '.join('%.2f'%x for x in v['per_seed']))

lines=['Pavia C+CE+LMMD: source-only single vs multiple subprototype diagnostic',
 'Frozen epoch100 checkpoints, seeds2100/2101/2102; high confidence>=0.9; all39348 target evaluation pixels.',
 'No training, no new loss, no checkpoint selection, no target GT reference construction or threshold selection.',
 'Each seed: actual180 labeled source samples/class ->120 prototype fit +60 calibration, private fixed RNG.',
 'Source prototypes: K=1 normalized class mean vs fixed K=3 source-only KMeans(n_init20), same data.',
 'T: actual4x64 tokenizer output, flattened/unit-normalized. Z: actual64D classifier-input CLS, unit-normalized.',
 'Distances: closest within-class cosine prototype; same number of prototypes in every class.',
 'Support rule: reject pseudo-class distance>source threshold. Relative-margin rule: reject d_pseudo-min_other(d)>source threshold.',
 'Both thresholds fixed at source calibration rank58/60, nominal95% acceptance, before reading any target GT.',
 'Raw target features/distances were saved without GT by all three workers, then GT barrier was released for offline reporting.',
 '', 'PRIMARY CORRESPONDENCE QUESTION: fraction of high-confidence C4/C7->C3 errors closer to a true-class prototype than any C3 prototype.',
 'stage | class | single (2100,2101,2102) | multiple (2100,2101,2102)']
for stage in ('T','Z'):
    for cls in ('C4','C7'):
        r=[x for x in s['paired_correspondence'] if x['stage']==stage and x['true_class']==cls]
        lines.append('%s | %s | %s | %s'%(stage,cls,
            ', '.join('%.2f%%'%x['single_true_closer_than_C3_percent'] for x in r),
            ', '.join('%.2f%%'%x['multi_true_closer_than_C3_percent'] for x in r)))
lines.extend(['','ALL PREDECLARED FILTERS (percentage mean +/- sample SD over3 seeds; AUROC unitless):'])
for stage in ('T','Z'):
 for count in (1,3):
  for rule in ('support','relative_margin'):
   lines.append('%s K%d %s'%(stage,count,rule))
   for cls in ('C4','C7','C4+C7'):
    a=aggregate(stage,count,rule,cls)
    lines.append('  '+cls)
    for key in ('wrong_C3_rejection_percent','correct_C3_false_rejection_percent',
                'correct_C4_C7_own_class_retention_percent','error_vs_correct_C3_AUROC'):
        lines.append('    '+key+': '+fmt(a[key]))
lines.extend(['','INTERPRETATION / GO-NO-GO:',
 'No-Go for a new Sub-class-Aware Alignment mechanism on current evidence; stop automatic alignment module/loss exploration.',
 'There is a real but limited C7 screening signal in T: relative-margin rejection rises from0.37% to27.11%, consistent across all3 seeds.',
 'Correct C3 false rejection rises from0.07% to3.04%. Correct C7 own-class acceptance with K3 is85.32/91.18/67.78%: appreciable collateral exclusion, especially seed2102.',
 'C7 error AUROC for the K3 T relative-margin score is0.9048/0.7734/0.7243; screening information must not be denied.',
 'However, only9.28/0/0.81% of C7 error samples become closer to the correct C7 prototype than C3; no consistent recovery of true source-class correspondence.',
 'C4 true-class-vs-C3 correspondence is2.31/0/0% with T K3, versus1.76/0.84/0% with K1; two seeds have no recovery.',
 'Z K1 and K3 both yield0% true-class-closer-than-C3 for all high-confidence C4/C7 errors in all seeds.',
 'T K3 improves nearest-class representation of already-correct C7 from74.61/85.26/72.88% to93.02/99.66/99.66%; source substructure is useful for these already-correct samples, not evidence that the wrong subgroup is recoverable.',
 'Absolute-distance screening is costly: T K3 misrejects62.91% correct C3, versus20.37% with K1; tightening source cluster support under domain shift is not reliable target rejection.',
 'Z support K3 rejection increases for errors but also increases correct C3 false rejection from3.92% to10.47%; do not report rejection rate alone as successful alignment.',
 'No target-optimal ROC threshold, no prototype-number sweep, no follow-up experiments launched.',
 '', 'LIMITATIONS:',
 'This is a diagnosis of three fixed final representations. It does not prove errors or gradients during the entire original training process.',
 'Calibration samples were not used to fit prototypes, but were used in original model training. Source patches may overlap; target coverage is not guaranteed.',
 'KMeans clusters are descriptive source feature groups, not established semantic subclasses.',
 'Negative results apply to these fixed one-vs-three cosine prototype tests, not every conceivable subclass model.',
 'Every extracted target logit matches original formal epoch100 logits with maximum error<1e-6; running BN and all weights/buffers unchanged.',
 'All original checkpoint, config, data and prediction hashes verified unchanged; no parameter gradients allocated.',
 'Evidence: frozen_plan.json/provenance.json; source features/references/calibration locks; target features/scores_NO_GT; CSV per-seed metrics and paired changes.'])
(OUT/'RESULTS.txt').write_text('\n'.join(lines)+'\n')

fig,axes=plt.subplots(2,2,figsize=(11,7));positions=np.arange(3);colors=['#4477aa','#cc6677']
for col,cls in enumerate(('C4','C7')):
    ax=axes[0,col]
    r=[x for x in s['paired_correspondence'] if x['stage']=='T' and x['true_class']==cls]
    for j,(key,label) in enumerate([('single_true_closer_than_C3_percent','Single prototype'),('multi_true_closer_than_C3_percent','Three subprototypes')]):
        ax.bar(positions+(j-.5)*.3,[x[key] for x in r],width=.28,label=label,color=colors[j])
    ax.set_xticks(positions);ax.set_xticklabels(seeds);ax.set_ylim(0,15);ax.set_ylabel('Errors closer to true class than C3 (%)');ax.set_title('T: %s correspondence recovery'%cls)
    ax.legend(fontsize=8)
    ax=axes[1,col]
    for j,count in enumerate((1,3)):
        a=aggregate('T',count,'relative_margin',cls)
        xx=positions+(j-.5)*.3
        ax.bar(xx,a['wrong_C3_rejection_percent']['per_seed'],width=.28,color=colors[j],label='K%d error rejection'%count)
        ax.scatter(xx,a['correct_C3_false_rejection_percent']['per_seed'],marker='x',s=50,color='black',zorder=3)
    ax.set_xticks(positions);ax.set_xticklabels(seeds);ax.set_ylim(0,36);ax.set_ylabel('Source-calibrated rejection (%)');ax.set_title('T: %s relative-margin screening'%cls)
    ax.legend(fontsize=8)
fig.suptitle('Source-only prototype diagnosis: black crosses = correct C3 false rejection')
fig.tight_layout();fig.savefig(OUT/'subprototype_diagnosis.png',dpi=180);fig.savefig(OUT/'subprototype_diagnosis.svg');plt.close(fig)
for file in Path(__file__).parent.glob('*.py'):shutil.copy2(file,OUT/file.name)
link=Path('/home/zhangzj26/dual_axis_hsi_uda/results/pavia_subprototype_diagnosis_v1')
if not link.exists():link.symlink_to(OUT,target_is_directory=True)
(OUT/'decision.json').write_text(json.dumps({'decision':'No-Go for new subclass alignment; partial C7 screening signal only',
    'further_alignment_exploration_stopped':True,'new_losses_or_modules_launched':False},indent=2)+'\n')
print('Report and figure saved. No follow-ups launched.')
