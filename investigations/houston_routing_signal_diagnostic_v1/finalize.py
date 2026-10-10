"""Post-scoring descriptive summaries; never changes any frozen router."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/houston_routing_signal_diagnostic_v1'
s=json.loads((OUT/'summary.json').read_text())
assert (OUT/'SCORING_COMPLETE.json').exists()

def fmt(rows,key,scale=1):
    a=np.array([r[key] for r in rows],float)*scale
    if len(a)==1:return f'{a.mean():.5f} +/- NA (n_seeds=1)'
    return f'{a.mean():.5f} +/- {a.std(ddof=1):.5f}'

lines=['Post-scoring descriptive interpretation (no router or parameter changes):',
 'T supplies complementary correct routing on some original Z-classifier errors, but has worse overall routing accuracy and damages many original correct decisions.',
 'Primary R0-R4 oracle availability covers only about one fifth of original errors. Prespecified V readouts raise availability to about two fifths; V is excluded from majority.',
 'These availability ceilings assume GT chooses the correct router. They do not show that an unlabeled selector can identify it.',
 'Fixed majority fails to improve fixed-statistics transport: both prespecified k values lower OA and AA versus D00. All three seeds have negative OA differences.',
 'Complementarity exists, but exploiting it safely remains unresolved; a correct router prediction alone does not guarantee that its transport is classified correctly.',
 'No best k/stage/router is selected and no subsequent experiment is started.','']
agg=[]
for family,key in [('routing_subsets','correct_fraction'),('transport_subsets','accuracy_after'),('transport_subsets','recovery_fraction_of_subset'),('transport_subsets','damage_fraction_of_subset'),('tie_audit','majority_tie_fraction')]:
    groups={}
    for r in s[family]:
        group=(r['k'],r.get('router',r.get('transport','Majority')),r['subset'])
        groups.setdefault(group,[]).append(r)
    for (k,name,subset),rows in groups.items():
        a=np.array([r[key] for r in rows])*100
        agg.append(dict(family=family,k=k,method=name,subset=subset,metric=key+'_percent',
                        n_seeds=len(a),mean=float(a.mean()),sample_std=float(a.std(ddof=1)) if len(a)>1 else None))
with (OUT/'subset_seed_aggregate.csv').open('x',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(agg[0]));w.writeheader();w.writerows(agg)

audits=[]
for k in [5,15]:
    lines.append(f'k{k} additional summaries, mean +/- sampleSD:')
    for a,b in [('R4','R2'),('R2','R4'),('R3','R4'),('R4','R3')]:
        rows=[r for r in s['pairwise_complementarity'] if r['k']==k and r['router_A']==a and r['router_B']==b]
        lines.append(f'  {a} correct while {b} wrong: '+fmt(rows,'A_extra_over_B_percent')+'% of original R0 errors.')
    for sub in ['C1_all','C7_all','C1_R0_error','C7_R0_error','C7_to_C6','C1_to_C7','C7_to_C1']:
        rows=[r for r in s['transport_subsets'] if r['k']==k and r['transport']=='Majority_transport' and r['subset']==sub]
        if rows:
            lines.append(f'  Majority transport {sub}: accuracy='+fmt(rows,'accuracy_after',100)+'%; original errors recovered / subset='+fmt(rows,'recovery_fraction_of_subset',100)+'%; original correct damaged / subset='+fmt(rows,'damage_fraction_of_subset',100)+'%.')
    for sub in ['R0_error','R0_correct']:
        rows=[r for r in s['tie_audit'] if r['k']==k and r['subset']==sub]
        lines.append(f'  Majority highest-count ties on {sub}: '+fmt(rows,'majority_tie_fraction',100)+'%.')
    for seed in [2100,2101,2102]:
        f=OUT/f'seed_{seed}'
        a=np.load(f/f'primary_R0_R4_k{k}_GT_availability.npz');m=np.load(f/f'Majority_k{k}_routing.npz')
        available=a['any_router_correct']&a['R0_error'];miss=available&(m['predictions']!=a['target_labels'])
        audits.append(dict(seed=seed,k=k,available_error_n=int(available.sum()),
            majority_correct_percent=100*float((available&~miss).sum()/available.sum()),
            missed_tie_percent=100*float((miss&m['tied']).sum()/available.sum()),
            missed_unique_wrong_winner_percent=100*float((miss&~m['tied']).sum()/available.sum())))
    rows=[r for r in audits if r['k']==k]
    lines.append('  Of original errors with at least one correct primary router: majority retains '+fmt(rows,'majority_correct_percent')+'%; loses via tie fallback '+fmt(rows,'missed_tie_percent')+'%; loses via a unique wrong vote winner '+fmt(rows,'missed_unique_wrong_winner_percent')+'%.')
lines.extend(['','Scope: one Houston source-target scene pair; three optimization seeds are not independent scenes. This is a descriptive diagnosis, not an efficacy claim for a new method.'])
with (OUT/'majority_availability_loss_audit.csv').open('x',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(audits[0]));w.writeheader();w.writerows(audits)
(OUT/'FINDINGS.txt').write_text('\n'.join(lines)+'\n\n'+(OUT/'FINDINGS.txt').read_text())
with (OUT/'REPORT.txt').open('a') as f:f.write('\n\n'+'\n'.join(lines)+'\n')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
names=['FINDINGS.txt','REPORT.txt','summary.json','routing_signal_summary.png','subset_seed_aggregate.csv','majority_availability_loss_audit.csv','ALL_ROUTING_TRANSPORT_FROZEN.json','SCORING_COMPLETE.json']
(OUT/'FINAL_DELIVERY.json').write_text(json.dumps({'post_scoring_text_and_aggregates_only':True,
 'construction_freeze_and_original_scoring_manifests_preserved':True,'no_training_tuning_followup':True,
 'files_sha256':{n:sha(OUT/n) for n in names},'finalize_code_sha256':sha(Path(__file__).resolve())},indent=2)+'\n')
print('\n'.join(lines))
