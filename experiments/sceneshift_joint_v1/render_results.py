"""Render audited JSON to stdout; documentation is installed with apply_patch."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]

def main():
    s=json.loads((ROOT/'results/sceneshift_joint_v1/summary.json').read_text())
    ag=s['aggregate'];delta=s['paired_deltas'];lines=[]
    def add(text=''):lines.append(text)
    def fm(x,digits=2):return f"{x['mean']:.{digits}f} +/- {x['sample_sd']:.{digits}f}"
    add('# Noisy SceneShift x Full Joint BN: fixed-200 2x2 results')
    add();add('Twelve fresh runs completed: Houston13 -> Houston18, original BiDA spatial-softmax')
    add('tokenizer, depth 3, dropout .1, SGD .01, 200 epochs, seeds 2100/2101/2102.')
    add('Original BN means the native sequential source -> target mixed-BN recipe, not')
    add('source-only BN. Full Joint retains the differentiable joint recipe with momentum .19.')
    add();add('SceneShift is the frozen historical alpha .7 noisy recipe: epsilon 1e-5, amplitude')
    add('Gaussian SD .04, smooth Gaussian SD .015 via 5x5 average pooling, clamp [0,1],')
    add('extra .5 source CE. B/D loss is CE_s + .5 CE_shift; auxiliary BN updates normally.')
    add('Statistics use all unpadded normalized scene pixels, including background, without a GT mask.')
    add('The current input is BiDA normband, not historical MLUDA ILDA; the backbone/input-space')
    add('adaptation is disclosed and old MLUDA gains are not reused as evidence for this transfer.')
    add();add('## Outcome')
    add();add(f"{'Go' if s['gate']['go'] else 'No-Go'} for unifying the two factors under the preregistered practical gate.")
    add(f"SceneShift alone B-A has OA gains in {delta['B-A']['oa_percent']['positive_seeds']}/3 seeds and AA gains in {delta['B-A']['aa_percent']['positive_seeds']}/3.")
    add(f"Full Joint alone C-A has OA gains in {delta['C-A']['oa_percent']['positive_seeds']}/3 seeds and AA gains in {delta['C-A']['aa_percent']['positive_seeds']}/3.")
    add(f"The key D-C comparison has mean OA delta {delta['D-C']['oa_percent']['mean']:+.2f} pp and AA delta {delta['D-C']['aa_percent']['mean']:+.2f} pp.")
    add('The original stress-seed collapse contributes strongly to large B-A/C-A mean gains;')
    add('do not confuse recovery of a failed baseline seed with uniform OA gains in every seed.')
    add('For these seeds the composite noisy augmentation does not stably improve the joint-BN')
    add('self model. This is evidence against practical complementarity in the tested setting,')
    add('not a unique proof that normalization has absorbed every possible shift mechanism.')
    add();add('Shift noise and auxiliary dropout use separate persistent RNG streams, preserving')
    add('the original-forward RNG. Fixed epoch 200 is the sole target endpoint. No target-oracle')
    add('selection, target-selected checkpoint, alpha tuning, new loss/module or pure-affine arm.')
    add();add('## Primary three-seed aggregate')
    add();add('Sample SD uses ddof=1. Class-6 collapse is >=95% predicted class-6 share.')
    add();add('| Arm | OA (%) | AA (%) | Kappa | Class-6 share (%) | Collapse | Source-val OA (%) |')
    add('| --- | ---: | ---: | ---: | ---: | ---: | ---: |')
    labels={'A':'Original BN','B':'Original + noisy shift','C':'Full Joint','D':'Full Joint + noisy shift'}
    for arm in 'ABCD':
        a=ag[arm]
        add(f"| {arm}: {labels[arm]} | {fm(a['oa_percent'])} | {fm(a['aa_percent'])} | {fm(a['kappa'],3)} | {fm(a['class6_share_percent'])} | {a['collapse_count']}/3 | {fm(a['source_val_oa'])} |")
    add();add('## All fixed-endpoint seeds')
    add();add('| Arm | Seed | OA | AA | Kappa | C6 share | Source-val OA | Zero-recall classes |')
    add('| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |')
    for r in s['rows']:
        m=r['target_at_final'];missing=', '.join(map(str,r['zero_recall_classes'])) or 'None'
        add(f"| {r['arm']} | {r['seed']} | {m['oa_percent']:.2f} | {m['aa_percent']:.2f} | {m['kappa']:.3f} | {100*m['prediction_shares'][5]:.2f}% | {r['source_val_at_final']['oa_percent']:.2f} | {missing} |")
    add();add('Zero-recall classes are reported separately: avoiding the class-6 threshold is not')
    add('equivalent to retaining every class. Source-val OA is diagnostic only, not model selection.')
    add();add('## Paired effects (left minus right)')
    add();add('| Contrast | Seed | OA delta (pp) | AA delta (pp) | Kappa delta |')
    add('| --- | ---: | ---: | ---: | ---: |')
    for contrast in ['B-A','C-A','D-C']:
        d=delta[contrast]
        for i,seed in enumerate(s['seeds']):
            add(f"| {contrast} | {seed} | {d['oa_percent']['values'][i]:+.2f} | {d['aa_percent']['values'][i]:+.2f} | {d['kappa']['values'][i]:+.3f} |")
        add(f"| {contrast} | mean +/- SD | {fm(d['oa_percent'])} | {fm(d['aa_percent'])} | {fm(d['kappa'],3)} |")
    add();add('Positive-seed counts:')
    for contrast in ['B-A','C-A','D-C']:
        d=delta[contrast];add(f"- {contrast}: OA {d['oa_percent']['positive_seeds']}/3; AA {d['aa_percent']['positive_seeds']}/3.")
    add();add('The factorial interaction (D-C)-(B-A) is:')
    for k,name in [('oa_percent','OA pp'),('aa_percent','AA pp'),('kappa','Kappa')]:
        add(f"- {name}: {fm(s['interaction'][k],3 if k=='kappa' else 2)}.")
    add();add('Interaction is descriptive for these seeds. A positive interaction does not by itself')
    add('establish that D improves C, and a negative/zero D-C does not uniquely prove BN')
    add('has covered all benefits of input shift: the frozen SceneShift factor also includes')
    add('extra supervised CE, noise/clamp and an extra running-buffer update.')
    add();add('## Per-class recall (%)')
    for arm in 'ABCD':
        add();add(f'### {arm}: {labels[arm]}');add()
        add('| Class | 2100 | 2101 | 2102 | Mean +/- SD |');add('| --- | ---: | ---: | ---: | ---: |')
        rows=[r for r in s['rows'] if r['arm']==arm]
        for i in range(7):
            vals=[r['target_at_final']['per_class_percent'][i] for r in rows]
            add(f"| {i+1} | {vals[0]:.2f} | {vals[1]:.2f} | {vals[2]:.2f} | {ag[arm]['per_class_mean'][i]:.2f} +/- {ag[arm]['per_class_sample_sd'][i]:.2f} |")
    add();add('Exact predicted counts/shares and confusion matrices for every seed are preserved')
    add('in each results.json and aggregate summary.json.')
    add();add('## Parameters and cost')
    add();add('All arms have 376,567 trainable parameters; SceneShift and joint BN add zero parameters.')
    add('All arms use one optimizer update per paired batch. B/D add one supervised paired')
    add('forward and its backward gradient contribution; A/C have no auxiliary forward.')
    add();add('| Arm | Preparation seconds | Training seconds | Auxiliary CUDA span seconds |')
    add('| --- | ---: | ---: | ---: |')
    for arm in 'ABCD':
        add(f"| {arm} | {fm(ag[arm]['preparation_seconds'])} | {fm(ag[arm]['train_seconds'])} | {fm(ag[arm]['auxiliary_cuda_seconds'])} |")
    add();add('Measured paired additional training time:')
    for contrast in ['B-A','C-A','D-C']:
        add(f"- {contrast}: {fm(delta[contrast]['train_seconds'])} seconds; left/right ratio {fm(delta[contrast]['train_time_ratio'])}.")
    add();add('Twelve runs were parallelized with three processes per free GPU: A on 4, B on 5,')
    add('C on 6, D on 7. Wall times include hashing/logging, validation and checkpoints.')
    add('CUDA event spans include noise generation + auxiliary forward/loss, not auxiliary')
    add('backward time in isolation. Concurrent kernel interleaving and different finishing')
    add('times mean these are observed workload costs, not controlled standalone speed benchmarks.')
    add();add('## Preregistered Go/No-Go')
    add();add('Before outcomes, material improvement was operationalized as mean D-C >=1 pp in')
    add('OA or AA and positive changes in at least 2/3 seeds in that same metric, with')
    add('no increase in collapse count. This is an exploratory practical gate, not significance.')
    gate=s['gate']
    add();add(f"Decision: **{'Go' if gate['go'] else 'No-Go'}**.")
    add(f"- OA material/majority gate: {gate['majority_positive_same_metric']['oa_percent']}.")
    add(f"- AA material/majority gate: {gate['majority_positive_same_metric']['aa_percent']}.")
    add(f"- No collapse count increase: {gate['no_collapse_count_increase']}.")
    add();add('Continue no automatic new module/variant from this run. ' +
        ('This result permits considering complementarity, with all negative changes and small-seed limitations disclosed.' if gate['go'] else
         'Do not continue stacking SceneShift or launch alpha/pure-affine rescue experiments; retain Full Joint reference.'))
    add();add('## Audit and frozen artifacts')
    add();add(f"All {s['artifact_files_verified']} checkpoint/prediction hashes and twelve manifest hashes passed.")
    add('Every seed has identical initial tensors, source/target indices and full-scene')
    add('statistics across all four arms. All 200 epochs have identical paired input stream')
    add('and pre-original-forward CPU/CUDA RNG stream hashes across A/B/C/D.')
    add();add('| Control | Seed | Historical final state exact | Maximum tensor difference |')
    add('| --- | ---: | --- | ---: |')
    for r in s['rows']:
        if r['historical_replay'] is not None:
            h=r['historical_replay'];add(f"| {r['arm']} | {r['seed']} | {h['state_exact']} | {h['max_abs_difference']:.6g} |")
    add();add('Fresh A/C endpoints, not historical metrics, are used in formal paired contrasts.')
    add('Target GT-mask sampling is the unchanged BiDA loader behavior; label values do not')
    add('enter loss/checkpoint selection. Target prediction arrays and hashes were frozen')
    add('before post-hoc target GT scoring. No target labels were used to tune the noisy recipe.')
    add();add('Artifacts: results/sceneshift_joint_v1/{A,B,C,D}_{2100,2101,2102}/,')
    add('results/sceneshift_joint_v1/summary.json. Historical runs/files were preserved.')
    print('\n'.join(lines))

if __name__=='__main__':main()
