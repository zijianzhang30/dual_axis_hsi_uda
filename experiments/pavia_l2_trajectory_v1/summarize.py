"""Complete checkpoint trajectories and fixed-window descriptions, no oracle."""
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import common as t

WINDOWS={'early':range(10,51,10),'middle':range(60,141,10),'late':range(150,201,10)}


def average(values):return None if not values else float(np.mean(values))


def main():
    assert not (t.OUT/'summary.json').exists()
    runs={};stage_reports={};missing=[]
    for seed in t.SEEDS:
        folder=t.OUT/('seed_%d'%seed)
        assert (folder/'completion.json').exists()
        trajectory=json.loads((folder/'trajectory.json').read_text())
        missing += [dict(record,seed=seed) for record in trajectory['missing']]
        stage_reports[seed]=json.loads((folder/'stages_epoch200.json').read_text())
        for arm in t.ARMS:
            records={int(record['epoch']):record['metrics'] for record in trajectory['records'].values() if record['arm']==arm}
            assert all(m['n']==39348 for m in records.values())
            window_stats={}
            for label,epochs in WINDOWS.items():
                available=[records[e] for e in epochs if e in records]
                window_stats[label]={'epochs_present':[e for e in epochs if e in records],
                     'oa_percent_mean':average([m['oa_percent'] for m in available]),
                     'aa_percent_mean':average([m['aa_percent'] for m in available]),
                     'class_recall_percent_mean':[average([m['per_class_percent'][i] for m in available]) for i in range(7)],
                     'class3_to_7_percent_mean':average([m['class3_to_7']['percent_of_true_class'] for m in available]),
                     'class4_to_7_percent_mean':average([m['class4_to_7']['percent_of_true_class'] for m in available])}
            recall_ranges={str(i+1):{'minimum':min(m['per_class_percent'][i] for m in records.values()),
                                     'maximum':max(m['per_class_percent'][i] for m in records.values())}
                           for i in (2,3)} if records else {}
            runs['%s_%d'%(arm,seed)]={'metrics_by_epoch':records,'fixed_windows':window_stats,
                'recall_ranges_on_saved_grid':recall_ranges}
    paired={}
    for seed in t.SEEDS:
        b=runs['B_%d'%seed]['metrics_by_epoch'];c=runs['C_%d'%seed]['metrics_by_epoch']
        rows={};absent=[]
        for epoch in t.EPOCHS:
            if epoch not in b or epoch not in c:absent.append(epoch);continue
            bm,cm=b[epoch],c[epoch]
            rows[epoch]={field:cm[field]-bm[field] for field in ('oa_percent','aa_percent','kappa')}
            rows[epoch].update(per_class_recall_delta_pp=(np.array(cm['per_class_percent'])-bm['per_class_percent']).tolist(),
               prediction_share_delta=(np.array(cm['prediction_shares'])-bm['prediction_shares']).tolist(),
               confusion_matrix_delta=(np.array(cm['confusion_matrix'])-bm['confusion_matrix']).tolist())
            for cls in (3,4):
                name='class%d_to_7'%cls
                rows[epoch][name]={'count_delta':cm[name]['count']-bm[name]['count'],
                     'percent_of_true_class_delta_pp':cm[name]['percent_of_true_class']-bm[name]['percent_of_true_class']}
        windows={}
        for label,epochs in WINDOWS.items():
            available=[rows[e] for e in epochs if e in rows]
            windows[label]={'epochs_present':[e for e in epochs if e in rows],
                'oa_delta_pp_mean':average([m['oa_percent'] for m in available]),
                'aa_delta_pp_mean':average([m['aa_percent'] for m in available]),
                'recall_delta_pp_mean':[average([m['per_class_recall_delta_pp'][i] for m in available]) for i in range(7)]}
        paired[seed]={'C_minus_B_by_epoch':rows,'missing_pairs':absent,'fixed_windows':windows}
    t.c.dump(t.OUT/'summary.json',{'only_repaired_L2':True,'runs':runs,'paired_C_minus_B':paired,
                  'stage_probes_epoch200':stage_reports,'missing_checkpoints':missing,
                  'no_checkpoint_or_version_selection':True})
    with (t.OUT/'trajectory.csv').open('w',newline='') as stream:
        w=csv.writer(stream)
        w.writerow(['arm','seed','epoch','OA','AA','Kappa']+['recall_%d'%i for i in range(1,8)]+
             ['prediction_ratio_%d'%i for i in range(1,8)]+['3_to_7_count','3_to_7_percent','4_to_7_count','4_to_7_percent']+
             ['cm_true%d_pred%d'%(i,j) for i in range(1,8) for j in range(1,8)])
        for name,run in runs.items():
            arm,seed=name.split('_')
            for epoch,m in run['metrics_by_epoch'].items():
                w.writerow([arm,seed,epoch,m['oa_percent'],m['aa_percent'],m['kappa']]+m['per_class_percent']+
                  m['prediction_shares']+[m['class3_to_7']['count'],m['class3_to_7']['percent_of_true_class'],
                  m['class4_to_7']['count'],m['class4_to_7']['percent_of_true_class']]+list(np.array(m['confusion_matrix']).ravel()))
    with (t.OUT/'paired_C_minus_B.csv').open('w',newline='') as stream:
        w=csv.writer(stream);w.writerow(['seed','epoch','OA_delta_pp','AA_delta_pp','Kappa_delta']+
                    ['recall_%d_delta_pp'%i for i in range(1,8)]+['3_to_7_count_delta','3_to_7_percent_delta_pp','4_to_7_count_delta','4_to_7_percent_delta_pp'])
        for seed,p in paired.items():
            for epoch,m in p['C_minus_B_by_epoch'].items():
                w.writerow([seed,epoch,m['oa_percent'],m['aa_percent'],m['kappa']]+m['per_class_recall_delta_pp']+
                     [m['class3_to_7']['count_delta'],m['class3_to_7']['percent_of_true_class_delta_pp'],
                      m['class4_to_7']['count_delta'],m['class4_to_7']['percent_of_true_class_delta_pp']])
    fig,axes=plt.subplots(3,2,figsize=(10,10),sharex=True,sharey=True)
    for row,seed in enumerate(t.SEEDS):
        for col,cls in enumerate((3,4)):
            ax=axes[row,col]
            for arm in t.ARMS:
                data=runs['%s_%d'%(arm,seed)]['metrics_by_epoch'];epochs=sorted(data)
                ax.plot(epochs,[data[e]['per_class_percent'][cls-1] for e in epochs],label=arm,marker='.',markersize=4)
            ax.set_title('Seed %d / class %d'%(seed,cls));ax.set_ylim(0,100);ax.grid(alpha=.2);ax.legend()
            ax.set_ylabel('Recall (%)');ax.set_xlabel('Epoch')
    fig.tight_layout();fig.savefig(t.OUT/'class34_trajectory.png',dpi=150);plt.close(fig)
    lines=['# Repaired Pavia checkpoint trajectories','',
           'B/C x seeds2100/2101/2102, all saved10-epoch checkpoints; inference only.',
           'Missing checkpoints: '+str(missing),
           'No target-best checkpoint selection, retraining, class weighting or model changes.', '',
           '## Class3/4 learning over the saved grid', '',
           '| Run | Class3 range | Class3 early / middle / late mean | Class4 range | Class4 early / middle / late mean |',
           '| --- | ---: | ---: | ---: | ---: |']
    for name,run in runs.items():
        def text(cls):
            r=run['recall_ranges_on_saved_grid'].get(str(cls))
            rng='missing' if r is None else '%.2f..%.2f'%(r['minimum'],r['maximum'])
            means=[run['fixed_windows'][label]['class_recall_percent_mean'][cls-1] for label in WINDOWS]
            return rng,' / '.join('missing' if v is None else '%.2f'%v for v in means)
        lines.append('| %s | %s | %s | %s | %s |'%(name,*text(3),*text(4)))
    lines+=['','Early=10..50; middle=60..140; late=150..200. Ranges describe only saved checkpoints.',
            'These continuous values distinguish persistent failure from deterioration without selecting a checkpoint.', '',
            '## Same-seed C-B across training phases', '',
            '| Seed / phase | OA delta | AA delta | Class3 recall delta | Class4 recall delta |',
            '| --- | ---: | ---: | ---: | ---: |']
    for seed,p in paired.items():
        for phase,values in p['fixed_windows'].items():
            data=[values['oa_delta_pp_mean'],values['aa_delta_pp_mean'],values['recall_delta_pp_mean'][2],values['recall_delta_pp_mean'][3]]
            lines.append('| %d / %s | %s | %s | %s | %s |'%(seed,phase,*['missing' if v is None else '%+.2f'%v for v in data]))
    lines+=['','## Fixed checkpoint snapshots', '',
            '| Run / epoch | OA | AA | Class3 recall | Class4 recall | 3->7 count (%) | 4->7 count (%) |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name,run in runs.items():
        for epoch in (10,50,100,150,200):
            m=run['metrics_by_epoch'].get(epoch)
            if m is None:continue
            lines.append('| %s / %d | %.2f | %.2f | %.2f | %.2f | %d (%.2f) | %d (%.2f) |'%
               (name,epoch,m['oa_percent'],m['aa_percent'],m['per_class_percent'][2],m['per_class_percent'][3],
                m['class3_to_7']['count'],m['class3_to_7']['percent_of_true_class'],
                m['class4_to_7']['count'],m['class4_to_7']['percent_of_true_class']))
    lines+=['','## Fixed200 stage probe: cross-seed replication', '',
            '| Run / class | CNN center cosine / nearest | Token mean cosine / nearest | Final z cosine / nearest |',
            '| --- | ---: | ---: | ---: |']
    for seed,models in stage_reports.items():
        for arm,probe_result in models.items():
            if 'stage_similarities' not in probe_result:continue
            for cls in (3,4):
                fields=[]
                for stage in ('stem_center','token_mean','classifier_z'):
                    s=probe_result['stage_similarities'][stage]
                    fields.append('%.4f / %d'%(s['same_class_cosine'][cls-1],s['nearest_source_class'][cls-1]))
                lines.append('| %s_%d / %d | %s | %s | %s |'%(arm,seed,cls,*fields))
    lines+=['','Stage definitions reuse the original probe; CNN spatial-mean results are also retained in JSON.',
            'Centroid proximity/readout changes are descriptive, not causal proof against a module.',
            'Full matrices, all recalls/shares, paired deltas and frozen prediction hashes are retained in JSON/CSV.',
            'No automatic follow-up experiments or checkpoint selection.']
    (t.OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('TRAJECTORY_ANALYSIS_COMPLETE',flush=True)


if __name__=='__main__':main()
