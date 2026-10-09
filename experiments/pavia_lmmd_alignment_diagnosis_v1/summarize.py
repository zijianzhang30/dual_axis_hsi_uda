"""Aggregate all three frozen read-only audits; no automatic method design."""
import csv
import json
import numpy as np
import diagnostics as d

def stats(a):
    a=np.asarray([v for v in a if v is not None and np.isfinite(v)],dtype=float)
    return {'n':len(a),'values':a.tolist(),'mean':float(a.mean()) if len(a) else None,
        'sample_std':float(a.std(ddof=1)) if len(a)>1 else None}

def csvout(name,rows):
    with (d.OUT/name).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(d.clean(rows))

def main():
    assert not (d.OUT/'summary.json').exists(),'Refuse overwrite';d.verify_files()
    phases=['CE100']+['LMMD%03d'%e for e in d.PHASES];results={};reliability=[];geometry=[];gradrows=[];classgrads=[];members=[];features=[];modes=[]
    for seed in d.SEEDS:
        folder=d.OUT/('seed_%d'%seed);complete=json.loads((folder/'complete.json').read_text())
        assert complete['original_checkpoint_and_input_hashes_unchanged'] and complete['no_training']
        for phase in phases:
            r=json.loads((folder/(phase+'.json')).read_text());results[seed,phase]=r
            for kind,key in [('eval_subset','eval_reliability'),('population_epoch100','full_population_reliability_epoch100')]:
                if r[key] is not None:
                    for cls in r[key]['classes']:reliability.append({'seed':seed,'phase':phase,'mode':kind,**cls})
            for cls in r['geometry']['classes']:
                row={'seed':seed,'phase':phase,**{k:v for k,v in cls.items() if not isinstance(v,dict)}}
                row.update(source_scatter=cls['source_moments']['covariance_trace'],target_scatter=cls['target_moments']['covariance_trace'],
                    source_effective_rank=cls['source_moments']['effective_rank'],target_effective_rank=cls['target_moments']['effective_rank'])
                geometry.append(row)
            for row in r['fixed_CE_mode_statistics']:modes.append({'seed':seed,'phase':phase,**row})
            if phase!='CE100':
                q=r['train_mode_diagnosis'];assert q['audit']['model_RNG_views_plan_match_history']
                for kind,key in [('train_JointBN_dropout','reliability'),('eval_same_augmented_patches','eval_on_identical_augmented_target_patches')]:
                    for cls in q[key]['classes']:reliability.append({'seed':seed,'phase':phase,'mode':kind,**cls})
                members.extend(q['membership']);features.extend(q['feature_gradient_samples'])
                for v in q['gradients']:
                    for block,value in [('all_parameters',v['CE_vs_LMMD'])]+list(v['blocks'].items()):
                        gradrows.append({'seed':seed,'epoch':v['epoch'],'batch':v['batch'],'block':block,**value,
                            'actual_vs_oracle_parameter_cosine':v['actual_vs_oracle_parameter_gradient']['cosine']})
                    for k,value in v['classes'].items():
                        if value['active']:
                            comp=value['CE_class_vs_LMMD_class'];oracle=value.get('actual_vs_oracle_class_gradient',{})
                            classgrads.append({'seed':seed,'epoch':v['epoch'],'batch':v['batch'],'class':int(k),
                                **comp,'actual_vs_oracle_class_cosine':oracle.get('cosine')})
    csvout('pseudo_label_reliability.csv',reliability);csvout('class_geometry.csv',geometry)
    csvout('gradient_by_block.csv',gradrows);csvout('gradient_by_class.csv',classgrads)
    csvout('LMMD_class_activation_and_purity.csv',members);csvout('target_feature_gradient_oracle.csv',features)
    csvout('fixed_CE_mode_statistics.csv',modes)
    aggregate={}
    for phase in phases:
        rr=[results[seed,phase] for seed in d.SEEDS]
        aggregate[phase]={'eval_subset':{k:stats([r['eval_reliability'][k] for r in rr]) for k in ('OA','AA','ECE10','NLL')},'classes':{}}
        for k in range(1,8):
            reliability_classes=[r['eval_reliability']['classes'][k-1] for r in rr]
            gc=[r['geometry']['classes'][k-1] for r in rr]
            aggregate[phase]['classes'][str(k)]={
                **{key:stats([v[key] for v in reliability_classes]) for key in ('recall_percent','pseudo_precision_percent','LMMD_soft_membership_purity','GT_mean_correct_class_probability')},
                **{key:stats([v[key] for v in gc]) for key in ('source_target_same_class_centroid_squared_L2','source_target_same_class_centroid_cosine',
                    'target_centroid_correct_vs_wrong_margin','target_sample_correct_source_margin_mean','target_sample_negative_correct_source_margin_fraction',
                    'soft_target_centroid_to_source_squared_L2','soft_target_centroid_to_GT_target_squared_L2')},
                'nearest_source_for_target_centroid':[v['target_centroid_nearest_source_class'] for v in gc],
                'target_covariance_trace':stats([v['target_moments']['covariance_trace'] for v in gc]),
                'target_effective_rank':stats([v['target_moments']['effective_rank'] for v in gc])}
    grad_summary={};member_summary={};feature_summary={};mode_summary=[]
    for epoch in d.PHASES:
        grad_summary[str(epoch)]={};member_summary[str(epoch)]={};feature_summary[str(epoch)]={}
        for block in sorted(set(v['block'] for v in gradrows)):
            byseed=[]
            for seed in d.SEEDS:
                chosen=[v for v in gradrows if v['epoch']==epoch and v['block']==block and v['seed']==seed]
                cos=[v['cosine'] for v in chosen if v['cosine'] is not None]
                ratios=[v['weighted_norm_ratio'] for v in chosen if v['weighted_norm_ratio'] is not None]
                direction=[v['CE_first_order_combined_descent_ratio'] for v in chosen if v['CE_first_order_combined_descent_ratio'] is not None]
                byseed.append({'seed':seed,'cosine_mean':float(np.mean(cos)) if cos else None,
                    'negative_cosine_fraction':float(np.mean(np.array(cos)<0)) if cos else None,
                    'weighted_norm_ratio_mean':float(np.mean(ratios)) if ratios else None,
                    'CE_first_order_uphill_fraction':float(np.mean(np.array(direction)<0)) if direction else None})
            grad_summary[str(epoch)][block]={'per_seed':byseed,**{key:stats([v[key] for v in byseed]) for key in ('cosine_mean','negative_cosine_fraction','weighted_norm_ratio_mean','CE_first_order_uphill_fraction')}}
        for k in range(1,8):
            byseed=[]
            for seed in d.SEEDS:
                chosen=[v for v in members if v['epoch']==epoch and v['class']==k and v['seed']==seed]
                active=[v for v in chosen if v['active']];mass=sum(v['native_class_loss_mass'] for v in active)
                purity=sum(v['native_class_loss_mass']*v['normalized_target_weight_purity'] for v in active)/mass if mass else None
                byseed.append({'seed':seed,'activation_rate':len(active)/38,'active_weight_purity':purity})
            member_summary[str(epoch)][str(k)]={'per_seed':byseed,**{key:stats([v[key] for v in byseed]) for key in ('activation_rate','active_weight_purity')}}
        for k in (4,7):
            feature_summary[str(epoch)][str(k)]={}
            for correct in (False,True):
                chosen=[v for v in features if v['epoch']==epoch and v['GT_class']==k and v['pseudo_correct']==correct]
                cos=[v['actual_vs_oracle_feature_gradient_cosine'] for v in chosen if v['actual_vs_oracle_feature_gradient_cosine'] is not None]
                dot=[v['actual_gradient_towards_true_source_centroid_dot'] for v in chosen if v['actual_gradient_towards_true_source_centroid_dot'] is not None]
                feature_summary[str(epoch)][str(k)]['pseudo_correct_'+str(correct)]={'n':len(chosen),'oracle_cosine':stats(cos),
                    'opposite_oracle_fraction':float(np.mean(np.array(cos)<0)) if cos else None,
                    'moves_away_from_true_source_centroid_fraction':float(np.mean(np.array(dot)<0)) if dot else None}
    for seed in d.SEEDS:
        for k in range(1,8):
            ce=[v for v in modes if v['seed']==seed and v['phase']=='CE100' and v['class']==k]
            lm=[v for v in modes if v['seed']==seed and v['phase']=='LMMD100' and v['class']==k]
            for j in range(2):
                a,b=next(v for v in ce if v['mode']==j),next(v for v in lm if v['mode']==j)
                mode_summary.append({'seed':seed,'class':k,'mode':j,'n':a['n'],
                    'LMMD_vs_CE_recall_delta':b['recall_percent']-a['recall_percent'],
                    'between_mode_distance_ratio':b['between_fixed_mode_squared_L2']/a['between_fixed_mode_squared_L2'],
                    'within_mode_scatter_ratio':b['covariance_trace']/a['covariance_trace'] if a['covariance_trace'] else None})
    result={'phases':phases,'seeds':d.SEEDS,'aggregate':aggregate,'gradient_summary':grad_summary,
        'native_membership_summary':member_summary,'target_feature_gradient_oracle_summary':feature_summary,
        'fixed_mode_CE_vs_LMMD':mode_summary,
        'limitations':['Frozen epoch-end gradients are not unsaved historical training updates.',
            'Train/eval matched-patch comparison combines BN and dropout;not pure BN causality.',
            'Negative gradient cosine alone is not harmful-alignment proof;norms and oracle/geometry must agree.',
            'Oracle weights use GT only offline and are never optimized.',
            'Two fixed CE feature clusters are descriptive,not proven semantic modes or a new classifier.',
            'Source centroid pool is seed2100 train selection;not source validation.',
            'Native target candidate-mask/augmentation provenance is inherited unchanged.'],
        'original_files_and_models_unchanged':True,'no_new_training_or_method':True}
    d.write(d.OUT/'summary.json',result)
    lines=['Pavia class-aware alignment diagnosis;original artifacts unchanged;no optimizer/no training.',
        'Source1260fixed;target8192uniformfixed;phases1/10/50/100;3seeds.',
        'All100 historicalRNG hashes reconstructed;phase queries input/views/RNG exactly matched.',
        'Frozen end-of-epoch weights mean queries are diagnostic,not actual historical step gradients.', '', 'Class4/7 eval features and weights:']
    for phase,a in aggregate.items():
        for k in ('4','7'):
            r=a['classes'][k];lines.append(phase+' C'+k+': '+', '.join('%s %.6f'%(key,r[key]['mean']) for key in ('recall_percent','pseudo_precision_percent','LMMD_soft_membership_purity','source_target_same_class_centroid_squared_L2','target_centroid_correct_vs_wrong_margin')))
    lines+=['','Global CE–LMMD gradient by phase:']
    for epoch,a in grad_summary.items():
        g=a['all_parameters'];lines.append(epoch+': '+', '.join('%s %s'%(k,g[k]['mean']) for k in ('cosine_mean','negative_cosine_fraction','weighted_norm_ratio_mean','CE_first_order_uphill_fraction')))
    lines+=['','Limitations:']+result['limitations'];(d.OUT/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
    print('Read-only three-seed diagnosis complete;no automatic followups.',flush=True)

if __name__=='__main__':main()
