import numpy as np
from common import *

def format_stat(row,key,digits=2):
 mean=row.get(key+'_mean');std=row.get(key+'_std')
 return 'NA' if mean is None else f'{mean:.{digits}f} +/- {std:.{digits}f}' if std is not None else f'{mean:.{digits}f}'

def report(s,maps,gt,coords):
 lines=['Houston frozen error atlas v1; ORIGINAL C Joint BN+C, seeds2100/2101/2102, fixed epoch200.',
  'No training/backprop/tuning. All V/T/Z+native logits/predictions+coordinates+raw spectra+full relations+source-only readouts frozen/hash checked BEFORE target GT.',
  'Source centroids use2393train only; V/T/Z centroid readouts exactly reused R5/R3/R1. Raw acquisition spectra S_RAW48; full C input relative descriptor R_FULL8112.',
  'Source same/wrong distances are unit Euclidean CLASS-CENTROID distances, not nearest individual source pixels.',
  'Three model seeds, mean +/- sampleSD; same scene/pixels repeated, not three independent scenes.','\nTop5 error-mode coverage:']
 for row in s['coverage']:lines.append(row['mode_definition']+': '+format_stat(row,'top5_all_errors_percent')+'%')
 lines.append(f"Pooled joint Top5: {s['pooled_joint_top5_coverage_percent']:.2f}% of {s['total_seed_pixel_error_observations']} seed-pixel error observations; not independent observations.")
 lines.extend(str(r) for r in s['pooled_joint_top5']);lines.append('\nDirected confusion error contributions (all42pairs CSV retained):')
 for row in sorted(s['confusions'],key=lambda r:-r['all_errors_percent_mean'])[:10]:lines.append(row['pair']+': '+format_stat(row,'all_errors_percent')+'%')
 lines.append('\nSpatial boundaries: <=6pixel radius vs >6, Euclidean distances to8-neighbor boundary pixels.')
 lines.append('class_mask includes edges against unknownGT; semantic excludes unknown, annotation is known/unknown edge. Scene edges separately recorded. Semantic no-boundary classes=NA, never fake interior.')
 for row in s['boundary_rates']:
  if row['control'] in ['all','annotation_far_scene_safe']:lines.append(str(row))
 lines.append('\nExclusive8bit source-centroid trajectories:')
 for row in s['trajectories']:lines.append(row['bits_VTZ']+' '+row['trajectory']+': '+format_stat(row,'R0_errors_percent')+'% of R0errors')
 lines.append('\nWhich frozen signals retain STRICT positive correct-class evidence on native classifier errors:')
 for row in s['evidence']:
  if row['subset']=='R0_errors':lines.append(row['descriptor']+': '+format_stat(row,'positive_evidence_percent')+'%')
 for row in s['unions']:
  if row['subset'] in ['R0_errors','Z_centroid_errors','R0_and_Z_centroid_errors']:lines.append(str(row))
 lines.extend(['\nInterpretation and limits:',
  'Centroid correctness is a source-anchored descriptive readout, not a trained replacement classifier. Evidence union requires oracle knowledge of GT and is not deployable routing.',
  'Raw spectrum/relation evidence with later centroid errors flags candidate readout disagreements, not proof of causal information destruction; dimensions/pooling/geometries differ.',
  'Overlapping requested mechanism flags are separate CSV;8bit patterns alone form a partition. Top5pair and Top5joint coverage are different questions, never add them.',
  'Incomplete GT can make annotation-mask edges look like class boundaries; separate semantic/annotation/scene-safe controls are essential. Distances are pixel units, no physical calibration.',
  'Raw spectral centroid uses angular geometry; raw intensity is not assumed sensor-calibrated across scenes. Relation vector is full band-major13x13 normalized center-relative input to C, not learned token.',
  'No target GT used to choose feature stage, mode definition, width6, centroid, formula, thresholds, checkpoint, or method. Post-GT frequency rankings are offline explanation only.',
  'No automatic additional training/variant; B2 previous No-Go unaffected.'])
 text='\n'.join(lines)+'\n';(OUT/'REPORT.txt').write_text(text);(OUT/'FINDINGS.txt').write_text(text)
 figures(s,maps,gt,coords)
 print('\n'.join(lines[:18]),flush=True)

def figures(s,maps,gt,coords):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.colors import ListedColormap,BoundaryNorm
 # Top confusion contributions and trajectories partition.
 fig,ax=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
 rows=sorted(s['confusions'],key=lambda r:-r['all_errors_percent_mean'])[:10]
 ax[0].barh(np.arange(len(rows)),[r['all_errors_percent_mean'] for r in rows],xerr=[r['all_errors_percent_std'] for r in rows]);ax[0].set(yticks=np.arange(len(rows)),yticklabels=[r['pair'] for r in rows],xlabel='All native-classifier errors (%)');ax[0].invert_yaxis()
 rows=s['trajectories'];ax[1].bar(np.arange(8),[r['R0_errors_percent_mean'] for r in rows],yerr=[r['R0_errors_percent_std'] for r in rows]);ax[1].set(xticks=np.arange(8),xticklabels=[r['bits_VTZ'] for r in rows],xlabel='V/T/Z source-centroid correctness bits',ylabel='All native-classifier errors (%)')
 fig.savefig(OUT/'error_confusions_and_trajectories.png',dpi=180);plt.close(fig)
 fig,ax=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
 for i,(dtype,label) in enumerate([('class_mask','GT mask edge (includes annotation)'),('semantic','Observed known-class boundary')]):
  rows=[r for r in s['boundary_rates'] if r['distance_definition']==dtype and r['control']=='all' and r['band'] in ['boundary_le6','interior_gt6']]
  valid=[r for r in rows if r['error_rate_percent_mean'] is not None]
  ax[i].bar(np.arange(len(valid)),[r['error_rate_percent_mean'] for r in valid],yerr=[r['error_rate_percent_std'] or 0 for r in valid]);ax[i].set(xticks=np.arange(len(valid)),xticklabels=[r['band'] for r in valid],ylabel='Native-classifier error rate (%)',title=label)
 fig.savefig(OUT/'boundary_vs_interior.png',dpi=180);plt.close(fig)
 rows=[r for r in s['evidence'] if r['subset']=='R0_errors'];fig,ax=plt.subplots(figsize=(8,4.5),constrained_layout=True)
 ax.bar(np.arange(5),[r['positive_evidence_percent_mean'] for r in rows],yerr=[r['positive_evidence_percent_std'] for r in rows]);ax.set(xticks=np.arange(5),xticklabels=[r['descriptor'] for r in rows],ylabel='Native errors with correct source-centroid evidence (%)',title='Frozen evidence retained across representations');fig.savefig(OUT/'retained_correct_class_evidence.png',dpi=180);plt.close(fig)
 fig,ax=plt.subplots(4,1,figsize=(16,10),constrained_layout=True);colors=['#bbbbbb','#377eb8','#4daf4a','#e41a1c','#984ea3','#ff7f00','#ffff33','#a65628'];cmap=ListedColormap(colors)
 ax[0].imshow(gt+1,cmap=cmap,vmin=0,vmax=7,interpolation='nearest');ax[0].set_title('Target GT (offline only), grey=unknown; native evaluation pool excludes some scene-edge centers')
 for i,record in enumerate(maps,1):
  image=np.full(gt.shape,-1,int);image[coords[:,0],coords[:,1]]=0
  for r,c,yy,pp,error in zip(coords[:,0],coords[:,1],gt[coords[:,0],coords[:,1]],record['predictions'],record['errors']):
   if error:
    key=f'C{yy+1}->C{pp+1}';image[r,c]=record['top5_pairs'].index(key)+2 if key in record['top5_pairs'] else 1
  ecmap=ListedColormap(['#eeeeee','#ffffff','#777777','#377eb8','#4daf4a','#e41a1c','#984ea3','#ff7f00'])
  ax[i].imshow(image,interpolation='nearest',cmap=ecmap,vmin=-1,vmax=6);ax[i].set_title(f"seed{record['seed']}: native errors,grey=other error; Top5(colors): "+', '.join(record['top5_pairs']))
 for a in ax:a.set(xlabel='Original pixel column',ylabel='Original pixel row')
 fig.savefig(OUT/'spatial_error_atlas.png',dpi=180);plt.close(fig)
