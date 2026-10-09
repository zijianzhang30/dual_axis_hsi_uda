"""Only three empirical conclusions, plus preserved D/E analytic fact."""
import json
from common import OUT,BASE,dump

def support(contrast):
 return {'oa_mean':contrast['oa_percent']['mean'],'aa_mean':contrast['aa_percent']['mean'],
  'paired_oa':contrast['oa_percent']['values'],'paired_aa':contrast['aa_percent']['values'],
  'both_positive_at_least2_seeds':sum(a>0 and b>0 for a,b in zip(contrast['oa_percent']['values'],contrast['aa_percent']['values']))>=2,
  'both_means_positive':contrast['oa_percent']['mean']>0 and contrast['aa_percent']['mean']>0}

def main():
 answers={'center_reference':{},'std_normalization':{},'capacity_matched_nonlinearity':{},
  'D_E_fact':'Same linear scorer parameters: local/center references differ only by a spatial constant canceled by spatial softmax. Separately trained score differences are trajectory differences, NOT linear center-reference gains.',
  'capacity_proof':json.loads((OUT/'verification.json').read_text()),'no_target_best':True,'no_new_loss_BN_variant_or_HyRANK_training':True}
 lines=['Capacity-matched G extension: fixed200; old A-F unchanged','']
 for ds in ['houston','pavia']:
  base_path=BASE/ds/'summary.json'
  if not base_path.exists():continue
  baseline=json.loads(base_path.read_text());gpath=OUT/ds/'summary.json';g=json.loads(gpath.read_text()) if gpath.exists() else None
  answers['center_reference'][ds]=support(baseline['paired_deltas']['C-B'])
  answers['std_normalization'][ds]=support(baseline['paired_deltas']['C-F'])
  answers['capacity_matched_nonlinearity'][ds]=support(g['paired_deltas']['C-G']) if g else {'tested':False,'reason':'Stage condition not passed or prior negative evidence stopped G.'}
  lines.append(ds)
  for title,key in [('center reference (C-B)','center_reference'),('std normalization (C-F)','std_normalization'),('matched-capacity nonlinearity (C-G)','capacity_matched_nonlinearity')]:
   lines.append(title+': '+json.dumps(answers[key][ds]))
  if g:
   lines.append('C/G scorer count and full model equality: '+json.dumps(g['C_G_capacity_proof']))
   lines.append('Complete A/B/C/F/G metrics/per-class/collapse/timing: '+str(OUT/ds/'RESULTS.txt'))
   lines.append('Weight entropy/diversity/A-reference and C-reference differences: '+str(OUT/ds/'weight_analysis.json'))
 lines+=['','Interpretation: positive C-G supports a nonlinear advantage under identical nominal depth/width/parameter count. It does not prove universal necessity or statistical significance with three seeds. A negative result does not trigger a new module.',
  answers['D_E_fact'],'No added training or design beyond the authorized G stages.']
 dump(OUT/'mechanism_answers.json',answers);(OUT/'FINAL_REPORT.txt').write_text('\n'.join(lines)+'\n');print('Independent capacity-matched conclusions saved')
if __name__=='__main__':main()
