import pathlib,json,hashlib,shutil,numpy as np
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
O=pathlib.Path(__file__).parent;B=O.parent/'tokenizer_mechanism_v2';d=json.loads((O/'REPORT_DATA.json').read_text());h=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
notes={'affine_end_to_end_consistent_robustness_supported':False,'reason':'Own-clean integrated OA drop is larger forC inall3seeds. NormalizedretentionOA islower in2of3,AA islower in2of3 despiteonepositive AAmean. Onepredefined affine directiononly. Waffineinvariance doesnotensureCNNvalues orclassificationinvariance.','target_center_excluded_semantic_selectivity_supported_all3seeds':True,'target_boundary_selectivity_supported_all3seeds':True,'source_boundary3x3':'not estimable:0 observedknownmixedclass patches;notdenseGT boundaryabsence','target_T_Z_class_margin_and_source_center_margin_improve_all3seeds':True,'candidate_explanation':'Center-conditioned semanticselection withimproved classdiscriminability andclasswise source-targetcorrespondence;associationfromdifferentlytrainedmodels,notcausality orall-benchmarkclaim.','no_training_or_new_module':True,'Pavia_untouched':True}
(O/'CONCLUSIONS.json').write_text(json.dumps(notes,indent=2)+'\n')
with (O/'REPORT.txt').open('a') as f:f.write('\nInterpretation andlimits:\n'+json.dumps(notes,indent=2)+'\n')
# Full artifact/source/checkpointprovenance; snapshots andfrozenweights mustremainunchanged.
orig=json.loads((B/'provenance.json').read_text());checks={name:h(B/'snapshot'/name)==v for name,v in orig['snapshot_sha256'].items()};assert all(checks.values())
checkpoints={f'{a}_{s}':h(B/'houston'/f'{a}_{s}/checkpoints/epoch_200.pth') for s in [2100,2101,2102] for a in 'AC'}
for s in [2100,2101,2102]:
 for a in 'AC':assert checkpoints[f'{a}_{s}']==d['all_per_seed_and_per_class_records'][str(s)][a]['checkpoint_sha256']
(O/'provenance.json').write_text(json.dumps({'official_provenance_sha256':h(B/'provenance.json'),'snapshot_checks':checks,'checkpoint_hashes':checkpoints,'diagnostic_code_hashes':{p.name:h(p) for p in O.glob('*.py')},'protocol_sha256':h(O/'PROTOCOL.txt'),'affine_vectors_sha256':h(O/'affine_vectors.npz'),'target_GT_file_sha256':orig['dataset_sha256']['Houston']['Houston18_7gt.mat'],'source_GT_file_sha256':orig['dataset_sha256']['Houston']['Houston13_7gt.mat'],'GT_usage':'offline scoringanddescriptive diagnostics only','no_training':True},indent=2)+'\n')
fig,ax=plt.subplots(1,2,figsize=(10,3.8));groups=['all_no_reflection','boundary3x3','mixed13x13'];x=np.arange(3)
for arm,color,shift in [('A','#64748b',-.17),('C','#2563eb',.17)]:
 vals=[d['selectivity_macro'][arm]['target'][g]['conditional_same_class_mass_excluding_center'] for g in groups]
 ax[0].bar(x+shift,[v['mean']*100 for v in vals],width=.32,yerr=[v['sample_std']*100 for v in vals],label=arm,color=color,capsize=3)
 vals=[d['class_separability'][arm][st]['target']['macro_margin_LOO'] for st in ['T','Z']]
 ax[1].bar(np.arange(2)+shift,[v['mean'] for v in vals],width=.32,yerr=[v['sample_std'] for v in vals],label=arm,color=color,capsize=3)
ax[0].set_xticks(x,['All valid','3x3 boundary','13x13 mixed']);ax[0].set_ylabel('Same-class weight among known neighbors (%)');ax[0].set_title('Target selectivity, center excluded');ax[0].legend()
ax[1].set_xticks(range(2),['Tokens T','Final CLS Z']);ax[1].set_ylabel('Cosine margin to nearest other class');ax[1].set_title('Target class margin, leave-one-out');ax[1].legend();fig.tight_layout();fig.savefig(O/'selectivity_and_margin.png',dpi=180);fig.savefig(O/'selectivity_and_margin.svg');plt.close(fig)
local=pathlib.Path('/home/zhangzj26/dual_axis_hsi_uda/investigations/tokenizer_semantic_diagnostics_v1');local.mkdir(exist_ok=False)
for p in O.glob('*.py'):shutil.copy2(p,local/p.name)
link=pathlib.Path('/home/zhangzj26/dual_axis_hsi_uda/results/tokenizer_semantic_diagnostics_v1');link.symlink_to(O,target_is_directory=True)
print('Final read-onlyreport andfigures saved; allcheckpoint hashesunchanged')
