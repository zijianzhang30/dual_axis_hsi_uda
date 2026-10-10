"""Required same-domain batch/fixed-summary diagnostic, no target GT read."""
import json
import numpy as np,torch
import study as s
from mechanism import Captured,install,json_load_cohorts,statistics,summary_loader

def main():
 s.verify();torch.set_num_threads(2);c=s.env();results=[]
 for arm in s.ARMS:
  m,ls,helper,classes,ba=c.setup('houston','C',2100);install(m,arm);m.cuda().eval();folder=s.OUT/(arm+'_2100');cp=folder/'checkpoints/epoch_100.pth';m.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True)['model']);att=m.blocks[0].attn;att.scene_summary_mode='fixed';before=c.state_hash(m);saved=att.scene_summary.clone();selected=json_load_cohorts();domain='source' if arm=='S1' else 'target';dataset=ls[0].dataset if domain=='source' else ls[2].dataset
  td=ls[-1].dataset;td.label=np.zeros_like(td.label);td.labels=np.zeros_like(td.labels);query=torch.stack([td[i][0] for i in range(128)]).cuda();rows=[];current=[]
  with torch.no_grad():
   att.scene_log=True;fixed=m(query,query)[1];fixeddiag=dict(att.last_diagnostics);att.scene_log=False
   def capture(mod,args):current.append(statistics(args[1]).detach());raise Captured()
   for i,(x,_) in enumerate(summary_loader(dataset,selected[domain]['positions'])):
    hook=att.register_forward_pre_hook(capture)
    try:
     x=x.cuda()
     try:m(x,x)
     except Captured:pass
    finally:hook.remove()
    context=current.pop();att.scene_summary.copy_(context);att.scene_log=True;z=m(query,query)[1];bd=dict(att.last_diagnostics);att.scene_log=False;att.scene_summary.copy_(saved)
    rows.append({'cohort_batch':i,'summary_domain':domain,'summary_L2_from_fixed':float((context-saved).norm()),'query_logits_max_error':float((z-fixed).abs().max()),'query_predictions_changed':int((z.argmax(1)!=fixed.argmax(1)).sum()),'attention_metric_mean_deviation':bd['modulation_abs_deviation_mean'],'query_probability_JS':float((.5*(fixed.softmax(-1)*(fixed.log_softmax(-1)-(.5*(fixed.softmax(-1)+z.softmax(-1))).log())).sum(-1)+.5*(z.softmax(-1)*(z.log_softmax(-1)-(.5*(fixed.softmax(-1)+z.softmax(-1))).log())).sum(-1)).mean())})
  assert len(rows)==8 and c.state_hash(m)==before;results.append({'arm':arm,'epoch':100,'fixed_diagnostics':fixeddiag,'same_domain_cohort_batch_comparisons':rows,'target_labels_accessed':False,'model_and_BN_buffers_restored_exactly':True,'extra_frozen_summary_forwards':8,'extra_query_forwards':9,'query_pixels':128})
  del m;torch.cuda.empty_cache()
 s.dump(s.OUT/'same_domain_summary_mismatch.json',{'plan':'prelocked before target scoring:8 batches of128 from existing frozen1024 same-domain cohort; fixed128 targetevalquery; no selection or tuning','results':results});s.verify();print('SAME_DOMAIN_SUMMARY_MISMATCH_COMPLETE',flush=True)
if __name__=='__main__':main()
