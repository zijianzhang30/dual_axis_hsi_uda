import ast,copy,json,random,shutil
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
import study as s

class Capture(Exception):pass
def main():
 s.env().verify_files();s.OUT.mkdir(exist_ok=False);(s.OUT/'code_snapshot').mkdir();files={}
 for p in Path(__file__).parent.glob('*.py'):
  ast.parse(p.read_text());q=s.OUT/'code_snapshot'/p.name;shutil.copy2(p,q);files[str(p.resolve())]=s.sha(p);files[str(q)]=s.sha(q)
 s.dump(s.OUT/'locked_config.json',s.PLAN);files[str(s.OUT/'locked_config.json')]=s.sha(s.OUT/'locked_config.json');s.dump(s.OUT/'code_manifest.json',files)
 cfg=json.loads((s.baseline()/'config.json').read_text());assert cfg['epochs']==200 and cfg['loss']=='source-self CE only' and cfg['batch']==128 and cfg['patch']==13 and cfg['lr']==.01 and cfg['depth']==3 and cfg['parameters']==378011
 refs={}
 for p in (s.baseline()/'config.json',s.baseline()/'history.jsonl',s.baseline()/'frozen_manifest.json'):refs[str(p)]=s.sha(p)
 old=s.B/'houston_shared_adapter_2x2_v1/CE0_2100/offline_predictions_frozen.json';refs[str(old)]=s.sha(old);rr=[r for r in json.loads(old.read_text())['records'] if r['epoch']<=100];assert [r['epoch'] for r in rr]==s.PLAN['grid']
 for r in rr:assert s.sha(r['path'])==r['sha256'];refs[r['path']]=r['sha256'];p=s.baseline()/'checkpoints'/('epoch_%03d.pth'%r['epoch']);refs[str(p)]=s.sha(p)
 s.dump(s.OUT/'baseline_manifest.json',refs);torch.set_num_threads(2);c=s.env();m,ls,helper,classes,ba=c.setup('houston','C',2100);assert ba['complete_initial_hash']==cfg['initial_full_model_sha256'];m.cuda().eval();initial=c.state_hash(m);cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state().clone();npstate=np.random.get_state();pystate=random.getstate();private=np.random.RandomState(2100+192000);ds=copy.copy(ls[0].dataset);positions=np.sort(private.choice(len(ds.indices),1024,replace=False));ds.indices=ds.indices[positions];ds.label=np.zeros_like(ds.label);ds.labels=np.zeros(1024,dtype=np.int64);ds.data_aug=False
 loader=torch.utils.data.DataLoader(ds,batch_size=128,shuffle=False,num_workers=0,generator=torch.Generator().manual_seed(195000));att=m.blocks[0].attn;squaresD=torch.zeros(8,dtype=torch.float64);squaresC=squaresD.clone();count=0;minimum_qk_norm=float('inf')
 def capture(mod,args):
  nonlocal count,minimum_qk_norm
  x=args[1];B,N,C=x.shape;assert N==5 and C==64;q,k,v=att.qkv(x).reshape(B,N,3,8,8).permute(2,0,3,1,4).unbind(0)
  D=(q@k.transpose(-2,-1))*att.scale;Co=F.normalize(q,dim=-1,eps=1e-12)@F.normalize(k,dim=-1,eps=1e-12).transpose(-2,-1);dd=D.double();cc=Co.double();squaresD.add_((dd-dd.mean(-1,keepdim=True)).square().sum((0,2,3)).cpu());squaresC.add_((cc-cc.mean(-1,keepdim=True)).square().sum((0,2,3)).cpu());count+=B*N*N;minimum_qk_norm=min(minimum_qk_norm,float(q.norm(dim=-1).min()),float(k.norm(dim=-1).min()));raise Capture()
 hook=att.register_forward_pre_hook(capture)
 try:
  with torch.no_grad():
   for x,_ in loader:
    x=x.cuda()
    try:m(x,x)
    except Capture:pass
 finally:hook.remove()
 rd=(squaresD/count).sqrt();rc=(squaresC/count).sqrt();assert count==1024*25 and torch.isfinite(rd).all() and torch.isfinite(rc).all() and (rc>=1e-8).all(),'calibration_degenerate_STOP';scale=rd/rc;assert torch.isfinite(scale).all() and (scale>0).all(),'calibration_scale_invalid_STOP';assert c.state_hash(m)==initial and torch.equal(cpu,torch.get_rng_state()) and torch.equal(cuda,torch.cuda.get_rng_state()) and random.getstate()==pystate;ns=np.random.get_state();assert ns[0]==npstate[0] and np.array_equal(ns[1],npstate[1]) and ns[2:]==npstate[2:]
 record={'plan':s.PLAN['calibration'],'s_h0':scale.tolist(),'D_row_centered_RMS':rd.tolist(),'C_row_centered_RMS':rc.tolist(),'samples':1024,'matrix_elements_perhead':count,'source_positions':positions.tolist(),'source_coordinates':ds.indices.tolist(),'source_indices_sha256':c.sha_array(ds.indices),'base_initial_sha256':initial,'minimum_QK_norm':minimum_qk_norm,'FP32_scale_cast_relative_RMS_error':((scale.float().double()*rc-rd).abs()/rd).tolist(),'target_input_or_GT_read_for_calibration':False,'BN_and_all_main_RNG_preserved':True}
 assert max(record['FP32_scale_cast_relative_RMS_error'])<1e-6;s.dump(s.OUT/'source_calibration.json',record);s.dump(s.OUT/'calibration_manifest.json',{str(s.OUT/'source_calibration.json'):s.sha(s.OUT/'source_calibration.json')});s.dump(s.OUT/'history_audit.json',{'matching_N1_N2_history_found':False,'N0_exact_originalC_reuse':True,'new_trainings':2});s.dump(s.OUT/'status.json',{'status':'prepared_calibrated_preflight_pending'});(s.ROOT/'results'/s.OUT.name).symlink_to(s.OUT,target_is_directory=True);s.verify();print('COSINE_ATTENTION_PREPARED',scale.tolist(),flush=True)
if __name__=='__main__':main()
