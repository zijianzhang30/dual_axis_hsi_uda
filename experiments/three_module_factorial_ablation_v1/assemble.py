import argparse,json
import numpy as np,torch
import torch.nn.functional as F
import study as s
def main(d):
 s.verify();torch.set_num_threads(2);records=[]
 for seed in s.SEEDS:
  for r in s.RELATIONS:
   archived=s.reused_relation(d,r,seed)
   if archived is None:
    mf=json.loads((s.OUT/d/('%s_%d'%(r,seed))/'TRAINING_PREDICTIONS_FROZEN.json').read_text());assert mf['epochs']==200 and mf['all200_reference_data_and_RNG_match'];records.extend([dict(q,dataset=d,relation=r,seed=seed) for q in mf['predictions']]);continue
   f=s.OUT/d/('%s_%d'%(r,seed));f.mkdir(parents=True,exist_ok=False);(f/'predictions').mkdir();local=[]
   if r=='J_A':
    m,values,rt=s.setup(d,r,seed);m.cuda().eval();ds=values[-1].dataset;ds.label=np.zeros_like(ds.label);ds.labels=np.zeros_like(ds.labels);loader=torch.utils.data.DataLoader(ds,batch_size=128,shuffle=False,num_workers=0,generator=torch.Generator().manual_seed(seed+185000))
   for epoch in range(10,201,10):
    for scope in (('native','full') if d=='hyrank' else ('native',)):
     rec=s.spec_file(d,seed,epoch,scope);sp=np.load(rec['path']);spec=sp['R1_logits'];xy=sp['coordinates']
     if r=='J_C':rel=sp['R0_logits']
     else:
      assert np.array_equal(xy,ds.indices);cp=archived/'checkpoints'/('epoch_%03d.pth'%epoch);m.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True)['model']);before=rt['hash'](m);rel=[]
      with torch.inference_mode():
       for x,dummy in loader:
        assert not dummy.any();x=x.cuda();rel.append(m(x,x)[1].cpu().numpy())
      rel=np.concatenate(rel);assert rt['hash'](m)==before
      if epoch==200:assert np.array_equal(rel,np.load(archived/'predictions_200.npz')['logits'])
     a=torch.from_numpy(rel.astype(np.float64));b=torch.from_numpy(spec.astype(np.float64));lf=F.log_softmax(F.log_softmax(a,1)+F.log_softmax(b,1),1).numpy();path=f/'predictions'/('%s_epoch_%03d.npz'%(scope,epoch));np.savez_compressed(path,coordinates=xy,relation_logits=rel,spectral_logits=spec,fusion_logits=lf,relation_predictions=rel.argmax(1),spectral_predictions=spec.argmax(1),fusion_predictions=lf.argmax(1),agreement=rel.argmax(1)==spec.argmax(1));q={'dataset':d,'relation':r,'seed':seed,'epoch':epoch,'scope':scope,'path':str(path),'sha256':s.sha(path),'relation_reused_from':str(archived),'immutable_spectral_prediction':rec};records.append(q);local.append(q)
    print('REUSE_GRID_FROZEN',d,r,seed,epoch,flush=True)
   s.dump(f/'REUSED_PREDICTIONS_FROZEN.json',{'no_new_training':True,'records':local,'target_GT_not_scored':True})
   if r=='J_A':del m,values;torch.cuda.empty_cache()
 for q in records:assert s.sha(q['path'])==q['sha256']
 s.dump(s.OUT/('ALL_'+d+'_PREDICTIONS_FROZEN.json'),{'records':records,'all_four_relation_combinations':True,'all9_spectral_experts_globally_reused':True,'target_GT_not_scored':True})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=s.DATASETS,required=True);main(p.parse_args().dataset)
