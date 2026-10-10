import json,shutil,numpy as np
from pathlib import Path
import common as c

def main():
 c.OUT.mkdir(exist_ok=False);fixed=json.loads((c.DATA.parent/'fixed_inputs.json').read_text());files={};cube={}
 for domain in ('source','target'):
  path=c.DATA/(domain+'_cube.npy');files[str(path)]=c.sha(path);cube[domain]=np.load(path,mmap_mode='r')[6:-6,6:-6];assert list(cube[domain].shape[:2])==fixed['domains'][domain]['shape']
 valid,hh,vv=c.angles(cube['source']);hv=valid[:,:-1]&valid[:,1:];vvv=valid[:-1]&valid[1:];d=np.concatenate((hh[hv],vv[vvv]));positive=d[d>1e-8];assert len(positive)>0;tau=max(float(np.median(positive)),1e-6)
 plan={'dataset':'Houston','seeds':list(c.SEEDS),'checkpoint':'formal C+FullJointBN+sourceCE epoch200 only','read_only':'eval,inference_mode,all main params frozen,BN buffers unchanged,no optimizer','groups':['A original','B uniform4neighbors','C source-calibrated SAM4neighbors'],'alpha':.5,'passes':1,'neighbor_order':['up','down','left','right'],'weight':'exp(-SAM_angle_squared/tau_squared),normalized percenter;stable logweights;no hardedge or confidence threshold','tau':tau,'tau_selection':'median positive valid source-image4neighbor edge SAM angles in radians,floor1e-6;source GT not used','inference_nodes':'all physical image pixels,never GT-mask nodes;nonfinite/zero spectral pixels disconnected based image only;no-edge nodes retain original probabilities','padding':'same inherited reflect6 for CNN patches;graph only actual physical4neighbors,no reflected graph edges','prediction_pairing':'eval target,target exactly historical C;13patch48bands,batch128,noaugmentation','evaluation':'same historical native52901 labeled pixels across A/B/C;GT read only after3seed full-scene outputs frozen','boundary':'offline labeled4neighbor hasdifferentlabel;strict interior all4labeled same;unknown groupseparate;not claim unlabeled neighbors semantic purity','target_best':'not performed in this first fixed200 diagnostic;no targetcheckpoint selection','followups':'no newtraining,GNN,newscales,Pavia,HyRANK,SND or extra checkpoints'}
 c.dump(c.OUT/'frozen_plan.json',plan);c.dump(c.OUT/'source_tau_audit.json',{'tau_radians':tau,'valid_source_nodes':int(valid.sum()),'source_edges':len(d),'positive_edges':len(positive),'SAM_min_mean_max':list(map(float,(d.min(),d.mean(),d.max()))),'source_GT_used':False})
 n,u,w,degree,v=c.graph(cube['target'],tau)
 for name,arr in [('physical_shape',np.asarray(cube['target'].shape[:2])),('neighbors',n),('uniform_weights',u),('spectral_weights',w),('degree',degree),('spectral_valid',v),('physical_coords',np.indices(v.shape).reshape(2,-1).T)]:np.save(c.OUT/(name+'.npy'),arr)
 c.dump(c.OUT/'graph_audit.json',{'physical_pixels':int(v.size),'spectrally_valid_pixels':int(v.sum()),'directed_edges':int(degree.sum()),'degree_counts':np.bincount(degree,minlength=5).tolist(),'target_GT_used':False,'nodes_selected_from_GT':False,'target_cube_shape':list(cube['target'].shape)})
 # Geometry/no-propagation preflight with a rectangular synthetic scene.
 toy=np.ones((3,4,2));nn,uu,ww,dd,vv=c.graph(toy,tau);assert sorted(nn[0][uu[0]>0].tolist())==[1,4];assert sorted(nn[11][uu[11]>0].tolist())==[7,10]
 pp=np.tile(np.array([.2,.8]),(12,1));assert np.allclose(c.aggregate(pp,nn,uu),pp) and np.array_equal(c.aggregate(pp,nn,ww,0.),pp)
 impulse=np.tile(np.array([1.,0.]),(12,1));impulse[0]=[0,1];r=c.aggregate(impulse,nn,uu);assert r[2,1]==0 and r[1,1]>0
 c.dump(c.OUT/'preflight.json',{'passed':True,'corner_bounds_correct':True,'constant_probabilities_preserved':True,'alpha0_exact_identity':True,'one_pass_synchronous_no_second_hop':True,'physical_grid_only':True})
 for seed in c.SEEDS:
  path=c.BASE/'houston'/('C_%d/checkpoints/epoch_200.pth'%seed);assert c.sha(path)==fixed['checkpoints'][str(seed)]['sha256'];files[str(path)]=c.sha(path)
 for p in list(c.BASE.glob('*.py'))+list((c.BASE/'snapshot').rglob('*.py')):files[str(p)]=c.sha(p)
 snap=c.OUT/'experiment_code';snap.mkdir()
 for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,snap/p.name)
 for p in list(snap.glob('*.py'))+list(c.OUT.glob('*.npy'))+[c.OUT/'frozen_plan.json']:files[str(p)]=c.sha(p)
 c.dump(c.OUT/'input_and_code_manifest.json',files);link=Path('/home/zhangzj26/dual_axis_hsi_uda/results/houston_scene_relation_diagnostic_v1');link.symlink_to(c.OUT);print('LOCKED_SOURCE_TAU',tau,'FULL_SCENE_PIXELS',v.size,flush=True)
if __name__=='__main__':main()
