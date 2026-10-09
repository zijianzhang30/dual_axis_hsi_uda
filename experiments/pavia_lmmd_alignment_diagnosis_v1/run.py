"""Read-only checkpoint evaluation and gradient queries; original files never written."""
import argparse
import gc
import hashlib
import json
import math
import time
import numpy as np
import torch
import torch.nn.functional as F
import diagnostics as d

def load(seed,kind,epoch):
    folder=d.BASE/('C_LMMD_%d'%seed) if kind=='LMMD' else d.c.run_folder('C_CE',seed)
    m=json.loads((folder/'frozen_manifest.json').read_text());item=next(v for v in m['checkpoints'] if v['epoch']==epoch)
    assert d.sha(item['path'])==item['sha256']
    model,_=d.c.build(seed,'C');state=torch.load(item['path'],map_location='cpu',weights_only=False)
    assert state['epoch']==epoch;model.load_state_dict(state['model'],strict=True)
    return model.cuda(),folder

def train_queries(model,seed,epoch,cpu,rng,b,mmd,aug,folder):
    model.train();before=d.c.state_hash(model)
    buffers={k:v.clone() for k,v in model.named_buffers()};current=dict(model.named_buffers())
    source,target=b.train_sets('P2',seed);loaders,plan=b.epoch_loaders(source,target,seed,epoch)
    # Offline GT audit only. Dummy target dataset labels are never supplied to main losses.
    all_gt=np.load(d.c.P2/'inputs/evaluation_labels_OFFLINE_ONLY.npy')
    params,slices=d.parameter_slices(model);scale=.3*(2/(1+math.exp(-10*epoch/100))-1)
    allp=[];eval_same_p=[];ally=[];membership=[];gradients=[];feature_rows=[];rnghash=hashlib.sha256();viewhash=hashlib.sha256()
    for step,((x,y),(t,_)) in enumerate(zip(*loaders),1):
        start_offset=d.set_query_rng(cpu,rng,epoch,step)
        rnghash.update(torch.get_rng_state().numpy().tobytes());rnghash.update(torch.cuda.get_rng_state().numpy().tobytes())
        vv=d.c.views(x,t,seed,epoch,step,aug)
        for sx,tx in vv:viewhash.update(sx.numpy().tobytes());viewhash.update(tx.numpy().tobytes())
        viewhash.update(y.numpy().tobytes())
        with torch.set_grad_enabled(step<=d.GRADIENT_BATCHES):
            rec=[model(sx.cuda(),tx.cuda()) for sx,tx in vv];main=rec[0];sy=y.cuda()
            p=main['lt'].softmax(1)
            # Exactly the main source CE and official LMMD, no target GT input.
            ce=F.cross_entropy(main['ls'],sy)
            lmmd=mmd.lmmd(main['fs'],main['ft'],sy,p,BATCH_SIZE=32,CLASS_NUM=7)
            terms,active,tv=d.class_terms(mmd,main,sy,p)
            assert abs(float((sum(terms.values())-lmmd).detach()))<2e-5
            # All target GT use starts here and remains diagnostic,with no update.
            indices=plan[1][(step-1)*32:step*32];truth=all_gt[indices]
            pn=p.detach().cpu().numpy();pred=pn.argmax(1)
            allp.append(pn);ally.append(truth)
            for k in range(7):
                membership.append({'seed':seed,'epoch':epoch,'batch':step,'class':k+1,
                    'source_available':bool((y.numpy()==k).any()),'argmax_target_available':bool((pred==k).any()),
                    'active':k in active,'active_class_count':len(active),
                    'native_class_loss_mass':1/len(active) if k in active else 0.,
                    'normalized_target_weight_purity':float(tv[truth==k,k].sum()) if k in active else None,
                    'class_loss_contribution':float(terms[k].detach()) if k in terms else 0.})
            if step<=d.GRADIENT_BATCHES:
                # Oracle uses GT only to diagnose weight/direction error. Never trained.
                oracle_p=F.one_hot(torch.from_numpy(truth).cuda(),7).float()
                oracle=mmd.lmmd(main['fs'],main['ft'],sy,oracle_p,BATCH_SIZE=32,CLASS_NUM=7)
                oterms,oactive,_=d.class_terms(mmd,main,sy,oracle_p)
                assert abs(float((sum(oterms.values())-oracle).detach()))<2e-5
                gce=d.flat_grad(ce,params);gl=d.flat_grad(lmmd,params);go=d.flat_grad(oracle,params)
                row={'seed':seed,'epoch':epoch,'batch':step,'CE':float(ce.detach()),'LMMD':float(lmmd.detach()),
                    'oracle_LMMD_diagnostic_only':float(oracle.detach()),'LMMD_weight':scale,
                    'CE_vs_LMMD':d.comparison(gce,gl,scale),'actual_vs_oracle_parameter_gradient':d.comparison(gl,go),
                    'blocks':{},'classes':{}}
                for name,parts in slices.items():
                    ca=torch.cat([gce[a:z] for a,z in parts]);la=torch.cat([gl[a:z] for a,z in parts])
                    row['blocks'][name]=d.comparison(ca,la,scale)
                for k in (3,6):
                    if k in active and (sy==k).any():
                        classce=F.cross_entropy(main['ls'][sy==k],sy[sy==k],reduction='sum')/32
                        gc_k=d.flat_grad(classce,params);gl_k=d.flat_grad(terms[k],params)
                        item={'active':True,'source_CE_component':float(classce.detach()),
                            'native_LMMD_component':float(terms[k].detach()),'CE_class_vs_LMMD_class':d.comparison(gc_k,gl_k,scale),
                            'all_CE_vs_LMMD_class':d.comparison(gce,gl_k,scale)}
                        if k in oactive:
                            go_k=d.flat_grad(oterms[k],params)
                            item['actual_vs_oracle_class_gradient']=d.comparison(gl_k,go_k)
                            del go_k
                        row['classes'][str(k+1)]=item
                        del classce,gc_k,gl_k
                    else:row['classes'][str(k+1)]={'active':False}
                gf=torch.autograd.grad(lmmd,main['ft'],retain_graph=True)[0].detach().cpu().numpy()
                of=torch.autograd.grad(oracle,main['ft'],retain_graph=True)[0].detach().cpu().numpy()
                sf=main['fs'].detach().cpu().numpy();tf=main['ft'].detach().cpu().numpy()
                for i in range(32):
                    k=int(truth[i]);norm=np.linalg.norm(gf[i])*np.linalg.norm(of[i]);sm=y.numpy()==k
                    delta=tf[i]-sf[sm].mean(0) if sm.any() else None
                    feature_rows.append({'seed':seed,'epoch':epoch,'batch':step,'target_index':int(indices[i]),
                        'GT_class':k+1,'predicted_class':int(pred[i])+1,'pseudo_correct':bool(pred[i]==k),
                        'confidence':float(pn[i].max()),'correct_class_probability':float(pn[i,k]),
                        'actual_vs_oracle_feature_gradient_cosine':float(np.dot(gf[i],of[i])/norm) if norm>1e-15 else None,
                        'actual_gradient_towards_true_source_centroid_dot':float(np.dot(gf[i],delta)) if delta is not None else None,
                        'oracle_gradient_towards_true_source_centroid_dot':float(np.dot(of[i],delta)) if delta is not None else None})
                gradients.append(d.clean(row))
                assert all(p.grad is None for p in params)
                del gce,gl,go,oracle,oterms,row
        # BN uses batch stats in train-mode queries but every buffer is restored immediately.
        with torch.no_grad():
            for k,v in buffers.items():current[k].copy_(v)
            # Same augmented patches: combined batch-BN/dropout vs checkpoint eval semantics.
            model.eval();re=model(x.cuda(),t.cuda())
            eval_same_p.append(re['lt'].softmax(1).cpu().numpy());del re
            model.train()
        expected=start_offset+3*rng['offset_per_paired_forward']
        assert int(np.frombuffer(torch.cuda.get_rng_state().numpy().tobytes(),dtype='<u8')[1])==expected
        del rec,main,ce,lmmd,terms,p,vv
    hist=[json.loads(r) for r in (folder/'history.jsonl').read_text().splitlines()][epoch-1]
    assert rnghash.hexdigest()==hist['model_rng_sha256']
    assert viewhash.hexdigest()==hist['all_input_views_sha256']
    assert d.c.trace_hash(plan)==hist['plan_sha256']
    assert before==d.c.state_hash(model) and all(p.grad is None for p in params)
    return {'reliability':d.reliability(np.concatenate(allp),np.concatenate(ally)),
        'eval_on_identical_augmented_target_patches':d.reliability(np.concatenate(eval_same_p),np.concatenate(ally)),
        'membership':d.clean(membership),'gradients':gradients,'feature_gradient_samples':d.clean(feature_rows),
        'audit':{'state_sha256_before_after':before,'BN_buffers_restored_each_query':True,
            'no_optimizer_or_parameter_updates':True,'no_grad_accumulation':True,'model_RNG_views_plan_match_history':True,
            'GT_only_in_offline_weight_oracle_and_reporting':True,'post_epoch_weights_not_historical_step_weights':True}}

def main():
    arg=argparse.ArgumentParser();arg.add_argument('--seed',type=int,choices=d.SEEDS,required=True);a=arg.parse_args()
    d.verify_files();folder=d.OUT/('seed_%d'%a.seed);folder.mkdir(exist_ok=False)
    torch.set_num_threads(2);torch.cuda.set_device(0);start=time.monotonic()
    cpu,rng=d.rng_audit(a.seed);d.write(folder/'rng_audit.json',rng)
    with np.load(d.OUT/'fixed_samples.npz') as z:
        sc=z['source_coords'];sy=z['source_labels'];tc=z['target_coords'];ty=z['target_labels'];indices=z['target_indices']
    b=d.c.base();_,mmd,_,aug=d.c.official()
    source=b.PatchSet('P2','source',sc);target=b.PatchSet('P2','target',tc)
    results={};mode_labels=None
    for kind,epoch in [('CE',100)]+[('LMMD',e) for e in d.PHASES]:
        key='%s%03d'%(kind,epoch);model,original_folder=load(a.seed,kind,epoch)
        before=d.c.state_hash(model);sf,sl=d.extract(model,source);tf,tl=d.extract(model,target)
        assert d.c.state_hash(model)==before
        if epoch==100:
            with np.load(original_folder/'predictions_epoch100.npz') as z:
                error=float(np.max(np.abs(tl-z['logits'][indices])))
                assert error<1e-6,(key,error)
                # Population calibration reuses existing formal prediction,not new checkpoint selection.
                full=d.reliability(d.probabilities(z['logits']),np.load(d.c.P2/'inputs/evaluation_labels_OFFLINE_ONLY.npy'))
        else:error=None;full=None
        prob=d.probabilities(tl)
        if kind=='CE':mode_labels=d.modes(tf,ty);np.save(folder/'fixed_CE_target_modes.npy',mode_labels)
        np.savez_compressed(folder/('features_'+key+'.npz'),source=sf,source_logits=sl,target=tf,target_logits=tl)
        result={'seed':a.seed,'kind':kind,'epoch':epoch,'eval_reliability':d.reliability(prob,ty),
            'full_population_reliability_epoch100':full,'geometry':d.geometry(sf,tf,sy,ty,prob),
            'fixed_CE_mode_statistics':d.mode_statistics(tf,prob,ty,mode_labels),
            'formal_epoch100_logit_max_error':error,'eval_complete_state_unchanged':True}
        if kind=='LMMD':result['train_mode_diagnosis']=train_queries(model,a.seed,epoch,cpu,rng,b,mmd,aug,original_folder)
        assert d.c.state_hash(model)==before
        d.write(folder/(key+'.json'),result);results[key]=str(folder/(key+'.json'))
        print(json.dumps({'seed':a.seed,'phase':key,'state_unchanged':True,'seconds':time.monotonic()-start}),flush=True)
        del model,sf,sl,tf,tl,result;gc.collect();torch.cuda.empty_cache()
    d.verify_files()
    d.write(folder/'complete.json',{'seed':a.seed,'results':results,'seconds':time.monotonic()-start,
        'original_checkpoint_and_input_hashes_unchanged':True,'no_training':True})

if __name__=='__main__':main()
