"""Fit source-only tiny selectors, freeze target decisions/transport before GT."""
import importlib.util
import json
import shutil
import numpy as np
import torch
from common import HERE,OUT,BASE,DATA,ROUTERS,SEEDS,STAGES,KS,AUGS,CONFIG,NAMES,sha,ahash,dump,unit,confidence,TinySelector,decisions,majority,ecdf_scores,csvwrite

spec=importlib.util.spec_from_file_location('exact_knn_helper',HERE.parent/'houston_local_support_diagnostic_v1/freeze_neighbors.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)

@torch.inference_mode()
def source_routers(folder,sy,train,coords):
    ids=np.flatnonzero(train);labels=sy[train]
    excluded=np.max(abs(coords[:,None,:]-coords[ids][None,:,:]),axis=2)<=12
    assert (excluded.sum(1)>0).all() and ((~excluded).sum(1)>=15).all()
    assert np.all(excluded[ids,np.arange(len(ids))])
    em=torch.tensor(excluded,device='cuda');classcounts=np.stack([((~excluded)&(labels==c)[None,:]).sum(1) for c in range(7)],1)
    assert (classcounts>0).all()
    np.savez_compressed(folder/'source_reference_exclusion.npz',reference_indices=ids,excluded=excluded,eligible_class_counts=classcounts)
    cs={v:{} for v in range(4)};kn={v:{} for v in range(4)}
    for stage in STAGES:
        ref=unit(np.load(folder/f'view0_{stage}.npy').astype(float)[train]);r=torch.tensor(ref,device='cuda',dtype=torch.float64)
        for view in range(4):
            query=unit(np.load(folder/f'view{view}_{stage}.npy').astype(float));q=torch.tensor(query,device='cuda',dtype=torch.float64)
            cents=[]
            for c in range(7):
                cm=torch.tensor(labels==c,device='cuda');subset=r[cm]
                sums=subset.sum(0)[None,:]-em[:,cm].double()@subset
                norm=sums.norm(dim=1);assert (norm>1e-12).all()
                cents.append(((q*sums).sum(1)/norm).cpu().numpy())
            score=q@r.T;score[em]=-torch.inf
            ix,vs,nt=helper.select_neighbors(score);assert not excluded[np.arange(2520)[:,None],ix].any()
            c=np.stack([(labels[ix]==j).sum(1) for j in range(7)],1)
            cs[view][stage]=np.stack(cents,1)
            kn[view][stage]={k:np.stack([(labels[ix[:,:k]]==j).sum(1) for j in range(7)],1) for k in KS}
            np.savez_compressed(folder/f'view{view}_{stage}_router_scores.npz',centroid_scores=cs[view][stage],source_indices=ids[ix],source_cosines=vs,
                counts_k5=kn[view][stage][5],counts_k15=kn[view][stage][15],cutoff_tie_rows=nt)
            del q,score,sums
        del r;torch.cuda.empty_cache()
        print(folder.name,stage,'source nonoverlap routers frozen',flush=True)
    return cs,kn

def train_selector(folder,k,x,pred,native,y,train):
    mask=np.tile(train,4);xt=x[mask];yt=y[mask];pt=pred[mask]
    mean=xt.mean(0);std=xt.std(0);std[std<1e-8]=1
    norm=(x-mean)/std
    r0correct=pt[:,0]==yt;correct=pt[:,1:]==yt[:,None]
    recoverable=~r0correct&correct.any(1);gate_label=recoverable.astype(np.int64)
    seed=int(folder.name.split('_')[-1]);rng=np.random.default_rng(seed+30000+k);torch.manual_seed(seed+30000+k)
    model=TinySelector();assert sum(p.numel() for p in model.parameters())==808
    opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
    assert sum(p.numel() for g in opt.param_groups for p in g['params'])==808
    inputs=torch.tensor(norm[mask],dtype=torch.float32);labels=torch.tensor(gate_label);pos=torch.tensor(recoverable)
    altcorrect=torch.tensor(correct);history=[]
    for epoch in range(1,61):
        model.train();losses=[]
        for start in range(0,len(xt),256):
            if start==0:order=rng.permutation(len(xt))
            ix=order[start:start+256];g,a=model(inputs[ix]);lg=torch.nn.functional.cross_entropy(g,labels[ix])
            use=pos[ix]
            if use.any():
                logp=torch.log_softmax(a[use],dim=1);valid=altcorrect[ix][use]
                la=-torch.logsumexp(logp.masked_fill(~valid,-torch.inf),dim=1).mean()
            else:la=a.sum()*0
            loss=lg+la;opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            losses.append([float(lg.detach()),float(la.detach())])
        history.append({'seed':seed,'k':k,'epoch':epoch,'gate_CE':float(np.mean(losses,0)[0]),'alternative_CE':float(np.mean(losses,0)[1])})
    model.eval();torch.save({'selector':model.state_dict(),'mean':mean,'std':std,'config':CONFIG},folder/f'selector_k{k}_epoch60.pth')
    np.savez_compressed(folder/f'source_supervision_k{k}.npz',gate_labels=gate_label,correct_alternative_mask=correct,source_training_rows=np.flatnonzero(mask),
        normalization_mean=mean,normalization_std=std,native_confidence_ECDF_reference=native[mask])
    route,router,g,a=decisions(model,norm,pred)
    np.savez_compressed(folder/f'source_selector_readouts_k{k}.npz',routing=route,selected_router=router,gate_logits=g,alternative_logits=a)
    audit=[]
    for split,sm in [('train',mask),('heldout',~mask)]:
        for view in range(4):
            m=sm&(np.repeat(np.arange(4),2520)==view)
            r0=pred[m,0];rr=route[m];yy=y[m];override=router[m]>0
            audit.append({'seed':seed,'k':k,'split':split,'view':AUGS[view]['name'],'n':int(m.sum()),
                'R0_OA':float(100*(r0==yy).mean()),'selector_routing_OA':float(100*(rr==yy).mean()),
                'override_fraction':float(override.mean()),'recoverable_R0_errors':int(((r0!=yy)&(pred[m,1:]==yy[:,None]).any(1)).sum()),
                'n_R0_error':int((r0!=yy).sum()),'n_recovered':int(((r0!=yy)&(rr==yy)).sum()),
                'n_damaged':int(((r0==yy)&(rr!=yy)).sum())})
    print(seed,'k',k,'selector epoch60 source-only complete; recoverable training errors',int(recoverable.sum()),flush=True)
    return model,mean,std,native[mask],history,audit

def copy_input(src,dst,gate):
    rel=str(src.relative_to(ROUTERS));assert sha(src)==gate['files_sha256'][rel]
    shutil.copyfile(src,dst);assert sha(dst)==gate['files_sha256'][rel]

def target_freeze(folder,k,model,mean,std,reference,gate):
    seed=int(folder.name.split('_')[-1]);src=ROUTERS/f'seed_{seed}'
    logits=np.load(folder/'target_logits.npy').astype(float)
    cs={stage:np.load(src/f'{r}_centroid_scores.npz')['target_scores'] for stage,r in [('V','R5'),('T','R3'),('Z','R1')]}
    cnt={stage:np.load(src/f'{r}_neighbors.npz')[f'counts_k{k}'] for stage,r in [('V','R6'),('T','R4'),('Z','R2')]}
    for stage,r in [('V','R5'),('T','R3'),('Z','R1')]:
        rel=f'seed_{seed}/{r}_centroid_scores.npz';assert sha(ROUTERS/rel)==gate['files_sha256'][rel]
    for stage,r in [('V','R6'),('T','R4'),('Z','R2')]:
        rel=f'seed_{seed}/{r}_neighbors.npz';assert sha(ROUTERS/rel)==gate['files_sha256'][rel]
    x,pred,native,names=confidence(logits,cs,cnt)
    for j,r in enumerate(NAMES):
        rel=f'seed_{seed}/'+(f'{r}_k{k}_predictions.npy' if r in ['R2','R4','R6'] else f'{r}_predictions.npy')
        assert sha(ROUTERS/rel)==gate['files_sha256'][rel]
        assert np.array_equal(pred[:,j],np.load(ROUTERS/rel))
    np.savez_compressed(folder/f'target_gate_inputs_k{k}.npz',inputs=x,standardized_inputs=(x-mean)/std,
        router_predictions=pred,native_confidence=native,feature_names=np.array(names))
    r0=pred[:,0];routes={'G0':r0,'D00_R0':r0};router_ids={}
    routes['G1'],tied=majority(pred);router_ids['G1']=np.full(len(pred),-1)
    ecdf=ecdf_scores(native,reference);j=ecdf.argmax(1);routes['G2']=pred[np.arange(len(pred)),j];router_ids['G2']=j
    routes['G3'],router_ids['G3'],g,a=decisions(model,(x-mean)/std,pred)
    np.savez_compressed(folder/f'target_selector_aux_k{k}.npz',G1_tied=tied,G2_ECDF=ecdf,G3_gate_logits=g,G3_alternative_logits=a,G3_selected_router=router_ids['G3'])
    z=np.load(folder/'target_rawZ.npy').astype(float);stats=np.load(folder/'predicted_factors.npz');delta=stats['source_means']-stats['pseudo_target_means']
    head=np.load(folder/'frozen_head.npz');w=head['weight'].astype(float);b=head['bias'].astype(float)
    assert np.array_equal((z@w.T+b).argmax(1),r0)
    for method,route in routes.items():
        transformed=z if method=='G0' else z+delta[route]
        scores=logits if method=='G0' else transformed@w.T+b
        if method=='D00_R0':
            baseline=np.load(src/'D00_readouts.npz');assert np.array_equal(scores,baseline['logits'])
        np.save(folder/f'{method}_k{k}_rawZ.npy',transformed)
        np.savez_compressed(folder/f'{method}_k{k}_readouts.npz',routing=route,classifier_logits=scores,predictions=scores.argmax(1),
            route_changed=route!=r0,selected_router=router_ids.get(method,np.zeros(len(r0),int)))
    print(seed,'k',k,'all target gate inputs/decisions/transport/final predictions frozen',flush=True)

def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;helper.check_knn()
    sg=json.loads((OUT/'ALL_SOURCE_FEATURES_FROZEN.json').read_text());assert not sg['target_GT_opened']
    for rel,h in sg['files_sha256'].items():assert sha(OUT/rel)==h
    parent=json.loads((ROUTERS/'ALL_ROUTING_TRANSPORT_FROZEN.json').read_text());assert not parent['target_GT_opened_by_constructor']
    ep=json.loads((OUT/'extraction_provenance.json').read_text());sy=np.load(OUT/'source_labels.npy');train=np.load(OUT/'source_train_mask.npy');coords=np.load(OUT/'source_coords.npy')
    assert ahash(sy)==ep['historical_metadata']['domains']['source']['labels_sha256']
    prov={'fixed_config':CONFIG,'feature_names':None,'parent_router_gate_sha256':sha(ROUTERS/'ALL_ROUTING_TRANSPORT_FROZEN.json'),
        'code_sha256':{str(p):sha(p) for p in HERE.glob('*') if p.is_file()},'historical_metadata':ep['historical_metadata'],'seeds':ep['seeds']}
    history,source_audits=[],[]
    for seed in SEEDS:
        folder=OUT/f'seed_{seed}';rec=ep['seeds'][str(seed)];assert sha(rec['checkpoint'])==rec['checkpoint_sha256']
        cs,kn=source_routers(folder,sy,train,coords)
        for name in ['target_logits.npy','target_predictions.npy','target_rawZ.npy','target_coords.npy','target_sample_indices.npy','frozen_head.npz','predicted_factors.npz']:
            copy_input(ROUTERS/f'seed_{seed}/{name}',folder/name,parent)
        for k in KS:
            inputs,preds,natives=[],[],[]
            for view in range(4):
                logits=np.load(folder/f'view{view}_logits.npy').astype(float)
                x,p,n,names=confidence(logits,cs[view],{stage:kn[view][stage][k] for stage in STAGES})
                inputs.append(x);preds.append(p);natives.append(n)
            x=np.concatenate(inputs);pred=np.concatenate(preds);native=np.concatenate(natives);y=np.tile(sy,4)
            prov['feature_names']=names
            np.savez_compressed(folder/f'source_gate_inputs_k{k}.npz',inputs=x,router_predictions=pred,native_confidence=native,feature_names=np.array(names),source_origin_index=np.tile(np.arange(2520),4))
            model,mean,std,ref,h,a=train_selector(folder,k,x,pred,native,y,train);history.extend(h);source_audits.extend(a)
            target_freeze(folder,k,model,mean,std,ref,parent)
        assert sha(rec['checkpoint'])==rec['checkpoint_sha256']
    csvwrite('selector_training_history.csv',history);csvwrite('source_selector_diagnostic.csv',source_audits)
    dump(OUT/'provenance.json',prov)
    dump(OUT/'ALL_TARGET_DECISIONS_FROZEN.json',{'all_seeds_completed':True,'target_GT_opened':False,
       'main_model_unchanged':True,'only_source_GT_supervised_808_parameter_selectors':True,
       'D00_pseudo_statistics_unchanged':True,'no_hyperparameter_selection_or_followup':True,
       'files_sha256':{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()}})
    print('All selectors and target inputs/routing/transport/predictions hash-frozen; target GT unopened.',flush=True)

if __name__=='__main__':main()
