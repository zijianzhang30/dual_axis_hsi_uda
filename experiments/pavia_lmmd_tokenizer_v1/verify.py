"""Audit shared initial tensors/streams, LMMD graph, token/value shapes before training."""
import gc
import json
import torch
import torch.nn.functional as F
import common as c

def main():
    c.forbid_target_label_loading();c.verify_files();c.audit_reuse()
    torch.set_num_threads(2);torch.cuda.set_device(0)
    b=c.base();_,mmd,contrast,aug=c.official();con=contrast.SupConLoss(temperature=.1);checks=[]
    for seed in c.SEEDS:
        source,target=b.train_sets('P2',seed);loaders,plan=b.epoch_loaders(source,target,seed,1)
        x,y=next(iter(loaders[0]));t,_=next(iter(loaders[1]));vv=c.views(x,t,seed,1,1,aug)
        shared=[];cpu=[];cuda=[];values=[];counts=[]
        for arm in c.NEW_ARMS:
            tokenizer=arm[0];model,initial=c.build(seed,tokenizer)
            shared.append(c.shared_hash(model));cpu.append(c.digest(torch.get_rng_state().numpy()))
            old=json.loads((c.run_folder('C_CE',seed)/'config.json').read_text())
            if tokenizer=='C':
                assert initial==old['initial_base_model_sha256'] and c.state_hash(model)==old['initial_complete_model_sha256']
                assert model.backbone.relation_scores.anchor(x[:,0]).shape==(32,102,1,1)
                assert torch.equal(model.backbone.relation_scores.anchor(x[:,0])[:,:,0,0],x[:,0,:,5,5])
            else:assert hasattr(model.backbone,'conv_a') and not hasattr(model.backbone,'relation_scores')
            params=sum(v.numel() for v in model.parameters())
            assert params==(645211 if tokenizer=='C' else 642039)
            model=model.cuda().train();v_maps=[];sequences=[]
            hook=model.backbone.conv2d_features[2].register_forward_hook(lambda m,i,o:v_maps.append(o.detach()))
            pre=model.backbone.blocks[0].register_forward_pre_hook(lambda m,i:sequences.append(tuple(i[0].shape)))
            rec=[]
            for i,(sx,tx) in enumerate(vv):
                tx=tx.cuda().requires_grad_();r=model(sx.cuda(),tx)
                if i==0:tg=tx
                assert r['fs'].shape==r['ft'].shape==(32,64) and r['ls'].shape==r['lt'].shape==(32,7)
                rec.append(r)
            assert len(v_maps)==3 and all(v.shape==(64,64,11,11) for v in v_maps)
            assert len(sequences)==3 and all(z==(32,5,64) for z in sequences)
            values.append([c.digest(v.cpu().numpy()) for v in v_maps]);cuda.append(c.digest(torch.cuda.get_rng_state().cpu().numpy()))
            bc=c.bn_counts(model);counts.append(bc);assert all(n==3 for n in bc.values())
            loss,terms=c.objective(rec,y.cuda(),1,arm,mmd,con)
            ce=F.cross_entropy(rec[0]['ls'],y.cuda())
            native=mmd.lmmd(rec[0]['fs'],rec[0]['ft'],y.cuda(),rec[0]['lt'].softmax(1),BATCH_SIZE=32,CLASS_NUM=7)
            expected=ce+.3*terms['lambda']*native;assert torch.equal(loss,expected)
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            assert float(tg.grad.abs().sum())>0
            assert all(p.grad is None for h in (model.head1,model.head2) for p in h.parameters())
            tok=model.backbone.relation_scores if tokenizer=='C' else model.backbone.conv_a
            grad=sum(float(p.grad.abs().sum()) for p in tok.parameters() if p.grad is not None);assert grad>0
            head_before=[c.state_hash(h) for h in (model.head1,model.head2)]
            before=c.state_hash(model);opt=c.optimizer(model,'O');opt.step();assert before!=c.state_hash(model)
            assert head_before==[c.state_hash(h) for h in (model.head1,model.head2)]
            checks.append({'seed':seed,'arm':arm,'parameters':params,'initial_sha256':initial,
                'complete_initial_sha256':c.state_hash(c.build(seed,tokenizer)[0]),
                'shared_initial_sha256':shared[-1],'value_grid':[64,64,11,11],'token_shape':[32,4,64],
                'CLS_sequence_shape':[32,5,64],'tokenizer_grad_l1':grad,'main_target_grad_l1':float(tg.grad.abs().sum()),
                'official_LMMD_sum_exact':True,'unused_head_grad_None_and_no_updates':True})
            hook.remove();pre.remove()
            del model,opt,rec,r,tx,tg,loss,expected,ce,native,tok,v_maps,hook,pre
            gc.collect();torch.cuda.empty_cache()
        assert len(set(shared))==len(set(cpu))==len(set(cuda))==1
        assert values[0]==values[1] and counts[0]==counts[1]
        del source,target,loaders
    c.dump(c.OUT/'verification.json',{'passed':True,'checks':checks,'target_GT_values_accessed':False,
        'shared_A_C_initial_parameters_CPU_CUDA_RNG_CNN_values_BN_exact':True,
        'only_trainable_structure_difference_is_tokenizer':True,'parameters_delta_C_minus_A':3172,
        'LMMD_weight_schedule_unchanged':True,'center11x11_anchor_asserted':True})
    print('All6 preflights passed; C first, A second, no target-based branching.',flush=True)

if __name__=='__main__':main()
