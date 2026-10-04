"""One frozen task; log only actually consumed draws and RNG states."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
import common as c


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--arm',choices=list(c.ARMS),required=True)
    parser.add_argument('--seed',type=int,choices=[2100,2101,2102],required=True)
    parser.add_argument('--device',type=int,required=True)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    assert json.loads((c.OUT/'verification.json').read_text())['passed']
    run=c.OUT/('%s_%d'%(args.arm,args.seed))
    assert not (run/'history.jsonl').exists(), 'No overwrite or automatic restart'
    config=json.loads((run/'config.json').read_text())
    for file,h in config['source_files_sha256'].items():
        assert c.helper.file_hash(file)==h, 'Frozen source changed: '+file
    torch.set_num_threads(2)
    device=torch.device('cuda:%d'%args.device); torch.cuda.set_device(device)
    norm,policy=c.ARMS[args.arm]
    start=time.monotonic()
    native=c.datasets(args.seed,norm)
    pool=json.loads((c.OUT/'pool.json').read_text())
    assert [c.digest(x.indices) for x in native]==pool['center_hashes']
    labels=np.asarray(native[0].labels,dtype=np.int64)
    samplers=[c.DrawSampler(),c.DrawSampler()]
    loaders=[]
    for i,(ds,sampler) in enumerate(zip([c.PlannedDataset(native[0]),c.PlannedDataset(native[2],True)],samplers)):
        loaders.append(torch.utils.data.DataLoader(ds,batch_size=128,sampler=sampler,
            num_workers=args.workers,generator=torch.Generator().manual_seed(args.seed+60000+i)))
    val=torch.utils.data.DataLoader(native[1],batch_size=128,num_workers=args.workers,
        generator=torch.Generator().manual_seed(args.seed+60002))
    target_eval=torch.utils.data.DataLoader(native[2],batch_size=128,num_workers=args.workers,
        generator=torch.Generator().manual_seed(args.seed+60003))
    model,initial=c.model(args.seed)
    assert initial==json.loads((c.OUT/'verification.json').read_text())['initial_model_hashes_match_repaired_B'][str(args.seed)]
    model.to(device)
    optimizer=torch.optim.SGD(model.parameters(),lr=.01)
    c.dump(run/'runtime.json',{'device':args.device,'initial_model_sha256':initial,
        'parameters':sum(p.numel() for p in model.parameters()),'source_pool_counts':np.bincount(labels,minlength=7).tolist(),
        'prep_seconds':time.monotonic()-start,'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,
        'tf32_cudnn':torch.backends.cudnn.allow_tf32})
    checkpoints=[]; seen=np.zeros(len(labels),dtype=np.int64)
    (run/'checkpoints').mkdir(); (run/'predictions').mkdir()
    torch.cuda.synchronize(device); start=time.monotonic()
    with (run/'history.jsonl').open('w') as history:
        for epoch in range(1,201):
            s,t,sa,ta=c.plan(args.seed,epoch,policy,labels,len(native[2]))
            for sampler,indices,aug in zip(samplers,(s,t),(sa,ta)):
                sampler.indices=indices; sampler.augmentation=aug
            model.train(); total=0.; steps=0
            actual_s=[]; actual_t=[]; actual_sa=[]; actual_ta=[]
            rng=hashlib.sha256()
            for (images,ys,si,ac),(target,unused,ti,tc) in zip(*loaders):
                assert len(images)==len(target)==128 and not unused.any()
                actual_s.append(si.numpy()); actual_t.append(ti.numpy())
                actual_sa.append(ac.numpy().astype(np.uint8)); actual_ta.append(tc.numpy().astype(np.uint8))
                rng.update(torch.get_rng_state().numpy().tobytes())
                rng.update(torch.cuda.get_rng_state(device).cpu().numpy().tobytes())
                images,ys,target=images.to(device),ys.to(device),target.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss=torch.nn.functional.cross_entropy(model(images,target)[0],ys)
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss; do not adjust protocol')
                loss.backward(); optimizer.step()
                total+=float(loss)*128; steps+=1
            arrays=[np.concatenate(v) for v in (actual_s,actual_t,actual_sa,actual_ta)]
            assert steps==290 and all(np.array_equal(x,y) for x,y in zip(arrays,(s,t,sa,ta)))
            visits=np.bincount(arrays[0],minlength=len(labels)); seen+=visits
            epoch_record={'epoch':epoch,'steps':steps,'samples':c.N,'source_ce':total/c.N,
                'source_index_sha256':c.digest(arrays[0]),'target_index_sha256':c.digest(arrays[1]),
                'source_center_sha256':c.digest(native[0].indices[arrays[0]]),
                'target_center_sha256':c.digest(native[2].indices[arrays[1]]),
                'source_augmentation_sha256':c.digest(arrays[2]),'target_augmentation_sha256':c.digest(arrays[3]),
                'model_rng_sha256':rng.hexdigest(),
                'source_draw_class_counts':np.bincount(labels[arrays[0]],minlength=7).tolist(),
                'source_unique_class_counts':np.bincount(labels[visits>0],minlength=7).tolist(),
                'source_unique_centers':int(np.count_nonzero(visits)),
                'cumulative_source_coverage':int(np.count_nonzero(seen))}
            if epoch==1 or epoch%10==0:
                epoch_record['source_val']=c.helper.source_evaluate(model,val,device)
                print(json.dumps(epoch_record),flush=True)
            if epoch%10==0:
                path=run/'checkpoints'/('epoch_%03d.pth'%epoch)
                torch.save({'model':model.state_dict(),'epoch':epoch,'config':config},path)
                checkpoints.append({'epoch':epoch,'path':str(path),'sha256':c.helper.file_hash(path)})
            history.write(json.dumps(epoch_record)+'\n'); history.flush()
    torch.cuda.synchronize(device); train_seconds=time.monotonic()-start
    np.savez_compressed(run/'source_actual_visits.npz',visits=seen,centers=native[0].indices,labels=labels)
    start=time.monotonic(); predictions=[]
    # Freeze the complete diagnostic grid before reading label values for metrics.
    for item in checkpoints:
        state=torch.load(item['path'],map_location=device)
        model.load_state_dict(state['model'])
        logits,pred=c.helper.predict(model,target_eval,device)
        path=run/'predictions'/('epoch_%03d.npz'%item['epoch'])
        np.savez_compressed(path,logits=logits,predictions=pred)
        predictions.append({'epoch':item['epoch'],'path':str(path),'sha256':c.helper.file_hash(path)})
    manifest={'checkpoints':checkpoints,'predictions':predictions,'train_seconds':train_seconds,
        'inference_seconds':time.monotonic()-start,'updates':58000,'source_total_draws':int(seen.sum()),
        'source_covered_centers':int(np.count_nonzero(seen)),
        'source_actual_class_counts':np.bincount(labels,weights=seen,minlength=7).astype(np.int64).tolist(),
        'source_covered_class_counts':np.bincount(labels[seen>0],minlength=7).tolist(),
        'parameters':sum(p.numel() for p in model.parameters()),
        'source_val_at_final':epoch_record['source_val'],
        'peak_cuda_memory_bytes':torch.cuda.max_memory_allocated(device)}
    c.dump(run/'frozen_manifest.json',manifest)
    gt=np.asarray(native[2].labels,dtype=np.int64)
    trajectory=[]
    for item in predictions:
        with np.load(item['path']) as values:
            m=c.metric(gt,values['predictions'],values['logits'])
        trajectory.append({'epoch':item['epoch'],**m})
    c.dump(run/'trajectory.json',trajectory)
    c.dump(run/'results.json',{'target_at_final':trajectory[-1],**manifest,
                              'frozen_manifest_sha256':c.helper.file_hash(run/'frozen_manifest.json')})
    print('RESULT '+json.dumps(trajectory[-1]),flush=True)


if __name__=='__main__': main()
