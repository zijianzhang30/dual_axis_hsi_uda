"""Label-free cache authentication, copying and exact nearest-neighbor search."""
import shutil
import time
import numpy as np
import torch
from common import HERE, OUT, CACHE, BASE, DATA, SEEDS, STAGES, dump, sha, ahash

def select_neighbors(scores, maxk=15):
    # topk+1 detects cutoff ties; sorting selected candidates fixes internal ties.
    values, indices = torch.topk(scores, maxk+1, dim=1, largest=True, sorted=True)
    v, ids = values.cpu().numpy(), indices.cpu().numpy()
    tied = v[:,maxk-1] == v[:,maxk]
    result_ids = np.empty((len(v),maxk),dtype=np.int64)
    result_values = np.empty((len(v),maxk),dtype=np.float64)
    for i in range(len(v)):
        if tied[i]:
            row = scores[i].cpu().numpy()
            candidate = np.flatnonzero(row >= v[i,maxk-1])
            order = np.lexsort((candidate, -row[candidate]))[:maxk]
            result_ids[i] = candidate[order]
            result_values[i] = row[result_ids[i]]
        else:
            order = np.lexsort((ids[i], -v[i]))[:maxk]
            result_ids[i], result_values[i] = ids[i,order],v[i,order]
    return result_ids, result_values, int(tied.sum())

@torch.inference_mode()
def exact_knn(query, reference, exclude_self=False):
    q = torch.as_tensor(np.array(query), device='cuda', dtype=torch.float64)
    r = torch.as_tensor(np.array(reference), device='cuda', dtype=torch.float64)
    q = torch.nn.functional.normalize(q,dim=1,eps=1e-12)
    r = torch.nn.functional.normalize(r,dim=1,eps=1e-12)
    ids, values, ties = [], [], 0
    for start in range(0,len(q),512):
        score = q[start:start+512] @ r.T
        if exclude_self:
            score[torch.arange(len(score),device='cuda'),torch.arange(start,start+len(score),device='cuda')] = -torch.inf
        ix, vs, nt = select_neighbors(score)
        if exclude_self:
            assert not (ix == np.arange(start,start+len(score))[:,None]).any()
        ids.append(ix);values.append(vs);ties += nt
        if start % (512*25) == 0:
            print('  exact neighbors',start,'/',len(q),'target LOO' if exclude_self else 'source',flush=True)
    return np.concatenate(ids),np.concatenate(values),ties

def check_knn():
    # Independent NumPy full-sort fixture, including identical vectors/cutoff ties.
    rng=np.random.default_rng(2100)
    x=rng.normal(size=(28,9));x[:17]=x[0]
    u=x/np.linalg.norm(x,axis=1,keepdims=True)
    score=torch.tensor(u,dtype=torch.float64,device='cuda')@torch.tensor(u.T,dtype=torch.float64,device='cuda')
    score.fill_diagonal_(-torch.inf)
    ids,_,_=select_neighbors(score)
    cpu=score.cpu().numpy()
    expected=np.stack([np.lexsort((np.arange(len(x)),-row))[:15] for row in cpu])
    assert np.array_equal(ids,expected)
    assert not (ids==np.arange(len(x))[:,None]).any()

def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    start=time.monotonic();OUT.mkdir(exist_ok=False)
    gate=__import__('json').loads((CACHE/'ALL_FEATURES_PREDICTIONS_FROZEN.json').read_text())
    original=__import__('json').loads((CACHE/'extraction_provenance.json').read_text())
    assert gate['all_seeds_completed'] and not gate['target_GT_opened_by_extractor']
    assert sha(CACHE/'extraction_provenance.json')==gate['files_sha256']['extraction_provenance.json']
    mask=np.load(DATA/'source_probe_train_mask.npy');assert mask.sum()==2393
    np.save(OUT/'source_train_mask.npy',mask)
    provenance={'cache_manifest_sha256':sha(CACHE/'ALL_FEATURES_PREDICTIONS_FROZEN.json'),
                'cached_extraction_provenance_sha256':sha(CACHE/'extraction_provenance.json'),
                'protocol_sha256':sha(HERE/'PROTOCOL.txt'),
                'code_sha256':{str(p):sha(p) for p in HERE.glob('*') if p.is_file()},
                'historical_metadata':original['historical_input_metadata'],'seeds':{}}
    for seed in SEEDS:
        folder=OUT/f'seed_{seed}';folder.mkdir()
        rec=original['seeds'][str(seed)]
        assert sha(rec['checkpoint'])==rec['checkpoint_sha256'] and rec['state_unchanged']
        assert rec['audit']['target']['native_predictions_exact']
        config=__import__('json').loads((BASE/'houston'/f'C_{seed}/config.json').read_text())
        assert config['arm']=='C' and config['epochs']==200 and config['BN']=='unchanged Full Joint/momentum.19'
        for domain in ['source','target']:
            for key in ['T','Z','logits','predictions','coords']:
                rel=f'seed_{seed}/{domain}_{key}.npy';src=CACHE/rel
                assert sha(src)==gate['files_sha256'][rel]
                shutil.copyfile(src,OUT/rel)
                assert sha(OUT/rel)==gate['files_sha256'][rel]
        coords=np.load(folder/'source_coords.npy')
        assert ahash(coords[mask])==config['indices_sha256']['source']
        assert ahash(coords[~mask])==config['indices_sha256']['val']
        provenance['seeds'][str(seed)]={'checkpoint':rec['checkpoint'],'checkpoint_sha256':rec['checkpoint_sha256'],
                                      'native_cached_replay':rec['audit'],'source_train_split_exact':True}
    dump(OUT/'provenance.json',provenance)
    files={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*.npy')}
    files['provenance.json']=sha(OUT/'provenance.json')
    dump(OUT/'ALL_FEATURES_PREDICTIONS_FROZEN.json',{'files_sha256':files,'all_seeds_completed':True,
         'target_GT_opened_by_this_diagnostic':False,'mode':'authenticated frozen eval cache reuse'})
    print('All3 seed feature/logit/prediction copies frozen; no labels read',flush=True)
    check_knn()
    for seed in SEEDS:
        folder=OUT/f'seed_{seed}';audits={}
        for stage in STAGES:
            source=np.load(folder/f'source_{stage}.npy')[mask]
            target=np.load(folder/f'target_{stage}.npy')
            for a in [source,target]:assert np.isfinite(a).all() and abs(np.linalg.norm(a,axis=1)-1).max()<2e-6
            si,sv,st=exact_knn(target,source)
            ti,tv,tt=exact_knn(target,target,True)
            np.savez_compressed(folder/f'{stage}_neighbors.npz',source_indices=np.flatnonzero(mask)[si],
                                source_cosines=sv,target_indices=ti,target_cosines=tv)
            audits[stage]={'source_cutoff_tie_rows':st,'target_cutoff_tie_rows':tt,'exact_target_self_exclusion':True}
            print(seed,stage,'neighbors frozen',audits[stage],flush=True)
        dump(folder/'neighbor_audit.json',audits)
    artifacts={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()}
    dump(OUT/'ALL_NEIGHBORS_FROZEN.json',{'files_sha256':artifacts,'all_seeds_completed':True,
         'target_GT_opened_by_this_diagnostic':False,'exact_knn_fixture_verified':True,'seconds':time.monotonic()-start})
    print('All label-free neighbors frozen; target GT still unopened',flush=True)

if __name__=='__main__':main()
