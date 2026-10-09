"""Same fixed metric definitions, G only; original A-F diagnostics untouched."""
import argparse
import json
import numpy as np
from common import BASE,OUT,SEEDS,module,dump

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['houston','pavia'],required=True);args=p.parse_args();ds=args.dataset
 original=module('fixed_weight_metric_definitions',BASE/'weight_analysis.py');report={}
 assert (OUT/ds/'all_G_predictions_frozen.json').exists()
 for seed in SEEDS:
  folder=OUT/ds/f'G_{seed}';gw=np.load(folder/'weights_200.npy',mmap_mode='r');gt=np.load(folder/'tokens_200.npy',mmap_mode='r')
  metrics=original.diversity(gw,gt);comparisons={}
  for a in 'ABCF':
   aw=np.load(BASE/ds/f'{a}_{seed}/weights_200.npy',mmap_mode='r');comparisons[f'{a}-G']=original.pair(aw,gw)
  report[str(seed)]={'G':metrics,'comparisons':comparisons};dump(OUT/ds/f'G_weights_{seed}.json',report[str(seed)])
  print('G weights analyzed',ds,seed,flush=True)
 aggregate={k:{'mean':float(np.mean([report[str(s)]['G'][k] for s in SEEDS])),'sample_std':float(np.std([report[str(s)]['G'][k] for s in SEEDS],ddof=1))} for k in report[str(SEEDS[0])]['G']}
 dump(OUT/ds/'weight_analysis.json',{'G_runs':report,'G_aggregate':aggregate,'original_A_F_weights':str(BASE/ds/'weight_analysis.json'),'metric_definitions':'unchanged; entropy nats/normalized, pair cosine/JS, token cosine/effective rank, 24-permutation matching','same_index_and_permutation_matched':True,'cross_checkpoint_coadaptation_limit':True})
if __name__=='__main__':main()
