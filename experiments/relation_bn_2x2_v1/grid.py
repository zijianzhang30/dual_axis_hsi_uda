"""Independent per-cell checkpoint-grid inference, without target scoring."""
import argparse
import json
import time
import numpy as np
import torch
import train
import audit

OUT=audit.OUT


def cell_folder(cell,seed):
    return OUT/f'cell2_{seed}' if cell=='2' else audit.folder(cell,seed)


def main():
    p=argparse.ArgumentParser();p.add_argument('--cell',choices=('1','2','3','4'),required=True)
    p.add_argument('--device',type=int,required=True)
    p.add_argument('--resume',action='store_true');args=p.parse_args()
    torch.set_num_threads(2);device=torch.device(f'cuda:{args.device}');torch.cuda.set_device(device)
    dest=OUT/f'grid_cell{args.cell}';dest.mkdir(exist_ok=args.resume)
    loader_args=argparse.Namespace(seed=2100,num_workers=4,dataset_dir='/home/zhangzj26/TGRS_MLUDA-2024/datasets/Houston/')
    train.base.seed_worker(2100)
    _,_,_,loader,_,_=train.base.make_loaders(loader_args)
    records=[];start=time.monotonic()
    for seed in audit.SEEDS:
        source=cell_folder(args.cell,seed)
        m=audit.read(source/('training_manifest.json' if args.cell=='2' else 'frozen_manifest.json'))
        train.base.seed_worker(seed)
        if args.cell=='1':
            model=train.base.get_model('BiDA','Houston13',13,argparse.Namespace(num_tokens=4,dim=64,depth=3))
        elif args.cell=='2':model,_=train.build(seed)
        else:model,_=train.base.build(seed,'A' if args.cell=='3' else 'C')
        model.to(device)
        for item in m['checkpoints']:
            assert audit.sha(item['path'])==item['sha256']
            path=dest/f'seed_{seed}_epoch_{item["epoch"]:03d}.npz'
            reusable=False
            if args.resume and path.exists():
                try:
                    with np.load(path) as previous:
                        reusable=(previous['logits'].shape==(len(loader.dataset),7) and
                                  previous['predictions'].shape==(len(loader.dataset),) and
                                  np.isfinite(previous['logits']).all())
                except Exception:
                    reusable=False
                if not reusable:
                    quarantine=path.with_suffix('.npz.partial_failed_write')
                    assert not quarantine.exists()
                    path.rename(quarantine) # Preserve failed writes, never delete them.
            if reusable:
                records.append({'cell':args.cell,'seed':seed,'epoch':item['epoch'],
                                'checkpoint_path':item['path'],'checkpoint_sha256':item['sha256'],
                                'prediction_path':str(path),'prediction_sha256':audit.sha(path)})
                print(json.dumps({'reused_frozen_prediction':str(path)}),flush=True)
                continue
            state=torch.load(item['path'],map_location='cpu',weights_only=True)
            assert state['epoch']==item['epoch']
            model.load_state_dict(state['model'],strict=True);del state
            logits,predictions=train.base.helper.predict(model,loader,device)
            np.savez_compressed(path,logits=logits,predictions=predictions)
            records.append({'cell':args.cell,'seed':seed,'epoch':item['epoch'],
                            'checkpoint_path':item['path'],'checkpoint_sha256':item['sha256'],
                            'prediction_path':str(path),'prediction_sha256':audit.sha(path)})
            print(json.dumps({'cell':args.cell,'seed':seed,'frozen_epoch':item['epoch']}),flush=True)
        del model
    (dest/'frozen_manifest.json').write_text(json.dumps({'records':records,
          'inference_seconds':time.monotonic()-start,'device':str(device),
          'resumed_after_disk_full':args.resume,'grid_code_sha256':audit.sha(__file__)},indent=2)+'\n')


if __name__=='__main__':main()
