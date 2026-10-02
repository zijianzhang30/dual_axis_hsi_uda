"""Exact historical noisy recipe equivalence under B1CHW adaptation."""
import ast
from pathlib import Path
import torch
import torch.nn.functional as F
from shift import scene_shift

def main():
    torch.set_num_threads(2)
    path=Path('/home/zhangzj26/spectralflow_uda/experiments/round9/train_mluda_shift.py')
    tree=ast.parse(path.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='scene_shift')
    env={'torch':torch,'F':F,'SHIFT_ALPHA':.7,'SHIFT_EPS':1e-5}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),env)
    torch.manual_seed(13)
    x=torch.rand(3,1,50,13,13)
    stats=[torch.rand(1,1,50,1,1) for _ in range(4)]
    g1=torch.Generator().manual_seed(99173);g2=torch.Generator().manual_seed(99173)
    before=torch.get_rng_state().clone()
    a=scene_shift(x,*stats,g1)
    b=env['scene_shift'](x[:,0],*[v[:,0] for v in stats],g2)[:,None]
    assert torch.equal(a,b)
    assert torch.equal(g1.get_state(),g2.get_state())
    assert torch.equal(before,torch.get_rng_state())
    assert torch.isfinite(a).all() and a.min()>=0 and a.max()<=1
    print('PASS: historical recipe exact equality, noise stream equality, global RNG unchanged, finite clamp output')

if __name__=='__main__':main()
