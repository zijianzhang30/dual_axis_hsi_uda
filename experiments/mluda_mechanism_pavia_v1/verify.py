"""Meaningful GPU preflight: exact native path, full loss, gradients, BN and RNG."""
import copy
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from common import OLD, OUT, dump, rng_hash, state_hash
from mechanisms import install, bn_counters


def optimizer(model):
    return torch.optim.SGD([
        {'params': model.feature_layers.parameters()},
        {'params': model.fc1.parameters(), 'lr': .001},
        {'params': model.fc2.parameters(), 'lr': .001},
        {'params': model.head1.parameters(), 'lr': .001},
        {'params': model.head2.parameters(), 'lr': .001}],
        lr=.001, momentum=.9, weight_decay=.0005)


def main():
    # Verification-only controls isolate code equivalence from CUDA reduction
    # nondeterminism. Production retains the native backend configuration.
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    sys.path.insert(0, str(OLD / 'native_snapshot'))
    import net2
    import utils
    import mmd
    from contrastive_loss import SupConLoss
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    utils.set_seed(1622)
    native = net2.DSANSS(102, 11, 7).cuda()
    # Clone without constructor RNG consumption.
    off = copy.deepcopy(native)
    install(off)
    s, t = torch.randn(32,102,11,11,device='cuda'), torch.randn(32,102,11,11,device='cuda')
    labels = torch.arange(32,device='cuda') % 7
    rng_cpu = torch.get_rng_state()
    rng_cuda = torch.cuda.get_rng_state_all()
    ce, con = nn.CrossEntropyLoss(), SupConLoss(temperature=.1).cuda()
    def step(model):
        torch.set_rng_state(rng_cpu)
        torch.cuda.set_rng_state_all(rng_cuda)
        opt = optimizer(model)
        outputs = [model(s,t), model(s*.93+.04,t*1.02), model(s.flip(1),t.flip(0))]
        o = outputs[0]
        pseudo = o[8].softmax(1).detach().argmax(1)
        lmmd = mmd.lmmd(o[0], o[5], labels.cpu(), o[8].softmax(1), BATCH_SIZE=32, CLASS_NUM=7)
        loss = ce(o[3], labels) + .3*.5*lmmd
        loss = loss + con(torch.stack([outputs[1][1],outputs[2][1]],1),labels.cpu())
        loss = loss + con(torch.stack([outputs[1][7],outputs[2][7]],1),pseudo)
        opt.zero_grad(); loss.backward()
        grads = {k:p.grad.clone() for k,p in model.named_parameters() if p.grad is not None}
        opt.step()
        return outputs, grads, loss, opt
    a,ga,la,_ = step(native)
    b,gb,lb,_ = step(off)
    assert all(torch.equal(x,y) for u,v in zip(a,b) for x,y in zip(u,v))
    gradient_errors = {k:float((ga[k]-gb[k]).abs().max()) for k in ga}
    assert ga.keys() == gb.keys() and all(torch.allclose(ga[k],gb[k],atol=2e-5,rtol=2e-5) for k in ga), gradient_errors
    assert torch.equal(la,lb)
    assert all(torch.allclose(v,off.state_dict()[k],atol=2e-6,rtol=2e-6) for k,v in native.state_dict().items())
    counters = bn_counters(native)
    assert all(counters['bn%d'%i] == 6 for i in range(1,8))
    assert counters['bn9'] == counters['bn10'] == 0

    # Initialization/insertion leaves every original tensor and global RNG unchanged.
    cpu = net2.DSANSS(102,11,7)
    original = copy.deepcopy(cpu.state_dict())
    rng = rng_hash()
    install(cpu, False, True, 1622)
    assert rng == rng_hash()
    assert all(torch.equal(v,cpu.state_dict()[k]) for k,v in original.items())
    pool = cpu.feature_layers.relation_pool
    patch = torch.randn(3,102,11,11)
    constant = torch.ones_like(patch)
    assert torch.equal(pool.weights(constant), torch.full((3,4,121),1/121))
    relation = (patch-patch[...,5:6,5:6])/(patch.std((-2,-1),correction=0,keepdim=True)+1e-6)
    assert torch.equal(relation[...,5,5], torch.zeros(3,102))
    assert pool.weights(patch).shape == (3,4,121)
    assert torch.allclose(pool.weights(patch).sum(-1),torch.ones(3,4))
    # Odd spatial flips preserve the actual geometric center and weight grid.
    assert torch.allclose(pool.weights(patch.flip(-1)).reshape(3,4,11,11),pool.weights(patch).reshape(3,4,11,11).flip(-1),atol=1e-6)
    pool.uniform = True
    uniform = copy.deepcopy(native)
    install(uniform,False,True,1622)
    uniform.feature_layers.relation_pool.uniform = True
    uniform.cuda().eval(); native.eval()
    with torch.no_grad():
        na, un = native(s,t), uniform(s,t)
    errors = [float((x-y).abs().max()) for x,y in zip(na,un)]
    assert all(torch.allclose(x,y,atol=2e-5,rtol=2e-5) for x,y in zip(na,un)),errors

    joint = net2.DSANSS(102,11,7)
    original = copy.deepcopy(joint.state_dict())
    install(joint,True,True,1622)
    assert all(torch.equal(v,joint.state_dict()[k]) for k,v in original.items())
    joint.cuda().train()
    before = {k:p.detach().clone() for k,p in joint.feature_layers.relation_pool.named_parameters()}
    _,grads,loss,opt = step(joint)
    jc = bn_counters(joint)
    assert all(jc['bn%d'%i] == 3 for i in range(1,8)) and jc['bn9']==jc['bn10']==0
    ids = {id(p) for p in opt.param_groups[0]['params']}
    assert all(id(p) in ids for p in joint.feature_layers.relation_pool.parameters())
    assert all(torch.isfinite(p).all() and not torch.equal(p,before[k]) for k,p in joint.feature_layers.relation_pool.named_parameters())
    # Direct source feature depends on target through joint CNN moments.
    joint.zero_grad(); ti = t.clone().requires_grad_(True)
    joint.feature_layers(s,ti)[0].square().mean().backward()
    assert ti.grad is not None and torch.isfinite(ti.grad).all() and ti.grad.abs().sum()>0
    # Three native view calls remain separate; eval does not update any BN buffer.
    joint.eval()
    buffers = {k:v.clone() for k,v in joint.named_buffers()}
    with torch.no_grad():
        joint(s,t)
    assert all(torch.equal(v,dict(joint.named_buffers())[k]) for k,v in buffers.items())
    report = {'passed':True,'native_off_outputs_full_loss':'bitwise identical',
        'native_off_gradients_update':'within 2e-5 gradient / 2e-6 state tolerance (native CUDA avgpool backward lacks deterministic implementation)',
        'native_off_max_gradient_error':max(gradient_errors.values()),
        'native_three_views_bn_counters':counters,'joint_three_views_bn_counters':jc,
        'inactive_bn9_bn10':'unchanged', 'joint_momentum':.19,
        'momentum_scope':'old-statistics decay match only; NOT moment/normalization equivalence',
        'uniform_avgpool_full_output_max_abs_errors':errors,
        'native_initialization_and_rng':'unchanged','relation_optimizer_group':0,
        'pool_parameter_updates':'all updated','joint_target_gradient':'nonzero finite',
        'eval_running_buffers':'unchanged','center_grid_softmax':'passed',
        'verification_only_deterministic_algorithms':True,
        'cuda_device':torch.cuda.get_device_name(0), 'torch':torch.__version__}
    dump(OUT / 'verification.json',report)
    print(report,flush=True)


if __name__ == '__main__':
    main()
