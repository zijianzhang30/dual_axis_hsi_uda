"""Isolate balanced running buffers from joint-training moment gradients."""

from types import MethodType

import torch


def canonical_state(model):
    return model.state_dict()


def set_domain(model, domain):
    pass


def capture_moments(model, name):
    def hook(module, inputs):
        if not model.training:
            return
        values = inputs[0].detach()
        dims = (0,) + tuple(range(2, values.ndim))
        variance, mean = torch.var_mean(values, dim=dims, unbiased=False)
        n = values.numel() // values.shape[1]
        model._moment_records[name].append((mean, variance, n))
    return hook


def commit_buffers(model):
    """Commit detached pooled moments after backward avoids autograd version changes."""
    if model._bn_arm != "buffer_only":
        return
    with torch.no_grad():
        for name, old in model._old_buffers.items():
            values = model._moment_records[name]
            if len(values) != 2:
                raise RuntimeError(f"Expected both domain moments for {name}")
            (mean_s, var_s, ns), (mean_t, var_t, nt) = values
            if ns != nt:
                raise RuntimeError("Balanced buffer update requires equal activation counts")
            mean = (mean_s + mean_t) / 2
            variance = (var_s + var_t) / 2 + (mean_s - mean_t).square() / 4
            unbiased = variance * ((ns + nt) / (ns + nt - 1))
            module = model.get_submodule(name)
            module.running_mean.copy_(0.81 * old[0] + 0.19 * mean)
            module.running_var.copy_(0.81 * old[1] + 0.19 * unbiased)
            module.num_batches_tracked.copy_(old[2] + 1)
    model._old_buffers = {}
    model._moment_records = {}


def detached_tokens(model, source, target):
    if source.shape != target.shape:
        raise RuntimeError("Equal domain shapes required")
    size = len(source)
    conv3d, bn3d, act3d = model.conv3d_features
    # Native BN preserves exactly Full Joint's forward arithmetic. Only the
    # target half entering joint moments is detached; source-only loss makes
    # the extra direct target-output detachment irrelevant to the objective.
    source_cube, target_cube = act3d(bn3d(torch.cat((conv3d(source), conv3d(target).detach())))).split(size)
    def flatten(cube):
        b, c, bands, h, w = cube.shape
        return cube.reshape(b, c * bands, h, w)
    conv2d, bn2d, act2d = model.conv2d_features
    source_feature = conv2d(flatten(source_cube))
    target_feature = conv2d(flatten(target_cube))
    source_feature, target_feature = act2d(bn2d(torch.cat((source_feature, target_feature.detach())))).split(size)
    return model._forward_semantic_tokens(source_feature), model._forward_semantic_tokens(target_feature)


def tokenize_cached(self, images):
    if self.training and self._bn_arm == "detach_target":
        if not self._paired_cache:
            raise RuntimeError("Call paired forward to construct joint tokens")
        return self._paired_cache.pop(0)
    return self._original_tokenize_func(self, images)


def forward_routed(self, source, target, *args, **kwargs):
    if self.training:
        if self._bn_arm == "buffer_only":
            if self._old_buffers:
                raise RuntimeError("Call commit_buffers after every backward")
            self._old_buffers = {name: (module.running_mean.clone(), module.running_var.clone(),
                                       module.num_batches_tracked.clone())
                                 for name, module in self.named_modules()
                                 if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)}
            self._moment_records = {name: [] for name in self._old_buffers}
        else:
            self._paired_cache = list(detached_tokens(self, source, target))
    try:
        return self._original_forward_func(self, source, target, *args, **kwargs)
    finally:
        self._paired_cache = []


def install(model, arm):
    if arm not in {"buffer_only", "detach_target"}:
        raise ValueError(arm)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    model._bn_arm = arm
    model._old_buffers, model._moment_records, model._paired_cache = {}, {}, []
    model._original_forward_func = model.forward.__func__
    model._original_tokenize_func = model._tokenize.__func__
    if arm == "buffer_only":
        model._capture_handles = [module.register_forward_pre_hook(capture_moments(model, name))
                                  for name, module in model.named_modules()
                                  if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
    else:
        model.conv3d_features[1].momentum = 0.19
        model.conv2d_features[1].momentum = 0.19
    model.forward = MethodType(forward_routed, model)
    model._tokenize = MethodType(tokenize_cached, model)
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
    return len(before)
