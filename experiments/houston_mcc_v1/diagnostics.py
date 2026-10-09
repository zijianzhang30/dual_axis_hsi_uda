import torch
from mcc_loss import entropy

def diagnostic(logits):
 z=logits.float();p=(z/2.).softmax(1);h=entropy(p);w=1+torch.exp(-h.detach());w=len(z)*w/w.sum();raw=(p*w[:,None]).T@p;denom=raw.sum(1)
 assert torch.isfinite(raw).all() and (denom>0).all(), 'MCC normalization invalid;STOP,do not modify'
 matrix=raw/denom
 assert torch.isfinite(matrix).all() and torch.allclose(matrix.sum(0),torch.ones(7,device=z.device),atol=1e-5), 'Official normalization abnormal;STOP'
 native=z.softmax(1);natural=-(native*native.clamp_min(1e-12).log()).sum(1)
 return {'target_entropy':float(natural.mean()),'MCC_temperature_entropy':float(h.mean()),'uncertainty_weight_sum':float(w.sum()),'raw_confusion':raw.detach().cpu().tolist(),'normalized_confusion':matrix.detach().cpu().tolist()}
