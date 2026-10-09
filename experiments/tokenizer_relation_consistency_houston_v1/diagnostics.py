import torch,math
from relations import spectral,kl
@torch.no_grad()
def describe(scorer,patch,seed=77700):
 g=torch.Generator(device=patch.device).manual_seed(seed);w=scorer(patch);a=scorer(spectral(patch,g));tiny=torch.finfo(w.dtype).tiny
 entropy=-(w*w.clamp_min(tiny).log()).sum(-1);unit=torch.nn.functional.normalize(w,dim=-1);cos=unit@unit.transpose(1,2);mask=~torch.eye(4,dtype=torch.bool,device=w.device)
 p=w[:,:,None,:];q=w[:,None,:,:];mix=(p+q)/2;js=.5*(p*(p.clamp_min(tiny).log()-mix.clamp_min(tiny).log())+q*(q.clamp_min(tiny).log()-mix.clamp_min(tiny).log())).sum(-1)
 va=(w+a)/2;augjs=.5*(w*(w.clamp_min(tiny).log()-va.clamp_min(tiny).log())+a*(a.clamp_min(tiny).log()-va.clamp_min(tiny).log())).sum(-1)
 peak=w.max(-1).values;top=w.topk(10,dim=-1).values.sum(-1)
 uniform=((entropy/math.log(169)>.999)&(peak*169<1.05)).all(-1)
 identical=((cos[:,mask].mean(-1)>.999)&(js[:,mask].mean(-1)<1e-4))
 return {'entropy_nats':float(entropy.mean()),'entropy_normalized':float(entropy.mean()/math.log(169)),'token_map_cosine':float(cos[:,mask].mean()),'token_map_JS':float(js[:,mask].mean()),'peak_maxW':float(peak.mean()),'peak_vs_uniform':float((peak*169).mean()),'top10_mass':float(top.mean()),'RI_KL':float(kl(w,a)),'RI_JS':float(augjs.mean()),'uniform_fraction':float(uniform.float().mean()),'identical_tokens_fraction':float(identical.float().mean()),'relation_collapse':bool(uniform.float().mean()>=.95 or identical.float().mean()>=.95)}
