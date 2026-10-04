"""Execute the independent audit with repaired loaders, not the failed old audit."""
import ast
from pathlib import Path
import sys
import common as c
import torch


def main():
    # Audit only: split-domain and concatenated convolution backward may select
    # different TF32 reductions. Keep the first failed audit artifact intact.
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    audit_out=c.OUT/'preflight_no_tf32'
    path=c.ROOT/'experiments/pavia_transfer_audit_v1/verify.py'
    tree=ast.parse(path.read_text())
    # Keep audit math/reference operators intact; only swap the experiment adapter.
    tree.body=[n for n in tree.body if not (isinstance(n,ast.Assign) and
               any(isinstance(t,ast.Name) and t.id=='c' for t in n.targets))]
    tree.body=[n for n in tree.body if not (isinstance(n,ast.If) and
               isinstance(n.test,ast.Compare) and isinstance(n.test.left,ast.Name) and n.test.left.id=='__name__')]
    for n in tree.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='OUT' for t in n.targets):
            n.value=ast.parse(repr(str(audit_out)),mode='eval').body
            n.value=ast.Call(func=ast.Name(id='Path',ctx=ast.Load()),args=[n.value],keywords=[])
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='OFFICIAL' for t in n.targets):
            n.value=ast.parse("Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/hyrank_repro_audit_v2/dependency_snapshot/official_BiDA_pinned')",mode='eval').body
    ast.fix_missing_locations(tree)
    ns={'__name__':'repaired_pavia_numerical_audit','__file__':str(Path(__file__).resolve()),'c':c}
    exec(compile(tree,str(path),'exec'),ns)
    ns['main']()
    # Strict historical identity of split/initial tensors, not old raw predictions.
    import json
    result=json.loads((audit_out/'verification.json').read_text())
    identities=[]
    for seed in (2100,2101,2102):
        c.pavia.seed_worker(seed)
        model,initial=c.build('pavia','A',seed)
        old=json.loads((c.ROOT/'results/pavia_joint_v1'/('original_%d'%seed)/'config.json').read_text())
        assert initial==old['initial_model_sha256']
        assert old['data_sha256']==result['data_protocol']['data_sha256']
        assert list(old['sample_counts'].values())==result['data_protocol']['sample_counts']
        identities.append({'seed':seed,'initial_model_sha256':initial,'historical_initialization_matches':True})
        del model
    result['three_seed_identity_checks']=identities
    result['audit_template_sha256']=c.helper.file_hash(path)
    result['TF32_disabled_for_numerical_audit_only']=True
    result['initial_failed_audit']='initial_preflight_fp32_failed.json'
    c.dump(c.OUT/'verification.json',result)


if __name__=='__main__':main()
