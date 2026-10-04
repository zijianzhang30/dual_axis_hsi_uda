"""Read-only source/environment archival while the prescribed runs continue.

Never imports a model, iterates a loader, or changes a running dependency.
All writes are confined to a new archival directory and a project symlink.
"""
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path('/home/zhangzj26/dual_axis_hsi_uda')
BIDA=Path('/home/zhangzj26/IEEE_TCSVT_BiDA')
RUN=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/hyrank_A_no_extra_val_v1')
ARCHIVE=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/hyrank_repro_audit_v2/dependency_snapshot')
COMMIT='66a92b542ffa1df3c8b628af3d8bd001b799f1f6'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def command(args,cwd=None,env=None):
    return subprocess.check_output(args,cwd=cwd,env=env,text=True,stderr=subprocess.STDOUT)


def main():
    assert not ARCHIVE.exists() or ('--resume' in sys.argv and not (ARCHIVE/'manifest.json').exists()),'Refuse to overwrite a completed archive'
    assert shutil.disk_usage(RUN).free>2*1024**3
    ARCHIVE.mkdir(parents=True,exist_ok=True)
    records=[]
    def copy(path,relative):
        before=digest(path);target=ARCHIVE/'local_sources'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and digest(target)!=before:
            target=target.with_name(target.name+'.snapshot_'+before[:12])
        if not target.exists():shutil.copy2(path,target)
        assert before==digest(path)==digest(target),'Source changed while archiving'
        records.append({'source':str(path),'archive':str(target.relative_to(ARCHIVE)),
                        'sha256':before,'bytes':target.stat().st_size})
    roots=[BIDA]+[ROOT/'experiments'/name for name in
                  ('official_bench_v1','hyrank_A_no_extra_val_v1','bida_self_dropout',
                   'relation_tokenizer_v1','bn_stabilization_v1','hyrank_repro_audit_v2')]
    for root in roots:
        for path in sorted(root.rglob('*')):
            if path.is_file() and (path.suffix in ('.py','.md','.csv') or path.name=='requirements.txt'):
                if root==BIDA:relative=Path('external_BiDA')/path.relative_to(BIDA)
                else:relative=Path('project')/path.relative_to(ROOT)
                copy(path,relative)
    for seed in (2100,2101,2102):
        old=ROOT/f'results/official_bench_v1/hyrank_A_{seed}'
        for filename in ('config.json','frozen_manifest.json'):
            copy(old/filename,Path('metadata/old_A')/str(seed)/filename)
        copy(RUN/f'seed_{seed}/config.json',Path('metadata/new_A')/str(seed)/'config.json')
    for filename in ('preflight.json','protocol.json','launcher_status.json'):
        copy(RUN/filename,Path('metadata/new_A')/filename)
    # Original startup code snapshot, not just the currently visible source tree.
    for path in (RUN/'code_snapshot').iterdir():
        if path.is_file():copy(path,Path('metadata/new_A_startup_code')/path.name)
    # NAS owner mapping and old Git prevent safe checkout there. Pin the public
    # source in a fresh task-owned /tmp checkout, then archive the complete tree.
    # Do not alter global Git configuration or the first archived clone attempt.
    temporary=Path(tempfile.mkdtemp(prefix='hyrank_official_pin_',dir='/tmp'))/'repo'
    subprocess.run(['git','clone','--no-checkout','https://github.com/YuxiangZhang-BIT/IEEE_TCSVT_BiDA.git',str(temporary)],check=True)
    git=['git','-C',str(temporary)]
    subprocess.run(git+['checkout','--detach',COMMIT],check=True)
    assert command(git+['rev-parse','HEAD']).strip()==COMMIT
    upstream=ARCHIVE/'official_BiDA_pinned'
    if not upstream.exists():shutil.copytree(temporary,upstream)
    else:
        for line in command(git+['ls-files']).splitlines():
            assert digest(temporary/line)==digest(upstream/line)
    # Fixed source content, local compatibility patches, and added source files.
    patches=[];official_files=[]
    for line in command(git+['ls-files']).splitlines():
        official=upstream/line
        if not official.is_file():continue
        official_files.append({'path':line,'sha256':digest(official)})
        local=BIDA/line
        if local.is_file():
            result=subprocess.run(['git','diff','--no-index','--',str(official),str(local)],text=True,capture_output=True)
            assert result.returncode in (0,1)
            if result.stdout:patches.append(result.stdout)
    (ARCHIVE/'official_to_local.patch').write_text(''.join(patches))
    (ARCHIVE/'official_file_hashes.json').write_text(json.dumps(official_files,indent=2)+'\n')
    # Enumerate installed metadata without requiring pip or changing packages.
    packages=sorted([{'name':d.metadata.get('Name',str(d)), 'version':d.version}
                     for d in importlib.metadata.distributions()],key=lambda d:d['name'].lower())
    whitelist=('OMP_NUM_THREADS','MKL_NUM_THREADS','CUDA_VISIBLE_DEVICES','PYTHONHASHSEED','TMPDIR')
    running=json.loads((RUN/'launcher_status.json').read_text());processes=[]
    for job in running['jobs']:
        proc=Path('/proc')/str(job['pid'])
        item=dict(job)
        if proc.exists():
            item['executable']=os.readlink(proc/'exe')
            item['argv']=(proc/'cmdline').read_bytes().decode().strip('\0').split('\0')
            env={x.split('=',1)[0]:x.split('=',1)[1] for x in (proc/'environ').read_bytes().decode().split('\0') if '=' in x}
            item['selected_environment']={key:env[key] for key in whitelist if key in env}
        processes.append(item)
    environment={'captured_at_utc':datetime.now(timezone.utc).isoformat(),
                 'python_executable':sys.executable,'python':sys.version,'platform':platform.platform(),
                 'packages':packages,'running_processes':processes,
                 'gpu_driver':command(['nvidia-smi','--query-gpu=index,name,driver_version,memory.used,memory.total','--format=csv,noheader'])}
    (ARCHIVE/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
    (ARCHIVE/'project_status.txt').write_text(command(['git','status','--short'],ROOT))
    (ARCHIVE/'project_training_changes.patch').write_text(command(['git','diff','--','experiments/official_bench_v1/common.py','experiments/official_bench_v1/train.py'],ROOT))
    manifest={'official_commit':COMMIT,'project_commit':command(['git','rev-parse','HEAD'],ROOT).strip(),
              'files':records,'source_scope':'Source superset of external BiDA and known project helpers, including transitive old experiment code; not a claim every archived file was imported',
              'archive_only':True,'training_or_model_calls':0,'loader_iterations':0,
              'old_and_running_dependencies_modified':False,
              'pending_order':['Finish existing three runs','Separate single-class concentration from zero-recall counts without changing OA/AA',
                               'Epoch100/101 one-step and short multi-step equivalence from fully restored state and RNG',
                               'Only after PASS: independent reproducible directory and standalone metrics, then repeat equivalence'],
              'no_extra_full_training':True,'no_target_based_configuration_selection':True}
    (ARCHIVE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    project=ROOT/'results/hyrank_repro_audit_v2'
    if not project.exists() and not project.is_symlink():project.symlink_to(ARCHIVE.parent,target_is_directory=True)
    print(json.dumps({'archive':str(ARCHIVE),'source_file_count':len(records),'official_commit':COMMIT,
                      'running_dependencies_unchanged':True},indent=2))


if __name__=='__main__':main()
