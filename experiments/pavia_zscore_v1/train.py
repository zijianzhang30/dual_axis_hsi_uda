"""Historical Pavia loop; only data normalization changes, with neutral auditing."""
import ast
from pathlib import Path
import common as c


def main():
    tree=ast.parse((c.ROOT/'experiments/pavia_joint_v1/train.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    for node in ast.walk(fn):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='add_argument':
            flag=node.args[0].value if node.args and isinstance(node.args[0],ast.Constant) else ''
            if flag=='--arm':
                next(k for k in node.keywords if k.arg=='choices').value=ast.Tuple(
                    elts=[ast.Constant('B'),ast.Constant('C')],ctx=ast.Load())
            if flag=='--seed':
                next(k for k in node.keywords if k.arg=='choices').value=ast.Tuple(elts=[ast.Constant(2100)],ctx=ast.Load())
        if isinstance(node,ast.If) and ast.dump(node.test)==ast.dump(ast.parse("args.arm == 'joint'",mode='eval').body):
            node.test=ast.Constant(True)
    for body in [fn.body]+[x.body for x in ast.walk(fn) if isinstance(x,(ast.With,ast.For))]:
        for i in range(len(body)-1,-1,-1):
            n=body[i]
            if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='config' for t in n.targets):
                body.insert(i+1,ast.parse('enrich(config,args,(train,val,target_train,target_eval))').body[0])
            if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='record' for t in n.targets):
                body.insert(i+1,ast.parse('check_stream(record)').body[0])
    scoring=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id=='labels' for t in n.targets))
    # Defer BOTH runs' target scoring until both predictions/manifests exist.
    fn.body=fn.body[:scoring]+ast.parse("print('PREDICTIONS_FROZEN epoch200; target scoring deferred',flush=True)").body
    state={'args':None};trace=c.Trace()
    old_history=[c.json.loads(x) for x in (c.reference('B')/'history.jsonl').read_text().splitlines()]
    def make_loaders(args):
        state['args']=args;args.dataset_dir=c.DATA
        return c.loaders(args.seed,args.num_workers,trace)
    def install(model,arm):
        c.base.joint.install(model,arm)
        if state['args'].arm=='C':c.base.relation(102)[0].install(model,'C',state['args'].seed)
    def enrich(config,args,values):
        checks=c.json.loads((c.OUT/'implementation_checks.json').read_text())
        assert checks['all_checks_passed']
        audit=c.audit(values)
        old=c.json.loads((c.reference(args.arm)/'config.json').read_text())
        for key in ('initial_model_sha256','normalization_sha256','model_code_sha256','dataset_code_sha256'):
            assert config[key]==old[key],key
        assert old['seed']==2100 and old['epochs']==200
        assert old['data_sha256']==audit['data_sha256'] and old['sample_counts']==audit['sample_counts']
        c.base.historical('pavia',2100,config['initial_model_sha256'],values)
        expected=checks['data_audit']['indices_sha256']
        assert audit['indices_sha256']==expected
        config.update(audit,norm='per-scene per-band sklearn.preprocessing.scale, before padding',
            bn='Full Joint',bn_momentum=.19,loss='source CE only',lr=.01,batch_size=128,
            depth=3,dim=64,tokens=4,dropout=.1,patch=13,source_split='95/5 stratified random_state23',
            study='normalization mechanism diagnostic, not MLUDA reproduction',
            target_training_labels='zero; original mask retained',
            reference_normband_path=str(c.reference(args.arm)),audit_shadow_channels_enter_model=False,
            criterion_for_scoring='both new epoch200 predictions frozen first',
            common_sha256=c.helper.file_hash(Path(__file__).with_name('common.py')),
            mluda_utils_sha256=c.helper.file_hash(Path('/home/zhangzj26/TGRS_MLUDA-2024-main/utils.py')),
            relation_sha256=c.helper.file_hash(c.ROOT/'experiments/relation_tokenizer_v1/relation.py'))
        if args.arm=='C':(args.out/'relation_band_adapter.py').write_text(c.base.relation(102)[1])
        c.dump(args.out/'compiled_training_AST.json',{'AST':ast.dump(fn),
            'base_sha256':c.helper.file_hash(c.ROOT/'experiments/pavia_joint_v1/train.py')})
    def check_stream(record):
        old=old_history[record['epoch']-1]
        assert record['steps']==trace.steps==old['steps']==290
        assert record['rng_sha256']==old['rng_sha256'],('RNG',record['epoch'])
        assert trace.shadow.hexdigest()==old['batch_sha256'],('sample/augmentation order',record['epoch'])
        record.update(normband_shadow_batch_sha256=trace.shadow.hexdigest(),
                      actual_dataset_indices_sha256=trace.indices.hexdigest(),
                      historical_normband_order_and_RNG_equal=True)
        trace.reset()
    ns=dict(c.pavia.__dict__)
    ns.update(__file__=str(Path(__file__).resolve()),make_loaders=make_loaders,
              install=install,enrich=enrich,check_stream=check_stream)
    ast.fix_missing_locations(fn)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),__file__,'exec'),ns)
    ns['main']()


if __name__=='__main__':main()
