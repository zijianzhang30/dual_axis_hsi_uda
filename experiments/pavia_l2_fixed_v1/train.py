"""Compile unchanged archived training main, swapping repaired data and arms."""
import ast
from pathlib import Path
import common as c


def main():
    assert __import__('json').loads((c.OUT/'verification.json').read_text())['passed']
    tree=ast.parse((c.ROOT/'experiments/pavia_joint_v1/train.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    for node in ast.walk(fn):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='add_argument':
            if node.args and isinstance(node.args[0],ast.Constant) and node.args[0].value=='--arm':
                next(k for k in node.keywords if k.arg=='choices').value=ast.Tuple(
                    elts=[ast.Constant('A'),ast.Constant('B'),ast.Constant('C')],ctx=ast.Load())
        if isinstance(node,ast.If) and ast.dump(node.test)==ast.dump(ast.parse("args.arm == 'joint'",mode='eval').body):
            node.test=ast.parse("args.arm != 'A'",mode='eval').body
    for body in [fn.body]+[n.body for n in ast.walk(fn) if isinstance(n,(ast.With,ast.For))]:
        for i in range(len(body)-1,-1,-1):
            node=body[i]
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='config' for t in node.targets):
                body.insert(i+1,ast.parse('enrich(config,args,(train,val,target_train,target_eval))').body[0])
    state={}
    def make_loaders(args):
        state['args']=args
        args.dataset_dir=c.DATA/'Pavia'
        return c.loaders('pavia',args.seed,args.num_workers)
    def install(model,arm):
        c.joint.install(model,arm)
        if state['args'].arm=='C':c.relation(102)[0].install(model,'C',state['args'].seed)
    def enrich(config,args,values):
        config.update(c.inputs_audit('pavia',values),study='repaired Pavia L2 transfer validation',
                      loss='source CE only',lr=.01,batch_size=128,depth=3,dropout=.1,
                      bn='Original Mixed' if args.arm=='A' else 'Full Joint',
                      bn_momentum=.1 if args.arm=='A' else .19,
                      eps=1e-6,common_sha256=c.helper.file_hash(Path(c.__file__)),
                      relation_sha256=c.helper.file_hash(c.ROOT/'experiments/relation_tokenizer_v1/relation.py'),
                      historical_base_sha256=c.helper.file_hash(c.ROOT/'experiments/pavia_joint_v1/train.py'))
        config['sample_counts']=dict(zip(('source_train','source_val','target_train','target_eval'),[len(v.dataset) for v in values]))
        assert list(config['sample_counts'].values())==[37236,1960,39348,39348]
        c.dump(args.out/'compiled_training_AST.json',{'AST':ast.dump(fn)})
        if args.arm=='C':(args.out/'relation_band_adapter.py').write_text(c.relation(102)[1])
    ns=dict(c.historical.__dict__)
    ns.update(__file__=str(Path(__file__).resolve()),make_loaders=make_loaders,install=install,enrich=enrich)
    ast.fix_missing_locations(fn)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),__file__,'exec'),ns)
    ns['main']()


if __name__=='__main__':main()
