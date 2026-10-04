"""Compile the existing Pavia main, adapting only metadata/bands/classes/tokenizer."""
import argparse
import ast
import sys
from pathlib import Path
import common as c


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--task', choices=('pavia','shanghai'), required=True)
    task_args, rest = p.parse_known_args()
    task = task_args.task
    sys.argv = [sys.argv[0]] + rest
    tree = ast.parse((c.ROOT/'experiments/pavia_joint_v1/train.py').read_text())
    fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name=='main')
    bands, classes = c.dimensions(task)
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr=='add_argument':
            if node.args and isinstance(node.args[0],ast.Constant) and node.args[0].value=='--arm':
                next(k for k in node.keywords if k.arg=='choices').value=ast.Tuple(
                    elts=[ast.Constant('B'),ast.Constant('C')],ctx=ast.Load())
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id=='BiDAnet':
            for kw in node.keywords:
                if kw.arg=='n_bands': kw.value=ast.Constant(bands)
                if kw.arg=='num_classes': kw.value=ast.Constant(classes)
        if isinstance(node, ast.If) and ast.dump(node.test)==ast.dump(ast.parse("args.arm == 'joint'",mode='eval').body):
            node.test=ast.Constant(True)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr=='metrics':
            node.args[-1]=ast.Constant(classes)
        if classes < 6 and isinstance(node, ast.Assign):
            if any(isinstance(t,ast.Subscript) and isinstance(t.slice,ast.Index)
                   and isinstance(t.slice.value,ast.Constant) and t.slice.value.value=='class6_prediction_ratio'
                   for t in node.targets):
                node.value=ast.Constant(None)
        if isinstance(node, ast.Constant) and node.value=='PaviaU -> PaviaC (existing 7-class mapping)':
            node.value = 'PaviaU -> PaviaC' if task=='pavia' else 'Shanghai -> Hangzhou'
    # Assertions/metadata never iterate a loader or change model state/RNG.
    for body in [fn.body] + [x.body for x in ast.walk(fn) if isinstance(x,(ast.With,ast.For))]:
        for i in range(len(body)-1,-1,-1):
            node=body[i]
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='config' for t in node.targets):
                body.insert(i+1,ast.parse('enrich(config, args, (train,val,target_train,target_eval))').body[0])
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='record' for t in node.targets):
                body.insert(i+1,ast.parse('check_stream(record)').body[0])
    state = {'reference':None, 'args':None}
    def install(model, arm):
        c.joint.install(model,arm)
        if state['args'].arm=='C': c.relation(bands)[0].install(model,'C',state['args'].seed)
    def make_loaders(args):
        state['args']=args
        # Use explicit canonical NAS data, not the historical parser default.
        args.dataset_dir=c.DATA/('Pavia' if task=='pavia' else 'Shanghai-Hangzhou')
        return c.loaders(task,args.seed,args.num_workers)
    def enrich(config,args,values):
        state['reference']=c.historical(task,args.seed,config['initial_model_sha256'],values)
        config.update(c.inputs_audit(task,values),task=task,study='MLUDA datasets, BiDA-style extension',
                      bn='Full Joint',bn_momentum=.19,loss='source CE only',lr=.01,
                      batch_size=128,depth=3,dropout=.1,eps=1e-6,
                      historical_B_reused=task=='pavia')
        config['sample_counts']=dict(zip(('source_train','source_val','target_train','target_eval'),
                                        [len(x.dataset) for x in values]))
        config['relation_sha256']=c.helper.file_hash(c.ROOT/'experiments/relation_tokenizer_v1/relation.py')
        if args.arm=='C': (args.out/'relation_band_adapter.py').write_text(c.relation(bands)[1])
        c.dump(args.out/'compiled_training_AST.json',{'AST':ast.dump(fn),
              'base_sha256':c.helper.file_hash(c.ROOT/'experiments/pavia_joint_v1/train.py')})
    def check_stream(record):
        reference=state['reference']
        if reference is not None:
            old=reference[record['epoch']-1]
            for key in ('steps','batch_sha256','rng_sha256'): assert record[key]==old[key], (record['epoch'],key)
    ns=dict(c.pavia.__dict__)
    ns.update(__file__=str(Path(__file__).resolve()),install=install,make_loaders=make_loaders,
              enrich=enrich,check_stream=check_stream)
    ast.fix_missing_locations(fn)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),__file__,'exec'),ns)
    ns['main']()


if __name__=='__main__': main()
