"""Fixed repaired experiment import and metric extraction only."""
import importlib.util
from pathlib import Path
import numpy as np

ROOT=Path('/home/zhangzj26/dual_axis_hsi_uda')
OUT=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/pavia_l2_trajectory_v1')
spec=importlib.util.spec_from_file_location('trajectory_repaired_common',ROOT/'experiments/pavia_l2_fixed_v1/common.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
EPOCHS=tuple(range(10,201,10))
SEEDS=(2100,2101,2102)
ARMS=('B','C')


def metrics(labels,predictions,logits):
    result=c.helper.metrics(labels,predictions,logits,7)
    cm=np.asarray(result['confusion_matrix'])
    result.update(single_class_collapse=max(result['prediction_shares'])>=.95,
                  zero_recall_classes=[i+1 for i,v in enumerate(result['per_class_percent']) if v==0])
    for true in (3,4):
        n=int(cm[true-1].sum());count=int(cm[true-1,6])
        result['class%d_to_7'%true]={'count':count,'true_class_n':n,
                                   'percent_of_true_class':float(count/n*100)}
    return result
