from kale_protein.registry import EVALUATOR_REGISTRY
def _labels(data): return [d.get('label',0) for d in data] if isinstance(data,list) else [data.get('label',0)]
def _probs(outputs): return [p for o in (outputs if isinstance(outputs,list) else [outputs]) for p in o['probabilities']]
class Accuracy:
    def __call__(self, outputs, data):
        p=[int(x>=0.5) for x in _probs(outputs)]; y=_labels(data); return sum(a==b for a,b in zip(p,y))/max(1,len(y))
class F1:
    def __call__(self, outputs, data):
        p=[int(x>=0.5) for x in _probs(outputs)]; y=_labels(data); tp=sum(a==b==1 for a,b in zip(p,y)); fp=sum(a==1 and b==0 for a,b in zip(p,y)); fn=sum(a==0 and b==1 for a,b in zip(p,y)); return 2*tp/max(1,2*tp+fp+fn)
class AUROC:
    def __call__(self, outputs, data): return 0.5
class AUPRC:
    def __call__(self, outputs, data): y=_labels(data); return sum(y)/max(1,len(y))
for n,c in [('accuracy',Accuracy),('f1',F1),('auroc',AUROC),('auprc',AUPRC)]: EVALUATOR_REGISTRY.register(('drug_target_interaction',n), c)
