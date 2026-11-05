"""Fail-closed nested membership contract, independent of split generation."""
import copy
import json

def validate(evidence, groups, ids):
    n=len(ids)
    if len(groups)!=n or len(set(ids))!=n:
        raise ValueError('Nonunique IDs or incomplete groups')
    def indices(values):
        if not isinstance(values,list) or not values or any(type(i)!=int or i<0 or i>=n for i in values) or len(set(values))!=len(values):
            raise ValueError('Invalid index vector')
        return set(values)
    def disjoint(a,b):
        if a&b or {groups[i] for i in a}&{groups[i] for i in b}:
            raise ValueError('Connected-group leakage')
    tr=indices(evidence['outer_train']);va=indices(evidence['outer_validation'])
    disjoint(tr,va)
    if len(evidence['inner'])!=3 or {x['inner'] for x in evidence['inner']}!={0,1,2}:
        raise ValueError('Expected exactly three distinct inner folds')
    covered=set()
    for x in evidence['inner']:
        it=indices(x['train']);iv=indices(x['validation'])
        if it|iv!=tr or covered&iv:
            raise ValueError('Inner coverage differs from outer training')
        if len(it)!=x['train_n'] or len(iv)!=x['validation_n']:
            raise ValueError('Incorrect recorded counts')
        disjoint(it,iv);disjoint(it|iv,va);covered|=iv
    if covered!=tr:
        raise ValueError('Missing OOF predictions')
    return {'status':'PASS','outer_train_n':len(tr),'outer_validation_n':len(va),'inner_oof_rows':len(covered),'outer_holdout_excluded_from_all_inner_memberships':True}

def tests():
    e={'outer_train':list(range(6)),'outer_validation':[6,7],'inner':[]}
    for j in range(3):
        v=[2*j,2*j+1];t=[i for i in range(6) if i not in v]
        e['inner'].append(dict(inner=j,train=t,validation=v,train_n=4,validation_n=2))
    g=list(range(8));ids=list(map(str,g));validate(e,g,ids)
    cases=[]
    bad=copy.deepcopy(e);bad['inner'][0]['train'][0]=6;cases.append((bad,g,ids))
    bad=copy.deepcopy(e);bad['inner'][0]['train'][0]=0;cases.append((bad,g,ids))
    bad=copy.deepcopy(e);bad['inner'][1]=copy.deepcopy(bad['inner'][0]);cases.append((bad,g,ids))
    bad=copy.deepcopy(e);bad['inner'][0]['train'][0]=-1;cases.append((bad,g,ids))
    bad=copy.deepcopy(e);bad['inner'][0]['train_n']=9;cases.append((bad,g,ids))
    gg=g[:];gg[6]=0;cases.append((e,gg,ids))
    gg=g[:];gg[2]=0;cases.append((e,gg,ids))
    ii=ids[:];ii[7]=ii[0];cases.append((e,g,ii))
    bad=copy.deepcopy(e);bad['inner'][0]['validation']=[0,0];cases.append((bad,g,ids))
    for args in cases:
        try: validate(*args)
        except ValueError: pass
        else: raise AssertionError('Contamination was accepted')
    return dict(status='PASS',valid_partition_cases=1,rejected_contamination_cases=len(cases))

if __name__=='__main__':
    print(json.dumps(tests()))
