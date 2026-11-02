"""Changed host inputs through the installed batch API, using its library default."""
import json
import numpy as np

from kerneldarwin.transaction_plan import TransactionPlan, source_identity


def main():
    documents=np.random.default_rng(49051).normal(size=(37,97)).astype(np.float32)
    queries=[documents[[0]].copy(),documents[[1,2]].copy(),documents[[35]].copy()]
    plan=TransactionPlan()
    first=plan.search(documents,queries)
    changed=plan.search(documents[::-1],queries)
    for output,reverse in ((first,False),(changed,True)):
        for (ids,_),targets in zip(output,((0,),(1,2),(35,))):
            for row,target in zip(ids,targets):
                if row[0]!=(36-target if reverse else target):
                    raise AssertionError('Changed host inputs returned a stale or wrong document ID')
    if plan.calls!=2 or plan.fallback_calls!=2:raise AssertionError('Default library fallback was not used')
    print(json.dumps({'operation':'batch_refresh_top5_v1','accepted_custom_artifact':False,
        'status':plan.status,'calls':plan.calls,'fallback_calls':plan.fallback_calls,
        'changed_input_top_match_checks':8,'source_sha256':source_identity(),
        'first':first,'changed_documents':changed},indent=2))


if __name__=='__main__':main()
