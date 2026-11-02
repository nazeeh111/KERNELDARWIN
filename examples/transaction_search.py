"""Two changed-input batch searches using only an installed library package."""
import json

from kerneldarwin.search import text_vectors
from kerneldarwin.transaction_search import search_batches, transaction_source_identity


def main():
    documents = [
        'Metal kernels numerical validation',
        'Vector index cosine similarity',
        'Compiled MLX query batching',
        'Fallback for unknown inputs',
        'Memory allocation and dispatch',
        'Independent frozen performance study',
    ]
    queries = [text_vectors([documents[0]],384),
               text_vectors([documents[1],documents[2]],384),
               text_vectors([documents[5]],384)]
    first = search_batches(text_vectors(documents,384),queries,k=3)
    second = search_batches(text_vectors(documents[::-1],384),queries,k=3)
    expected = ((0,),(1,2),(5,))
    for batches,reverse in ((first,False),(second,True)):
        for (ids,_),targets in zip(batches,expected):
            for row,target in zip(ids,targets):
                wanted = len(documents)-1-target if reverse else target
                if row[0] != wanted:
                    raise AssertionError('Changed-input batch search returned a stale or wrong document ID')
    print(json.dumps({'operation':'compiled_batch_refresh_and_search',
        'dispatch':'compiled_stable_library','accepted_custom_artifact':False,
        'changed_input_top_match_checks':8,'source_sha256':transaction_source_identity(),
        'first':first,'changed_documents':second},indent=2))


if __name__=='__main__':main()
