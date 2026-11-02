"""CPU regressions for independently associating returned search IDs and scores."""
from importlib.util import module_from_spec,spec_from_file_location
from pathlib import Path
import unittest


class ApplicationMeasurementTests(unittest.TestCase):
    def test_wrong_ids_cannot_pass_with_correct_top_scores(self):
        import numpy as np
        path=Path(__file__).resolve().parents[1]/'scripts/workload_study.py'
        spec=spec_from_file_location('row_study_id_check',path)
        study=module_from_spec(spec);spec.loader.exec_module(study)
        docs=np.array([[1,0],[.8,.6],[0,1],[-.8,.6],[-1,0]],np.float32)
        query=np.array([[1,0]],np.float32)
        output=[]
        for j in range(3):
            q=(query+j*.001).astype(np.float64)
            q/=np.linalg.norm(q,axis=1,keepdims=True)
            d=docs.astype(np.float64);d/=np.linalg.norm(d,axis=1,keepdims=True)
            scores=q@d.T;ids=np.argsort(-scores,axis=1)
            output.append((ids.tolist(),np.take_along_axis(scores,ids,axis=1).tolist()))
        study.check_application(output,docs,query)
        wrong=[([list(reversed(ids[0]))],values) for ids,values in output]
        with self.assertRaises(AssertionError):study.check_application(wrong,docs,query)
        duplicate=[([[0,0,2,3,4]],values) for ids,values in output]
        with self.assertRaises(AssertionError):study.check_application(duplicate,docs,query)
