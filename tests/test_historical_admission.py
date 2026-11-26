"""Integrity and precision boundaries for the historical evidence route."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import importlib.util
from kerneldarwin.control import reference
from kerneldarwin.records import read_record_file
from kerneldarwin import _historical_validation

ROOT = Path(__file__).resolve().parents[1]

class HistoricalAdmissionTests(unittest.TestCase):
    def test_duplicate_keys_rejected_before_historical_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'record.json'
            path.write_text('{"id":"one","id":"two"}')
            with self.assertRaisesRegex(ValueError, 'Duplicate record key'):
                read_record_file(path, validator=_historical_validation.validate_record)

    def test_optimized_checker_refuses_to_run(self):
        env={**os.environ,'PYTHONPATH':str(ROOT/'src')}
        p=subprocess.run([sys.executable,'-O',str(ROOT/'scripts/check_evidence.py')],cwd=ROOT,env=env,capture_output=True,text=True)
        self.assertNotEqual(p.returncode,0)
        self.assertIn('requires Python without optimization',p.stderr)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is optional")
    def test_fused_reference_uses_float64_intermediate(self):
        import numpy as np
        x=np.array([[0.027]],dtype=np.float32)
        actual=reference('fused',x,0)
        expected=np.maximum(x.astype(np.float64)*1.5+.25,0).astype(np.float32)
        float32=np.maximum(x*1.5+.25,0)
        np.testing.assert_array_equal(actual,expected)
        self.assertNotEqual(actual.item(),float32.item())
