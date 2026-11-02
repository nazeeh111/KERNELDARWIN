"""CPU checks for timing units and prevention of early confirmation access."""
import importlib.util
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('transaction_measurement_contract',ROOT/'scripts/transaction_study.py')
study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)


class TransactionMeasurementTests(unittest.TestCase):
    def test_palindrome_balances_each_nominal_position_and_folds_once(self):
        methods={name:object() for name in study.METHODS}
        order,raw=study.group(methods,lambda method:{'cost_us':3.},random.Random(44211))
        self.assertEqual(order[:8],order[8:][::-1])
        self.assertEqual(set(order[:8]),set(methods))
        for name in methods:
            positions=[i for i,n in enumerate(order) if n==name]
            self.assertEqual(sum(positions),15)
            self.assertEqual(len(raw[name]),2)
            self.assertEqual(study.folded([raw[name]]),[3.])

    def test_rank_normalizes_full_outer_window_and_keeps_all_inner_costs(self):
        with patch.object(study.base,'measure',return_value=1280.),patch.object(study.time,'perf_counter_ns',side_effect=[0,50_000_000]):
            result=study.window_rank(object(),[(None,None)]*128)
        self.assertEqual(result['inner_us'],[10.]*32)
        self.assertEqual(result['outer_us'],50_000.)
        self.assertEqual(result['cost_us'],50_000/(32*128))
        with self.assertRaises(ValueError):study.window_rank(object(),[])

    def test_both_clocks_and_completion_hash_are_required_before_reserving(self):
        previous={'session':0,'policy_sha256':'pin','protocol_sha256':'pin'}
        completed={'result_sha256':'pin','session_end_unix_ns':1_000_000_000,'session_end_monotonic_ns':2_000_000_000}
        def read(name):return completed if name.endswith('-completed.json') else previous
        with tempfile.TemporaryDirectory() as temp,patch.object(study,'EVIDENCE',Path(temp)),patch.object(study,'read',side_effect=read),patch.object(study,'digest',return_value='pin'):
            for unix,monotonic in ((121_000_000_000,121_999_999_999),(120_999_999_999,122_000_000_000)):
                with self.assertRaises(RuntimeError):study.session_preflight(1,{}, {},unix,monotonic)
            study.session_preflight(1,{}, {},121_000_000_000,122_000_000_000)
            completed['result_sha256']='changed'
            with self.assertRaises(ValueError):study.session_preflight(1,{}, {},121_000_000_000,122_000_000_000)
            self.assertEqual(list(Path(temp).iterdir()),[])

    def test_cold_work_refuses_early_or_failed_confirmation(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(study,'EVIDENCE',Path(temp)):
            for i in range(3):
                with self.assertRaises(ValueError):study.require_confirmations()
                (Path(temp)/f'confirmation-{i}-completed.json').write_text('{}')
            study.require_confirmations()
            (Path(temp)/'confirmation-1-failed.json').write_text('{}')
            with self.assertRaises(ValueError):study.require_confirmations()
