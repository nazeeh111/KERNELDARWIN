"""CPU checks of the prospective measurement rules, not GPU evidence."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "scripts/rank_study.py"
SPEC = importlib.util.spec_from_file_location("rank_measurement_rules", SOURCE)
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


class RankMeasurementTests(unittest.TestCase):
    def test_pairing_preserves_exact_gain_through_large_shared_outliers(self):
        baseline = [1., 1024., 2., 8., 4., 4096., 16., 64.] * 5
        result = study.intervals(baseline, [v / 2 for v in baseline])
        self.assertEqual(result["speedup_total_cost"], 2.)
        for key in ("ci95", "iid_ci95", "block4_ci95", "block8_ci95"):
            self.assertEqual(result[key], [2., 2.])

    def test_serial_dependence_can_reject_an_iid_only_application_win(self):
        result = study.intervals([100.] * 40, [55.] * 20 + [112.] * 20)
        self.assertGreater(result["iid_ci95"][0], 1.03)
        self.assertLess(result["ci95"][0], 1.03)
        self.assertGreater(result["speedup_total_cost"], 1.03)

    def test_label_swap_retains_noise_margin_and_never_weakens_old_margin(self):
        a = [10., 12., 14., 20., 40., 15., 11., 13.] * 5
        b = [12., 11., 15., 18., 35., 17., 13., 14.] * 5
        forward, reverse = study.intervals(a, b), study.intervals(b, a)
        self.assertAlmostEqual(study.symmetric_noise_floor([forward], 1.10),
                               study.symmetric_noise_floor([reverse], 1.10), places=12)
        for bounds in ([.86, 1.01], [.98, 1.14], [1.01, 1.03], [.95, .99], [1., 1.]):
            old = max(1.10, 1 + 2 * max(abs(v - 1) for v in bounds))
            self.assertGreaterEqual(study.symmetric_noise_floor([{"ci95": bounds}], 1.10), old)

    def test_invalid_costs_counts_and_resample_changes_fail(self):
        for value in (True, False, 0., -1., float("nan"), float("inf"), "1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                study.intervals([value] * 40, [1.] * 40)
        for count in (0, 39, 41, 42, 1001):
            with self.subTest(count=count), self.assertRaises(ValueError):
                study.intervals([1.] * count, [1.] * count)
        with self.assertRaises(ValueError):
            study.intervals([1.] * 40, [1.] * 39)
        with self.assertRaises(ValueError):
            study.intervals([1.] * 40, [1.] * 40, count=2000)

    def test_session_preflight_requires_both_clocks_and_current_identity(self):
        protocol = {"source_sha256": "source", "study_identity": "study", "environment": {},
                    "corpus_sha256": "corpus", "numerical_report_sha256": "numeric",
                    "confirmation": {"seeds": [23101, 24203, 25307],
                                     "sequence_orders": [[1, 0, 3, 2], [2, 3, 0, 1], [0, 2, 1, 3]],
                                     "minimum_separation_seconds": 120}}
        prior = {"schema": 1, "session": 0, "source_sha256": "source", "study_identity": "study",
                 "environment": {}, "corpus_sha256": "corpus", "numerical_report_sha256": "numeric",
                 "protocol_sha256": "digest", "policy_sha256": "digest", "development_sha256": "digest",
                 "seed": 23101, "sequence_order": [1, 0, 3, 2], "process_id": 42,
                 "session_start_unix_ns": 1, "session_end_unix_ns": 10,
                 "session_start_monotonic_ns": 1, "session_end_monotonic_ns": 10}
        boundary = 120_000_000_010
        with patch.object(study, "read", return_value=prior), patch.object(study, "digest", return_value="digest"):
            study.session_preflight(1, {"study_identity": "study"}, protocol, boundary, boundary)
            for clocks in ((boundary - 1, boundary), (boundary, boundary - 1)):
                with self.subTest(clocks=clocks), self.assertRaises(RuntimeError):
                    study.session_preflight(1, {"study_identity": "study"}, protocol, *clocks)
        bad = copy.deepcopy(prior)
        bad["study_identity"] = "stale"
        with patch.object(study, "read", return_value=bad), patch.object(study, "digest", return_value="digest"):
            with self.assertRaises(ValueError):
                study.session_preflight(1, {"study_identity": "study"}, protocol, boundary, boundary)
