import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kerneldarwin.records import CandidateConfig, ExperimentStore, WorkloadSpec, validate_record
from kerneldarwin.statistics import compare_paired, compare_regression, summarize
from kerneldarwin.hypotheses import diagnose


def record():
    return {"schema_version": 1, "id": "control-1", "status": "no_improvement",
            "evidence_kind": "cpu_control", "workload": WorkloadSpec("softmax", (3, 17)).to_dict(),
            "device": {"kind": "cpu", "name": "test"}, "experiments": [], "winner_id": None}


def confirmed_record():
    """Synthetic CPU-only statistics fixture, never published hardware evidence."""
    data = record()
    first = [10. + (index % 7) for index in range(20)]
    second = [value / 2. for value in first]
    holdout = [12. + (index % 3) for index in range(20)]
    improved = [value / 2.1 for value in holdout]
    discovery = {"baseline_us": first, "candidate_us": second,
                 "comparison": compare_paired(first, second, seed=4), "seed": 11}
    confirmation = {"baseline_us": holdout, "candidate_us": improved,
                    "comparison": compare_paired(holdout, improved, seed=7001),
                    "seed": 7001, "correctness_passed": True, "actual_strides": [17, 1]}
    candidate = {"id": "candidate", "status": "accepted", "correctness": {"passed": True},
                 "timing": discovery, "confirmation": confirmation}
    data.update(status="accepted", winner_id="candidate", experiments=[candidate])
    data["workload"]["actual_strides"] = [17, 1]
    return data


class ContractTests(unittest.TestCase):
    def test_workload_roundtrip_and_semantic_limits(self):
        spec = WorkloadSpec("sum", (1, 129), stride=3, dtype="bfloat16")
        self.assertEqual(WorkloadSpec.from_dict(json.loads(json.dumps(spec.to_dict()))), spec)
        for change in ({"shape": (1, 0)}, {"shape": (1,)}, {"shape": (True, 3)},
                       {"stride": False}, {"eps": float("nan")}, {"dtype": "float64"}, {"seed": -1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                WorkloadSpec.from_dict({**spec.to_dict(), **change})
        self.assertEqual(WorkloadSpec.from_dict({**spec.to_dict(), "actual_strides": [387, 3]}), spec)
        with self.assertRaises(ValueError):
            WorkloadSpec.from_dict({**spec.to_dict(), "actual_strides": [1, 0]})
        self.assertEqual(CandidateConfig(1, 1).candidate_id, "b1-w1")
        with self.assertRaises(ValueError):
            CandidateConfig(255, 4)

    def test_store_preserves_bytes_and_rejects_traversal_or_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ExperimentStore(directory)
            original = record()
            path = store.save(original)
            original_bytes = path.read_bytes()
            self.assertEqual(store.load("control-1"), original)
            self.assertEqual(store.list(), [original])
            with self.assertRaises(FileExistsError):
                store.save({**original, "title": "replacement"})
            self.assertEqual(path.read_bytes(), original_bytes)
            with self.assertRaises(ValueError):
                store.load("../outside")
            with self.assertRaises(ValueError):
                store.save({**original, "id": "../outside"})
            self.assertFalse(list(Path(directory).glob(".record-*")))

    def test_gpu_provenance_consistency_and_nonfinite_rejection(self):
        valid = record()
        with self.assertRaises(ValueError):
            validate_record({**valid, "evidence_kind": "cuda_run"})
        valid.update(evidence_kind="cuda_run", device={"kind": "cuda"}, provenance={"gpu_execution": True})
        validate_record(valid)
        with self.assertRaises(ValueError):
            validate_record({**valid, "details": {"counter": float("inf")}})

    def test_invalid_candidate_cannot_be_promoted(self):
        valid = confirmed_record()
        validate_record(valid)
        mutations = [
            lambda data: data["experiments"][0]["correctness"].update(passed=False),
            lambda data: data["experiments"][0]["timing"]["comparison"].update(accepted=False),
            lambda data: data["experiments"][0]["confirmation"].update(correctness_passed=False),
            lambda data: data["experiments"][0]["confirmation"].pop("correctness_passed"),
            lambda data: data["experiments"][0].pop("confirmation"),
            lambda data: data.update(experiments=[]),
            lambda data: data.update(status="no_improvement"),
            lambda data: data.update(winner_id=None),
        ]
        for mutation in mutations:
            data = copy.deepcopy(valid)
            mutation(data)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_record(data)

    def test_confirmation_must_have_fresh_seeds_and_preserve_layout(self):
        valid = confirmed_record()
        for seed in (None, 0, 4, 11, True):
            data = copy.deepcopy(valid)
            data["experiments"][0]["confirmation"]["seed"] = seed
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                validate_record(data)
        copied = copy.deepcopy(valid)
        copied["experiments"][0]["confirmation"] = copy.deepcopy(copied["experiments"][0]["timing"])
        copied["experiments"][0]["confirmation"]["correctness_passed"] = True
        with self.assertRaises(ValueError):
            validate_record(copied)
        data = copy.deepcopy(valid)
        data["experiments"][0]["confirmation"]["actual_strides"] = [34, 2]
        with self.assertRaises(ValueError):
            validate_record(data)
        with self.assertRaises(ValueError):
            WorkloadSpec.from_dict({**valid["workload"], "actual_strides": [34, 2]})

    def test_promotion_statistics_are_recomputed_from_raw_pairs(self):
        valid = confirmed_record()
        for phase_name in ("timing", "confirmation"):
            for field, value in (("ci_low", 1.9), ("ci_high", 2.9), ("speedup", 9.),
                                 ("confidence_level", .99), ("n", 19), ("method", "invented")):
                data = copy.deepcopy(valid)
                data["experiments"][0][phase_name]["comparison"][field] = value
                with self.subTest(phase=phase_name, field=field), self.assertRaises(ValueError):
                    validate_record(data)
        # A positive median speedup alone cannot justify a noise-spanning CI.
        data = copy.deepcopy(valid)
        discovery = data["experiments"][0]["timing"]
        discovery["baseline_us"] = [100.] * 20
        discovery["candidate_us"] = [25., 125.] * 10
        discovery["comparison"].update(speedup=100/75, ci_low=1.2, ci_high=4.)
        with self.assertRaises(ValueError):
            validate_record(data)

    def test_promotion_recomputation_is_resource_bounded(self):
        valid = confirmed_record()
        for resamples in (None, True, 99, 10001, 10**12):
            data = copy.deepcopy(valid)
            data["experiments"][0]["timing"]["comparison"]["bootstrap_resamples"] = resamples
            with self.subTest(resamples=resamples), patch("kerneldarwin.statistics.compare_paired") as recompute:
                with self.assertRaises(ValueError):
                    validate_record(data)
                recompute.assert_not_called()
        data = copy.deepcopy(valid)
        data["experiments"][0]["timing"]["baseline_us"] = [10.] * 10001
        data["experiments"][0]["timing"]["candidate_us"] = [5.] * 10001
        with patch("kerneldarwin.statistics.compare_paired") as recompute:
            with self.assertRaises(ValueError):
                validate_record(data)
            recompute.assert_not_called()

    def test_hypotheses_expose_missing_evidence_and_disproof(self):
        hints = diagnose(WorkloadSpec("layernorm", (16, 511), stride=2).to_dict(),
                         {"summary": {"median_us": 12}}, {"kind": "cpu"})
        self.assertIn("strided-access", [hint["id"] for hint in hints])
        for hint in hints:
            self.assertEqual(hint["status"], "untested")
            self.assertTrue(hint["evidence"])
            self.assertTrue(hint["missing_evidence"])
            self.assertTrue(hint["falsifier"])
            self.assertEqual(hint["confidence"], "low")


class StatisticsTests(unittest.TestCase):
    def test_repeatable_speedup_and_regression(self):
        base = [100 + i % 7 for i in range(40)]
        candidate = [value / 2 for value in base]
        comparison = compare_paired(base, candidate)
        self.assertTrue(comparison["accepted"])
        self.assertAlmostEqual(comparison["speedup"], 2)
        self.assertAlmostEqual(comparison["ci_low"], 2)
        self.assertTrue(compare_regression(candidate, base)["regression"])

    def test_noisy_or_small_effect_and_insufficient_pairs_rejected(self):
        self.assertFalse(compare_paired([100] * 40, [99] * 40)["accepted"])
        self.assertFalse(compare_paired([100] * 5, [10] * 5)["accepted"])
        self.assertFalse(compare_paired([100] * 40, [25, 200] * 20)["accepted"])

    def test_invalid_timings_and_descriptive_statistics(self):
        self.assertEqual(summarize([1, 2, 3])["median_us"], 2)
        for samples in ([], [0], [-1], [float("nan")], [True]):
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                summarize(samples)
        with self.assertRaises(ValueError):
            compare_paired([1], [1, 2])
        self.assertEqual(compare_paired([10, 11] * 20, [5, 6] * 20, seed=3),
                         compare_paired([10, 11] * 20, [5, 6] * 20, seed=3))


if __name__ == "__main__":
    unittest.main()
