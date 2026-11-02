"""Synthetic CPU guard evidence only, never numerical or performance evidence.

No MLX runtime, Metal compilation or GPU execution is used. Manifests deliberately
mock device/study identities; their confirmation flags do not represent a study.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import kerneldarwin.rank_search as rank


class MetadataArray:
    """Array metadata sufficient for the real validation and routing guards."""
    def __init__(self, shape=(1, 384), dtype="float32", strides=None):
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = dtype
        self.strides = strides


@contextmanager
def metadata_mlx():
    core = ModuleType("mlx.core")
    core.array = MetadataArray
    core.float32, core.float16, core.bfloat16 = "float32", "float16", "bfloat16"
    core.gpu, core.cpu = "gpu", "cpu"
    core.default_device = Mock(return_value=core.gpu)
    parent = ModuleType("mlx")
    parent.core = core
    with patch.dict(sys.modules, {"mlx": parent, "mlx.core": core}):
        yield core


class RankContractTests(unittest.TestCase):
    def setUp(self):
        self.study_patch = patch.object(rank, "rank_implementation_identity", return_value="a" * 64)
        self.environment_patch = patch.object(rank, "environment_identity", return_value={"cpu_guard_fixture": True})
        self.study_patch.start()
        self.environment_patch.start()
        self.addCleanup(self.study_patch.stop)
        self.addCleanup(self.environment_patch.stop)

    def manifest(self):
        return {
            "schema": 1, "operation": "stable_top5_pairs_v1", "epsilon": 1e-12,
            "fixed_k": 5, "key_policy": rank.KEY_POLICY,
            "compile_options": {"math_mode": "safe"},
            "source_sha256": rank.rank_source_identity(), "study_identity": "a" * 64,
            "environment": {"cpu_guard_fixture": True},
            "sources": {name: hashlib.sha256(content).hexdigest() for name, content in rank._SOURCES},
            "rules": [{"shape": [1, 64, 384, 5], "candidate": "simd32",
                       "dtype": "float32", "layout": "any_mlx_strides"}],
            "evidence": {"numerical_passed": True, "confirmation_passed": True},
        }

    @contextmanager
    def artifact(self, manifest=None):
        with tempfile.TemporaryDirectory(prefix="kd-rank-cpu-guard-") as directory:
            root = Path(directory)
            for name, content in rank._SOURCES:
                (root / name).write_bytes(content)
            (root / "manifest.json").write_text(json.dumps(self.manifest() if manifest is None else manifest))
            yield root

    def assert_fallback(self, plan):
        self.assertTrue(plan.status.startswith("fallback:"), plan.status)
        self.assertEqual(plan._rules, {})
        self.assertEqual((plan.calls, plan.fallback_calls), (0, 0))
        self.assertEqual(plan.last_decision.implementation, "library")

    def test_default_and_valid_synthetic_artifact_status(self):
        default = rank.RankPlan()
        self.assertEqual(default.status, "library_only")
        self.assertEqual(default._rules, {})
        with self.artifact() as root:
            for path in (root, root / "manifest.json"):
                installed = rank.RankPlan(path)
                self.assertEqual(installed.status, "installed")
                self.assertEqual(installed._rules, {(1, 64, 384, 5): "simd32"})

    def test_missing_malformed_and_oversized_manifests_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            self.assert_fallback(rank.RankPlan(path))
            for data in (b"{", b"[]", b"null", b"{}", b"\xff", b" " * 1_000_001):
                with self.subTest(data=data[:12]):
                    path.write_bytes(data)
                    self.assert_fallback(rank.RankPlan(path))
            with patch.object(Path, "open", side_effect=AssertionError("oversized manifest must not open")):
                self.assert_fallback(rank.RankPlan(path))

    def test_stale_source_short_circuits_study_and_device_probes(self):
        data = self.manifest()
        data["source_sha256"] = "0" * 64
        with self.artifact(data) as root, \
                patch.object(rank, "rank_implementation_identity", side_effect=AssertionError("stale source")), \
                patch.object(rank, "environment_identity", side_effect=AssertionError("stale source")):
            self.assert_fallback(rank.RankPlan(root))

    def test_stale_identity_environment_and_unavailable_pin_fail_closed(self):
        for field, replacement in (("study_identity", "b" * 64), ("environment", {"other_device": True})):
            with self.subTest(field=field):
                data = self.manifest()
                data[field] = replacement
                with self.artifact(data) as root:
                    self.assert_fallback(rank.RankPlan(root))
        with self.artifact() as root, patch.object(rank, "rank_implementation_identity", side_effect=FileNotFoundError("no real pin")):
            self.assert_fallback(rank.RankPlan(root))

    def test_operation_contract_and_field_types_are_rejected(self):
        changes = {
            "schema": [True, 1.0, "1", 2], "operation": [None, "other"],
            "epsilon": ["1e-12", 1e-6], "fixed_k": [True, 5.0, "5", 4, 6],
            "key_policy": [None, "other"],
            "compile_options": [None, [], {}, {"math_mode": "fast"}, {"math_mode": "safe", "other": True}],
            "rules": [None, {}, "rules"], "evidence": [None, [], {}],
        }
        for field, values in changes.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    data = self.manifest()
                    data[field] = value
                    with self.artifact(data) as root:
                        self.assert_fallback(rank.RankPlan(root))

    def test_rule_bounds_types_layout_dtype_and_candidate_fail_closed(self):
        bad_shapes = [[], [1, 64, 384], [1, 64, 384, 5, 1],
                      [0, 64, 384, 5], [129, 64, 384, 5], [1, 4, 384, 5],
                      [1, 2049, 384, 5], [1, 64, 0, 5], [1, 64, 2049, 5],
                      [1, 64, 384, 4], [128, 2048, 2048, 5],
                      [True, 64, 384, 5], [1.0, 64, 384, 5], ["1", 64, 384, 5],
                      [1, 64, 384, None], {"q": 1}]
        changes = [("shape", shape) for shape in bad_shapes]
        changes += [("candidate", "unapproved.metal"), ("candidate", None),
                    ("dtype", "float16"), ("layout", "contiguous_only")]
        for field, value in changes:
            with self.subTest(field=field, value=value):
                data = self.manifest()
                data["rules"][0][field] = value
                with self.artifact(data) as root:
                    self.assert_fallback(rank.RankPlan(root))
        for rule in (None, [], {}, {"shape": [1, 64, 384, 5]}):
            data = self.manifest()
            data["rules"] = [rule]
            with self.subTest(rule=rule), self.artifact(data) as root:
                self.assert_fallback(rank.RankPlan(root))

    def test_duplicate_and_excessive_rules_cannot_leave_partial_plan(self):
        for count in (2, 129):
            data = self.manifest()
            data["rules"] *= count
            with self.subTest(count=count), self.artifact(data) as root:
                self.assert_fallback(rank.RankPlan(root))
        data = self.manifest()
        data["rules"].append({"shape": [2, 64, 384, 5], "candidate": "third", "dtype": "float32", "layout": "any_mlx_strides"})
        with self.artifact(data) as root:
            self.assert_fallback(rank.RankPlan(root))

    def test_exact_boundary_rules_accept_at_most_two_known_variants(self):
        data = self.manifest()
        shapes = ([1, 5, 1, 5], [128, 2048, 64, 5])
        data["rules"] = [{"shape": shape, "candidate": variant, "dtype": "float32", "layout": "any_mlx_strides"}
                         for shape, variant in zip(shapes, rank.RANK_CANDIDATES)]
        with self.artifact(data) as root:
            installed = rank.RankPlan(root)
            self.assertEqual(installed.status, "installed")
            self.assertEqual(set(installed._rules), {tuple(shape) for shape in shapes})

    def test_confirmation_required_publicly_numerical_required_even_for_trial(self):
        data = self.manifest()
        data["evidence"]["confirmation_passed"] = False
        with self.artifact(data) as root:
            self.assert_fallback(rank.RankPlan(root))
        trial = rank.RankPlan._trial(data)
        self.assertEqual(trial.status, "numerical_trial")
        self.assertEqual(trial._rules, {(1, 64, 384, 5): "simd32"})
        for field in ("numerical_passed", "confirmation_passed"):
            for value in (False, 1, "true", None):
                rejected = self.manifest()
                rejected["evidence"][field] = value
                with self.subTest(field=field, value=value), self.artifact(rejected) as root:
                    self.assert_fallback(rank.RankPlan(root))
        data["evidence"]["numerical_passed"] = False
        with self.assertRaisesRegex(ValueError, "Unconfirmed"):
            rank.RankPlan._trial(data)

    def test_artifact_bytes_cannot_be_relabelled_by_manifest_repinning(self):
        for mutation in ("bytes", "bytes_and_manifest", "manifest_only", "extra_hash", "missing_source"):
            with self.subTest(mutation=mutation), self.artifact() as root:
                path = root / "manifest.json"
                data = json.loads(path.read_text())
                name, original = rank._SOURCES[0]
                if mutation in ("bytes", "bytes_and_manifest"):
                    changed = bytes([original[0] ^ 1]) + original[1:]
                    (root / name).write_bytes(changed)
                    if mutation == "bytes_and_manifest":
                        data["sources"][name] = hashlib.sha256(changed).hexdigest()
                elif mutation == "manifest_only":
                    data["sources"][name] = "0" * 64
                elif mutation == "extra_hash":
                    data["sources"]["other.metal"] = "0" * 64
                else:
                    (root / name).unlink()
                path.write_text(json.dumps(data))
                self.assert_fallback(rank.RankPlan(root))

    def test_source_length_checked_before_open_even_if_manifest_claims_original(self):
        for size in (1, rank.MAX_MSL_BYTES + 1):
            with self.subTest(size=size), self.artifact() as root:
                source = root / rank._SOURCES[0][0]
                source.write_bytes(b"x" * size)
                original_open = Path.open

                def guarded_open(path, *args, **kwargs):
                    if path == source:
                        raise AssertionError("source length must reject before open")
                    return original_open(path, *args, **kwargs)

                with patch.object(Path, "open", autospec=True, side_effect=guarded_open):
                    self.assert_fallback(rank.RankPlan(root))

    def test_source_identity_uses_loaded_snapshot(self):
        expected = rank.rank_source_identity()
        self.assertEqual(len(expected), 64)
        self.assertLessEqual(sum(len(content) for _, content in rank._SOURCES), rank.MAX_MSL_BYTES)
        with patch.object(Path, "read_bytes", side_effect=AssertionError("must use immutable snapshot")):
            self.assertEqual(rank.rank_source_identity(), expected)

    def test_metadata_only_routing_requires_exact_tuple_dtype_and_default_gpu(self):
        trial = rank.RankPlan._trial(self.manifest())
        with metadata_mlx() as mx:
            docs = MetadataArray((64, 384), strides=(0, -2))
            query = MetadataArray(strides=(-384, 0))
            self.assertEqual(trial._route(query, docs, 5), "simd32")
            self.assertEqual((trial.calls, trial.fallback_calls), (1, 0))
            self.assertEqual(trial.last_decision.implementation, "simd32")
            cases = [(MetadataArray((2, 384)), docs, 5), (MetadataArray((129, 384)), docs, 5),
                     (query, MetadataArray((65, 384)), 5),
                     (MetadataArray((1, 385)), MetadataArray((64, 385)), 5),
                     (query, docs, 4), (query, docs, True),
                     (MetadataArray(dtype="float16"), docs, 5),
                     (query, MetadataArray((64, 384), dtype="bfloat16"), 5)]
            for q, d, k in cases:
                self.assertIsNone(trial._route(q, d, k))
                self.assertEqual(trial.last_decision.implementation, "library")
            mx.default_device.return_value = mx.cpu
            self.assertIsNone(trial._route(query, docs, 5))
            self.assertEqual((trial.calls, trial.fallback_calls), (10, 9))
            before = trial.calls
            with self.assertRaises(ValueError):
                trial._route(MetadataArray((1, 383)), docs, 5)
            with self.assertRaises(TypeError):
                trial._route(object(), docs, 5)
            self.assertEqual(trial.calls, before)

    def test_index_dispatch_forwards_current_arrays_and_whole_stable_fallback(self):
        trial = rank.RankPlan._trial(self.manifest())
        index = object.__new__(rank.RankIndex)
        index.plan, index.vectors = trial, MetadataArray((64, 384))
        custom_graph, stable_graph = Mock(return_value="custom"), Mock(return_value="stable")
        with metadata_mlx() as mx, \
                patch.object(rank, "compiled_query", return_value=custom_graph) as custom, \
                patch.object(rank, "_compiled_search", return_value=stable_graph) as fallback, \
                patch.object(rank, "_search_results", side_effect=lambda value: value):
            first, second = MetadataArray(), MetadataArray()
            self.assertEqual(index.search(first), "custom")
            self.assertEqual(index.search(second), "custom")
            self.assertIs(custom_graph.call_args_list[0].args[0], first)
            self.assertIs(custom_graph.call_args_list[1].args[0], second)
            self.assertIs(custom_graph.call_args_list[1].args[1], index.vectors)
            self.assertEqual(custom.call_count, 2)
            fallback.assert_not_called()
            self.assertEqual(index.search(first, 4), "stable")
            fallback.assert_called_with("stable", 4)
            stable_graph.assert_called_with(first, index.vectors)
            mx.default_device.return_value = mx.cpu
            self.assertEqual(index.search(first), "stable")
            fallback.assert_called_with("stable", 5)
            self.assertEqual((trial.calls, trial.fallback_calls), (4, 2))
            self.assertEqual(index.search(MetadataArray((0, 384))), ([], []))
            with self.assertRaises(ValueError):
                index.search(first, 0)
            self.assertEqual((trial.calls, trial.fallback_calls), (4, 2))

    def test_non_inference_mode_rejected_before_index_construction(self):
        with self.assertRaisesRegex(ValueError, "inference"):
            rank.RankPlan().index(object(), mode="training")


if __name__ == "__main__":
    unittest.main()
