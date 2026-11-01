"""Opt-in actual Metal rank checks. Manufactured plans test guards, not speed."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(os.getenv("KD_RUN_METAL") == "1", "Actual Metal execution is opt-in")
class RankSearchTests(unittest.TestCase):
    gpu_executed = False
    completed_methods = []

    @classmethod
    def setUpClass(cls):
        import mlx.core as mx
        import numpy as np
        import kerneldarwin.rank_search as rank
        if not mx.metal.is_available():
            raise unittest.SkipTest("Native Metal backend is unavailable")
        mx.set_default_device(mx.gpu)
        mx.set_memory_limit(2 * 1024 ** 3)
        mx.set_cache_limit(256 * 1024 ** 2)
        # Do not label a skipped suite or a merely imported backend GPU-executed.
        probe = mx.sum(mx.array([1., 2., 3.], dtype=mx.float32))
        mx.eval(probe)
        mx.synchronize()
        if probe.item() != 6. or mx.default_device() != mx.gpu:
            raise RuntimeError("Native GPU execution probe failed")
        cls.mx, cls.np, cls.rank = mx, np, rank
        cls.gpu_executed = True
        cls.completed_methods = []

    def tearDown(self):
        self.mx.synchronize()
        type(self).completed_methods.append(self._testMethodName)

    def view(self, host, layout):
        mx, np = self.mx, self.np
        if layout == "contiguous":
            return mx.array(host)
        if layout == "stride2":
            backing = np.zeros((len(host), host.shape[1] * 2 + 1), np.float32)
            backing[:, 1::2] = host
            return mx.array(backing)[:, 1::2]
        if layout == "negative_columns":
            return mx.array(host)[:, ::-1]
        if layout == "transpose":
            return mx.array(host.T.copy()).T
        if layout == "broadcast_rows":
            return mx.broadcast_to(mx.array(host[:1]), host.shape)
        if layout == "negative_rows":
            return mx.array(host)[::-1, :]
        if layout == "stepped_rows":
            backing = np.zeros((len(host) * 2 + 1, host.shape[1]), np.float32)
            backing[1::2] = host
            return mx.array(backing)[1::2]
        if layout == "broadcast_columns":
            return mx.broadcast_to(mx.array(host[:, :1]), host.shape)
        if layout == "offset_rows_columns":
            backing = np.zeros((len(host) + 2, host.shape[1] + 3), np.float32)
            backing[1:-1, 2:-1] = host
            return mx.array(backing)[1:-1, 2:-1]
        raise ValueError(layout)

    def expected_rank(self, host):
        """Independent integer comparison model, preserving original value bits.

        IEEE float sign/magnitude becomes monotonic unsigned keys; every
        subnormal and signed zero gets the same native FTZ comparison key.
        Argmax on the remaining original columns resolves ties by first index.
        """
        np = self.np
        host = np.asarray(host, dtype=np.float32)
        bits = host.view(np.uint32)
        magnitude = bits & np.uint32(0x7fffffff)
        keys = np.where(bits & np.uint32(0x80000000), ~bits,
                        bits ^ np.uint32(0x80000000)).astype(np.uint32)
        keys = np.where(magnitude < np.uint32(0x00800000), np.uint32(0x80000000), keys)
        active = np.ones(host.shape, dtype=bool)
        chosen = []
        for _ in range(5):
            ids = np.argmax(np.where(active, keys, np.uint32(0)), axis=1)
            chosen.append(ids)
            active[np.arange(len(host)), ids] = False
        ids = np.stack(chosen, axis=1)
        return ids, np.take_along_axis(bits, ids, axis=1)

    def assert_rank(self, scores, variant):
        mx, np, rank = self.mx, self.np, self.rank
        mx.eval(scores)
        host = np.array(scores).copy()
        expected_ids, expected_bits = self.expected_rank(host)
        outputs = {"sort": rank.library_rank()(scores),
                   "reduction": rank.library_reduction_rank()(scores),
                   "direct": rank.candidate_rank(scores, variant),
                   "compiled": rank.compiled_rank(variant)(scores)}
        mx.eval(*[item for pair in outputs.values() for item in pair])
        for name, (ids, values) in outputs.items():
            with self.subTest(variant=variant, implementation=name, shape=scores.shape):
                np.testing.assert_array_equal(np.array(ids), expected_ids)
                np.testing.assert_array_equal(np.array(values).view(np.uint32), expected_bits)
        np.testing.assert_array_equal(np.array(scores).view(np.uint32), host.view(np.uint32))

    def manifest(self, variant="simd32", shape=(3, 9, 33, 5)):
        from kerneldarwin.workload import environment_identity
        rank = self.rank
        return {"schema": 1, "operation": "stable_top5_pairs_v1", "epsilon": 1e-12,
            "fixed_k": 5, "key_policy": "native_ftz_stable_index_original_bits_v1",
            "compile_options": {"math_mode": "safe"},
            "source_sha256": rank.rank_source_identity(),
            "study_identity": rank.rank_implementation_identity(), "environment": environment_identity(),
            "sources": {name: hashlib.sha256(content).hexdigest() for name, content in rank._SOURCES},
            "rules": [{"shape": list(shape), "candidate": variant, "dtype": "float32", "layout": "any_mlx_strides"}],
            "evidence": {"numerical_passed": True, "confirmation_passed": False}}

    def reference(self, queries, docs):
        np = self.np
        def normalize(values):
            host = np.array(values).astype(np.float64)
            host[np.abs(host) < np.finfo(np.float32).tiny] = 0
            peak = np.maximum(np.max(np.abs(host), axis=1, keepdims=True), 1e-12)
            z = host / peak
            return z / np.maximum(np.sqrt(np.sum(z * z, axis=1, keepdims=True)), 1e-12 / peak)
        return normalize(queries) @ normalize(docs).T

    def assert_application(self, index, query, original_docs):
        np = self.np
        ids, values = index.search(query, k=5)
        ids, values = np.asarray(ids), np.asarray(values)
        expected = self.reference(query, original_docs)
        self.assertEqual(ids.shape, (query.shape[0], 5))
        self.assertEqual(values.shape, ids.shape)
        self.assertTrue(np.issubdtype(ids.dtype, np.integer))
        self.assertTrue(np.all((ids >= 0) & (ids < original_docs.shape[0])))
        self.assertTrue(all(len(set(row.tolist())) == 5 for row in ids))
        np.testing.assert_allclose(values, np.take_along_axis(expected, ids, axis=1), rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(values, -np.sort(-expected, axis=1)[:, :5], rtol=2e-5, atol=2e-6)
        return ids, values

    def test_01_rank_shapes_tails_maximum_and_changed_inputs(self):
        np = self.np
        for variant in self.rank.RANK_CANDIDATES:
            for q, n in ((1, 5), (3, 31), (3, 33), (8, 64), (8, 65), (17, 161),
                         (3, 257), (3, 1023), (128, 2048)):
                for seed in (2101, 2102):
                    host = np.random.default_rng(seed).normal(size=(q, n)).astype(np.float32)
                    self.assert_rank(self.mx.array(host), variant)

    def test_02_rank_signed_strides_offsets_and_broadcast(self):
        np = self.np
        layouts = ("stride2", "negative_columns", "transpose", "negative_rows", "stepped_rows",
                   "broadcast_rows", "broadcast_columns", "offset_rows_columns")
        for variant in self.rank.RANK_CANDIDATES:
            for q, n in ((3, 5), (3, 33), (7, 161), (3, 1023), (128, 2048)):
                host = np.random.default_rng(2501).normal(size=(q, n)).astype(np.float32)
                for layout in layouts:
                    with self.subTest(layout=layout):
                        self.assert_rank(self.view(host, layout), variant)

    def test_03_exact_ties_signed_zero_and_extreme_bits(self):
        np = self.np
        cases = (np.zeros((3, 64), np.float32),
            np.tile(np.array([5, 4, 3, 2, 1, 1, 1, -7] * 8, np.float32), (3, 1)),
            np.tile(np.array([-0., 0., -0., 0.] * 16, np.float32), (3, 1)),
            np.tile(np.array([np.finfo(np.float32).max, -np.finfo(np.float32).max,
                              1, -1, 0, -0., 2, 2] * 8, np.float32), (3, 1)))
        for variant in self.rank.RANK_CANDIDATES:
            for host in cases:
                for layout in ("contiguous", "stride2", "negative_columns"):
                    self.assert_rank(self.view(host, layout), variant)

    def test_04_discriminating_subnormals_original_bits(self):
        np = self.np
        host = np.full((3, 64), -1, np.float32)
        host[:, :10] = [np.finfo(np.float32).tiny, 5e-40, 3e-40, 0., -0.,
            np.nextafter(np.float32(0), np.float32(1)), np.nextafter(np.float32(0), np.float32(-1)),
            -5e-40, -3e-40, -np.finfo(np.float32).tiny]
        # This distinguishes FTZ ties from a native CPU non-FTZ float sort.
        expected, _ = self.expected_rank(host)
        self.assertFalse(np.array_equal(expected, np.argsort(-host, axis=1, kind="stable")[:, :5]))
        for variant in self.rank.RANK_CANDIDATES:
            for layout in ("contiguous", "stride2", "negative_columns"):
                self.assert_rank(self.view(host, layout), variant)

    def test_05_nonfinite_private_rows_direct_and_compiled(self):
        mx, np, rank = self.mx, self.np, self.rank
        host = np.ones((4, 64), np.float32)
        host[0, -1], host[1, 4], host[2, 1] = np.nan, np.inf, -np.inf
        for variant in rank.RANK_CANDIDATES:
            for layout in ("stride2", "negative_rows", "negative_columns"):
                scores = self.view(host, layout)
                bad = ~np.all(np.isfinite(np.array(scores)), axis=1)
                for call in (lambda x: rank.candidate_rank(x, variant), rank.compiled_rank(variant)):
                    ids, values = call(scores)
                    mx.eval(ids, values)
                    ids, values = np.array(ids), np.array(values)
                    np.testing.assert_array_equal(ids[bad], np.full((3, 5), np.iinfo(np.uint32).max, np.uint32))
                    self.assertTrue(np.isnan(values[bad]).all())
                    np.testing.assert_array_equal(ids[~bad], np.arange(5)[None, :])
                    np.testing.assert_array_equal(values[~bad], np.ones((1, 5)))

    def test_06_whole_queries_changed_refresh_all_libraries_float64(self):
        from kerneldarwin.search import CompiledCosineIndex
        mx, np, rank = self.mx, self.np, self.rank
        for variant in rank.RANK_CANDIDATES:
            for q, n, d in ((1, 64, 384), (8, 161, 768), (17, 161, 1536),
                            (3, 37, 97), (3, 5, 1), (128, 2048, 37)):
                rng = np.random.default_rng(2901)
                plan = rank.RankPlan._trial(self.manifest(variant, (q, n, d, 5)))
                indexes = None
                for change in (0, 1):
                    docs = self.view(rng.normal(size=(n, d)).astype(np.float32), "stride2")
                    query = self.view(rng.normal(size=(q, d)).astype(np.float32), "negative_columns")
                    if indexes is None:
                        indexes = [plan.index(docs), *(CompiledCosineIndex(docs, kind) for kind in ("stable", "fast")),
                                   *(rank.ReductionIndex(docs, kind) for kind in ("stable", "fast"))]
                    else:
                        for index in indexes:
                            index.refresh(docs)
                    before = plan.calls
                    outputs = [self.assert_application(index, query, docs) for index in indexes]
                    self.assertEqual(plan.calls, before + 1)
                    self.assertEqual(plan.fallback_calls, 0)
                    for _, values in outputs[1:]:
                        np.testing.assert_allclose(values, outputs[0][1], rtol=2e-5, atol=2e-6)
                    for graph in (rank.compiled_query(variant), rank.compiled_reduction_query("stable"),
                                  rank.compiled_reduction_query("fast")):
                        ids, values, finite = graph(query, indexes[0].vectors)
                        mx.eval(ids, values, finite)
                        self.assertTrue(finite.item())
                        reference = self.reference(query, docs)
                        np.testing.assert_allclose(np.array(values), np.take_along_axis(reference, np.array(ids), axis=1),
                                                   rtol=2e-5, atol=2e-6)

    def test_07_complete_query_duplicate_document_ties(self):
        from kerneldarwin.search import CompiledCosineIndex
        mx, np, rank = self.mx, self.np, self.rank
        rng = np.random.default_rng(3101)
        docs = mx.array(np.repeat(rng.normal(size=(1, 33)).astype(np.float32), 64, axis=0))
        query = mx.array(rng.normal(size=(3, 33)).astype(np.float32))
        expected = np.tile(np.arange(5), (3, 1))
        for variant in rank.RANK_CANDIDATES:
            plan = rank.RankPlan._trial(self.manifest(variant, (3, 64, 33, 5)))
            for index in (plan.index(docs), *(CompiledCosineIndex(docs, kind) for kind in ("stable", "fast")),
                          *(rank.ReductionIndex(docs, kind) for kind in ("stable", "fast"))):
                ids, _ = self.assert_application(index, query, docs)
                np.testing.assert_array_equal(ids, expected)

    def test_08_plan_counters_shape_k_dtype_and_empty_fallback(self):
        from kerneldarwin.search import CompiledCosineIndex
        mx, np, rank = self.mx, self.np, self.rank
        rng = np.random.default_rng(3601)
        for variant in rank.RANK_CANDIDATES:
            docs = mx.array(rng.normal(size=(9, 33)).astype(np.float32))
            plan = rank.RankPlan._trial(self.manifest(variant))
            index = plan.index(docs)
            for seed in (1, 2, 3):
                query = mx.array(np.random.default_rng(seed).normal(size=(3, 33)).astype(np.float32))
                self.assert_application(index, query, docs)
            self.assertEqual((plan.calls, plan.fallback_calls), (3, 0))
            self.assertEqual(index.search(mx.zeros((0, 33))), ([], []))
            self.assertEqual(plan.calls, 3)
            for invalid_k in (0, 10, True):
                with self.assertRaises(ValueError):
                    index.search(mx.ones((3, 33)), invalid_k)
            with self.assertRaises(ValueError):
                index.search(mx.ones((3, 34)))
            self.assertEqual(plan.calls, 3)
            for query, k in ((mx.ones((2, 33)), 5), (mx.ones((129, 33)), 5), (mx.ones((3, 33)), 3)):
                expected = CompiledCosineIndex(docs, "stable").search(query, k)
                self.assertEqual(index.search(query, k), expected)
            self.assertEqual((plan.calls, plan.fallback_calls), (6, 3))
            half_docs = docs.astype(mx.float16)
            half_query = mx.ones((3, 33), dtype=mx.float16)
            half = plan.index(half_docs)
            self.assertEqual(half.search(half_query), CompiledCosineIndex(half_docs, "stable").search(half_query))
            self.assertEqual((plan.calls, plan.fallback_calls), (7, 4))
            empty = rank.RankPlan._trial({**self.manifest(variant), "rules": []})
            self.assertEqual(empty.index(docs).search(mx.ones((3, 33))),
                             CompiledCosineIndex(docs, "stable").search(mx.ones((3, 33))))
            self.assertEqual((empty.calls, empty.fallback_calls), (1, 1))

    def test_09_public_nonfinite_empty_rollback_all_indexes(self):
        from kerneldarwin.search import CompiledCosineIndex
        mx, np, rank = self.mx, self.np, self.rank
        docs = mx.ones((64, 17))
        for variant in rank.RANK_CANDIDATES:
            plan = rank.RankPlan._trial(self.manifest(variant, (3, 64, 17, 5)))
            for index in (plan.index(docs), *(CompiledCosineIndex(docs, kind) for kind in ("stable", "fast")),
                          *(rank.ReductionIndex(docs, kind) for kind in ("stable", "fast"))):
                before = np.array(index.vectors).copy()
                for bad in (float("nan"), float("inf"), -float("inf")):
                    host = np.ones((64, 17), np.float32)
                    host[0, 0] = bad
                    with self.assertRaises(ValueError):
                        index.refresh(mx.array(host))
                    np.testing.assert_array_equal(np.array(index.vectors), before)
                    query = np.ones((3, 17), np.float32)
                    query[1, 0] = bad
                    with self.assertRaises(ValueError):
                        index.search(mx.array(query))
                self.assertEqual(index.search(mx.zeros((0, 17))), ([], []))
                ids, _ = index.search(mx.ones((3, 17)), k=3)
                np.testing.assert_array_equal(ids, np.tile(np.arange(3), (3, 1)))
                # A failed refresh leaves the old valid index usable.
                self.assert_application(index, mx.ones((3, 17)), docs)

    def test_10_artifact_source_schema_and_confirmation_guards(self):
        rank = self.rank
        original = self.manifest()
        self.assertEqual(rank.RankPlan._trial(original).status, "numerical_trial")
        malformed = [("operation", "other"), ("fixed_k", True), ("key_policy", "other"),
                     ("compile_options", {"math_mode": "fast"}), ("source_sha256", "0" * 64),
                     ("study_identity", "0" * 64), ("environment", {}), ("sources", {})]
        for key, bad in malformed:
            altered = copy.deepcopy(original)
            altered[key] = bad
            with self.subTest(key=key), self.assertRaises((ValueError, TypeError, KeyError)):
                rank.RankPlan._trial(altered)
        for edit in (lambda x: x["rules"].append(copy.deepcopy(x["rules"][0])),
                     lambda x: x["rules"][0].update(shape=[129, 9, 33, 5]),
                     lambda x: x["rules"][0].update(shape=[3, 9, 33, 3]),
                     lambda x: x["rules"][0].update(dtype="float16"),
                     lambda x: x["rules"][0].update(layout="contiguous"),
                     lambda x: x["rules"][0].update(candidate="unknown"),
                     lambda x: x["evidence"].update(numerical_passed=False)):
            altered = copy.deepcopy(original)
            edit(altered)
            with self.assertRaises((ValueError, TypeError, KeyError)):
                rank.RankPlan._trial(altered)
        # Confirmed fixtures below exercise loading only. They are never
        # accepted acceleration evidence or exported as a usable performance plan.
        with tempfile.TemporaryDirectory(prefix="kd-rank-guard-") as directory:
            path = Path(directory)
            for name, content in rank._SOURCES:
                (path / name).write_bytes(content)
            (path / "manifest.json").write_text(json.dumps(original))
            self.assertTrue(rank.RankPlan(path).status.startswith("fallback:"))
            fixture = copy.deepcopy(original)
            fixture["evidence"]["confirmation_passed"] = True
            (path / "manifest.json").write_text(json.dumps(fixture))
            installed = rank.RankPlan(path)
            self.assertEqual(installed.status, "installed")
            rng = self.np.random.default_rng(3901)
            docs = self.mx.array(rng.normal(size=(9, 33)).astype(self.np.float32))
            query = self.mx.array(rng.normal(size=(3, 33)).astype(self.np.float32))
            self.assert_application(installed.index(docs), query, docs)
            self.assertEqual((installed.calls, installed.fallback_calls), (1, 0))
            (path / rank.RANK_SOURCE_FILES[0]).write_text("tampered")
            loaded = rank.RankPlan(path)
            self.assertTrue(loaded.status.startswith("fallback:"))
            self.assertEqual(loaded._rules, {})
            self.assert_application(loaded.index(docs), query, docs)
            self.assertEqual((loaded.calls, loaded.fallback_calls), (1, 1))

    def test_11_bounds_mode_and_autodiff_regression(self):
        mx, np, rank = self.mx, self.np, self.rank
        for shape in ((0, 5), (1, 4), (129, 5), (1, 2049)):
            with self.assertRaises(ValueError):
                rank.candidate_rank(mx.ones(shape), "simd32")
        with self.assertRaises(TypeError):
            rank.candidate_rank(mx.ones((1, 5), dtype=mx.float16), "simd32")
        with self.assertRaises(ValueError):
            rank.candidate_rank(mx.ones((1, 5)), "unknown")
        for mode in ("train", "autodiff"):
            with self.assertRaises(ValueError):
                rank.candidate_rank(mx.ones((1, 5)), "simd32", mode=mode)
            with self.assertRaises(ValueError):
                rank.RankPlan().index(mx.ones((5, 1)), mode=mode)
        for shape in ((128, 2048, 2048, 5), (3, 9, 33, 4)):
            with self.assertRaises(ValueError):
                rank.RankPlan._trial(self.manifest(shape=shape))
        self.assert_rank(mx.ones((128, 2048)), "simd32")
        # Deliberately unstable tie order and values detached from IDs must be
        # rejected by our independent oracle, rather than bless any fast output.
        host = np.ones((1, 9), np.float32)
        ids, bits = self.expected_rank(host)
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(ids[:, ::-1], ids)
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(bits ^ np.uint32(1), bits)

    def test_12_plan_layouts_and_extreme_application_inputs(self):
        mx, np, rank = self.mx, self.np, self.rank
        rng = np.random.default_rng(4101)
        host_docs = rng.normal(size=(37, 33)).astype(np.float32)
        host_query = rng.normal(size=(3, 33)).astype(np.float32)
        host_query[0] *= np.finfo(np.float32).max / (2 * np.max(np.abs(host_query[0])))
        host_query[1] *= 1e-30
        host_query[2] = 0
        for variant in rank.RANK_CANDIDATES:
            for layout in ("contiguous", "stride2", "negative_columns", "negative_rows", "transpose",
                           "stepped_rows", "offset_rows_columns", "broadcast_rows", "broadcast_columns"):
                plan = rank.RankPlan._trial(self.manifest(variant, (3, 37, 33, 5)))
                docs, query = self.view(host_docs, layout), self.view(host_query, layout)
                before_docs, before_query = np.array(docs).copy(), np.array(query).copy()
                self.assert_application(plan.index(docs), query, docs)
                self.assertEqual((plan.calls, plan.fallback_calls), (1, 0))
                np.testing.assert_array_equal(np.array(docs).view(np.uint32), before_docs.view(np.uint32))
                np.testing.assert_array_equal(np.array(query).view(np.uint32), before_query.view(np.uint32))
        self.assertLessEqual(sum(len(value) for _, value in rank._SOURCES), 12 * 1024)
