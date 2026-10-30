#!/usr/bin/env python3
"""Opt-in, bounded compatibility probe. No timing, tuning, or artifact install.

Run only after other GPU work stops, using the intended MLX interpreter:
    PYTHONPATH=src python .work/compiled-score-probe.py --output NEW_DIRECTORY

Imports and argument parsing do not execute this probe. Only main() does.
The three graph outputs match _search_results. Requesting k=N lets the probe
recover every original score by ID without adding a diagnostic graph output.
The synthetic in-memory rules below are fixtures, never confirmed plans.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path


@lru_cache(maxsize=32)
def whole_query(variant, k):
    import mlx.core as mx
    from kerneldarwin.fused_search import candidate_scores

    def graph(q, docs):
        scores = candidate_scores(q, docs, variant)
        finite = mx.all(mx.isfinite(scores))
        ids = mx.argsort(-scores, axis=-1)[:, :k]
        return ids, mx.take_along_axis(scores, ids, axis=-1), finite

    return mx.compile(graph)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    # Delay every MLX/NumPy import and device operation until explicit execution.
    import mlx.core as mx
    import numpy as np
    from kerneldarwin import fused_search as fused
    from kerneldarwin.search import _compiled_search, _search_inputs, _search_results

    mx.set_default_device(mx.gpu)
    mx.set_memory_limit(2 * 1024**3)
    mx.set_cache_limit(256 * 1024**2)
    report = {
        'kind': 'compiled_authored_score_compatibility_probe',
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'score_source_identity': fused.score_source_identity(),
        'loaded_fused_module': str(Path(fused.__file__).resolve()),
        'limitation': 'Compatibility and numerical checks only; no timing, tuning, candidate selection, or performance claim.',
        'checks': [], 'graphs': {},
    }

    def save():
        (args.output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')

    def reference(q, docs):
        q64 = np.asarray(q).astype(np.float64)
        q64[np.abs(q64) < np.finfo(np.float32).tiny] = 0
        peak = np.maximum(np.max(np.abs(q64), axis=1, keepdims=True), 1e-12)
        scaled = q64 / peak
        normed = scaled / np.maximum(np.sqrt(np.sum(scaled * scaled, axis=1, keepdims=True)), 1e-12 / peak)
        return normed @ np.asarray(docs).astype(np.float64).T

    def normalized_docs(n, d, seed):
        raw = np.random.default_rng(seed).normal(size=(n, d))
        raw /= np.linalg.norm(raw, axis=1, keepdims=True)
        return mx.array(raw.astype(np.float32))

    def check(label, q, docs, variant, *, atol=2e-6, rtol=2e-5):
        before_q, before_docs = np.array(q).copy(), np.array(docs).copy()
        expected = reference(before_q, before_docs)
        n = docs.shape[0]
        result = whole_query(variant, n)(q, docs)
        # Export the current lazy graph before evaluation can discard its edges.
        if label == 'N5_D33/contiguous/contiguous':
            graph_path = args.output / (variant + '.dot')
            if hasattr(mx, 'export_to_dot'):
                try:
                    mx.export_to_dot(str(graph_path), *result, queries=q, index=docs)
                    report['graphs'][variant] = {
                        'file': graph_path.name,
                        'sha256': hashlib.sha256(graph_path.read_bytes()).hexdigest(),
                        'primitive_rectangles': graph_path.read_text().count('shape=rectangle'),
                    }
                except Exception as exc:
                    report['graphs'][variant] = {'export_error': type(exc).__name__ + ': ' + str(exc)}
            else:
                report['graphs'][variant] = {'unsupported': 'mx.export_to_dot unavailable'}
        mx.eval(*result)
        ids, values, finite = result
        assert finite.item(), label
        hi, hv = np.array(ids), np.array(values)
        recovered = np.empty_like(hv)
        np.put_along_axis(recovered, hi, hv, axis=-1)
        np.testing.assert_allclose(recovered, expected, atol=atol, rtol=rtol)
        np.testing.assert_allclose(hv, np.take_along_axis(expected, hi, axis=-1), atol=atol, rtol=rtol)
        np.testing.assert_allclose(hv, np.sort(expected, axis=-1)[:, ::-1], atol=atol, rtol=rtol)
        np.testing.assert_array_equal(np.sort(hi, axis=-1), np.broadcast_to(np.arange(n), hi.shape))
        # Float64 defines strict orders; numerically indistinguishable scores
        # retain the existing MLX tie policy, checked against the prior path.
        strict_rows = 0
        for r in range(len(hi)):
            ordered = np.sort(expected[r])[::-1]
            tolerance = atol + rtol * np.max(np.abs(expected[r]))
            if np.all(ordered[:-1] - ordered[1:] > 2 * tolerance):
                np.testing.assert_array_equal(hi[r], np.argsort(-expected[r], kind='stable'))
                strict_rows += 1
        old = fused._rank_scores(n)(fused.candidate_scores(q, docs, variant))
        mx.eval(*old)
        np.testing.assert_array_equal(hi, np.array(old[0]))
        np.testing.assert_allclose(hv, np.array(old[1]), atol=atol, rtol=rtol)
        # Exercise a real top-k slice separately from the k=N score recovery.
        k = min(3, n)
        selected = whole_query(variant, k)(q, docs)
        mx.eval(*selected)
        np.testing.assert_array_equal(np.array(selected[0]), hi[:, :k])
        np.testing.assert_allclose(np.array(selected[1]), hv[:, :k], atol=atol, rtol=rtol)
        assert selected[2].item()
        np.testing.assert_array_equal(np.array(q), before_q)
        np.testing.assert_array_equal(np.array(docs), before_docs)
        report['checks'].append({'case': label, 'variant': variant, 'shape': [q.shape[0], n, q.shape[1]], 'strict_float64_id_rows': strict_rows, 'passed': True})
        return result

    class ProbeIndex(fused.FusedCosineIndex):
        """Local proposed integration; production files are never patched."""

        def search(self, queries, k=5):
            _search_inputs(queries, self.vectors, k)
            if queries.shape[0] == 0:
                return [], []
            if isinstance(self.scorer, fused.FusedPlan):
                candidate = self.scorer._route(queries, self.vectors)
                if candidate is None:
                    return _search_results(_compiled_search('stable', k)(queries, self.vectors))
                return _search_results(whole_query(candidate, k)(queries, self.vectors))
            scores = self.scorer(queries, self.vectors)
            return _search_results(fused._rank_scores(k)(scores))

    try:
        rng = np.random.default_rng(412)
        qraw = rng.normal(size=(4, 33)).astype(np.float32)
        docs = normalized_docs(5, 33, 413)
        draw = np.array(docs)
        qp = np.zeros((4, 67), np.float32); qp[:, 1::2] = qraw
        dp = np.zeros((5, 67), np.float32); dp[:, 1::2] = draw
        qviews = {
            'contiguous': mx.array(qraw), 'transpose': mx.array(qraw.T.copy()).T,
            'offset_stride': mx.array(qp)[:, 1::2], 'negative_cols': mx.array(qraw)[:, ::-1],
            'negative_rows': mx.array(qraw)[::-1],
            'broadcast': mx.broadcast_to(mx.array(qraw[:1]), (4, 33)),
        }
        dviews = {
            'contiguous': docs, 'transpose': mx.array(draw.T.copy()).T,
            'offset_stride': mx.array(dp)[:, 1::2], 'negative_cols': docs[:, ::-1],
            'negative_rows': docs[::-1], 'broadcast': mx.broadcast_to(docs[:1], (5, 33)),
        }
        for variant in fused.SCORE_CANDIDATES:
            for qname, q in qviews.items():
                for dname, d in dviews.items():
                    check('N5_D33/' + qname + '/' + dname, q, d, variant)
            # Same tuple, fresh query AND document contents, cached callable.
            for seed in (414, 415):
                changed_q = mx.array(np.random.default_rng(seed).normal(size=(4, 33)).astype(np.float32))
                check('changed_inputs_' + str(seed), changed_q, normalized_docs(5, 33, seed + 10), variant)
            check('row_offsets', mx.array(qraw)[1:3], docs[1:5], variant)
            extremes = qraw.copy()
            extremes[0] *= np.finfo(np.float32).max / (2 * np.max(np.abs(extremes[0])))
            extremes[1] *= 1e-30
            extremes[2] = 0
            extremes[3] = np.where(np.arange(33) % 2, -1., 1.)
            check('huge_tiny_zero_cancellation', mx.array(extremes), docs, variant)
            check('uniform_tiny', mx.full((4, 33), 1e-35), mx.full((5, 33), 1 / np.sqrt(33)), variant, atol=0, rtol=3e-5)
            check('subnormal_flush', mx.full((4, 33), float(np.nextafter(np.float32(0), np.float32(1)))), docs, variant, atol=0, rtol=0)
            bad = np.ones((4, 33), np.float32)
            bad[0, 3] = np.nan; bad[1, 5] = np.inf; bad[2, 1] = -np.inf
            invalid = whole_query(variant, 5)(mx.array(bad), docs)
            mx.eval(*invalid)
            assert not invalid[2].item()
            assert np.isnan(np.array(invalid[1])[:3]).all()
            assert np.isfinite(np.array(invalid[1])[3]).all()
            try:
                _search_results(invalid)
            except ValueError as exc:
                assert str(exc) == 'Search query vectors must be finite'
            else:
                raise AssertionError('nonfinite query was accepted')
            report['checks'].append({'case': 'nonfinite_graph_and_host_rejection', 'variant': variant, 'passed': True})

            # Only a synthetic in-memory routing fixture, never a public plan.
            plan = fused.FusedPlan()
            plan._rules[(4, 5, 33)] = variant
            plan.status = 'synthetic_probe_fixture'
            index = ProbeIndex(docs, plan)
            old_index = fused.FusedCosineIndex(docs, lambda q, d: fused.candidate_scores(q, d, variant))
            for q in (mx.array(qraw), mx.array(qraw * -1.7 + .13)):
                before = plan.calls
                actual, actual_values = index.search(q, 3)
                assert plan.calls == before + 1 and plan.fallback_calls == 0
                prior, prior_values = old_index.search(q, 3)
                assert actual == prior
                np.testing.assert_allclose(actual_values, prior_values, atol=2e-6, rtol=2e-5)
                expected = reference(q, index.vectors)
                np.testing.assert_allclose(actual_values, np.take_along_axis(expected, np.array(actual), axis=-1), atol=2e-6, rtol=2e-5)
            before = plan.calls
            assert index.search(mx.zeros((0, 33)), 3) == ([], []) and plan.calls == before
            for q in (mx.array(qraw).astype(mx.float16), mx.array(qraw[:2])):
                before = plan.calls
                actual = index.search(q, 3)
                expected = _search_results(_compiled_search('stable', 3)(q, index.vectors))
                assert actual == expected and plan.calls == before + 1
            assert plan.fallback_calls == 2
            before = plan.calls
            try:
                index.search(mx.array(bad), 3)
            except ValueError as exc:
                assert str(exc) == 'Search query vectors must be finite'
            else:
                raise AssertionError('integrated nonfinite query was accepted')
            assert plan.calls == before + 1
            before_docs = np.array(index.vectors).copy()
            try:
                index.refresh(mx.full((5, 33), float('nan')))
            except ValueError:
                pass
            else:
                raise AssertionError('invalid document refresh was accepted')
            np.testing.assert_array_equal(np.array(index.vectors), before_docs)
            report['checks'].append({'case': 'local_index_guards_changed_inputs_fallback_empty_nonfinite_rollback', 'variant': variant, 'passed': True})
        report['compiled_callable_cache'] = whole_query.cache_info()._asdict()
        report['passed'] = True
        save()
        print(json.dumps({'passed': True, 'checks': len(report['checks']), 'output': str(args.output.resolve()), 'timed': False}))
    except Exception as exc:
        report['passed'] = False
        report['error'] = type(exc).__name__ + ': ' + str(exc)
        save()
        raise


if __name__ == '__main__':
    main()
