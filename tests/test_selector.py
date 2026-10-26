"""Synthetic fixtures test algorithm behavior, never GPU performance claims."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from kerneldarwin.selector import InterventionSelector, evaluate_grouped, bayesian_order, _gp_posterior


def fixtures():
    result = []
    for family in ("sum", "softmax", "layernorm"):
        for rows, columns in ((8, 17), (64, 129), (256, 511)):
            for block, reward in ((1, -1.), (256, .5), (4096, 1.)):
                result.append({"workload": {"family": family, "shape": [rows, columns]},
                               "candidate": {"block_size": block, "num_warps": 1},
                               "reward": reward, "cost": .01, "valid": True,
                               "group": f"fixture-{family}-{rows}", "device_kind": "cpu"})
    return result


class SelectionTests(unittest.TestCase):
    def test_training_changes_ranking_and_saved_model_is_reproducible(self):
        observations = fixtures()
        candidates = [item["candidate"] for item in observations[:3]]
        model = InterventionSelector(alpha=0).fit(observations)
        ranked = model.rank({"family": "sum", "shape": [64, 129], "device_kind": "cpu"}, candidates)
        self.assertEqual(ranked[0]["candidate"]["block_size"], 4096)
        inverse = [{**item, "reward": -item["reward"]} for item in observations]
        reversed_model = InterventionSelector(alpha=0).fit(inverse)
        self.assertEqual(reversed_model.rank({"family": "sum", "shape": [64, 129]}, candidates)[0]["candidate"]["block_size"], 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.save(path)
            restored = InterventionSelector.load(path)
            self.assertEqual(restored.rank({"family": "sum", "shape": [64, 129]}, candidates), ranked)
            data = json.loads(path.read_text())
            data["weights"][0] = float("nan")
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                InterventionSelector.load(path)

    def test_no_pretraining_and_device_mismatch(self):
        with self.assertRaises(ValueError):
            InterventionSelector().rank({"family": "sum", "shape": [1, 4]}, [])
        with self.assertRaises(ValueError):
            InterventionSelector().fit([])
        mixed = fixtures()
        mixed[-1]["device_kind"] = "cuda"
        with self.assertRaises(ValueError):
            InterventionSelector().fit(mixed)
        model = InterventionSelector().fit(fixtures())
        with self.assertRaises(ValueError):
            model.rank({"family": "sum", "shape": [1, 4], "device_kind": "cuda"}, [])

    def test_invalid_results_penalized(self):
        data = fixtures()
        for item in data:
            if item["candidate"]["block_size"] == 4096:
                item.update(reward=.8, valid=False)
        model = InterventionSelector(alpha=0, failure_penalty=3).fit(data)
        ranking = model.rank({"family": "sum", "shape": [64, 129]}, [item["candidate"] for item in data[:3]])
        self.assertNotEqual(ranking[0]["candidate"]["block_size"], 4096)

    def test_gaussian_process_posterior_and_sequential_reveal(self):
        mean, uncertainty = _gp_posterior([[0., 0.]], [2.], [0., 0.])
        _, distant_uncertainty = _gp_posterior([[0., 0.]], [2.], [1., 1.])
        self.assertAlmostEqual(mean, 2., places=3)
        self.assertLess(uncertainty, .011)
        self.assertGreater(distant_uncertainty, .99)
        candidates = [{"block_size": value, "num_warps": 1} for value in (1, 32, 256, 4096)]
        revealed = []
        def observe(candidate):
            self.assertNotIn(candidate, revealed)
            revealed.append(candidate)
            return float(candidate["block_size"] == 256)
        selected = bayesian_order(candidates, observe, seed=11)
        self.assertEqual(selected, revealed)
        self.assertEqual(len(selected), len(candidates))
        # Altering not-yet-seen rewards cannot alter the first selection or
        # the second selection if the initial observed reward stays fixed.
        first = selected[0]
        altered = bayesian_order(candidates, lambda candidate: float(candidate["block_size"] == 256) if candidate == first else -99., seed=11)
        self.assertEqual(selected[:2], altered[:2])
        with self.assertRaises(ValueError):
            bayesian_order([candidates[0], candidates[0]], observe)

    def test_grouped_replay_baselines_and_ablations(self):
        data = fixtures()
        report = evaluate_grouped(data, seed=7)
        self.assertEqual(report["device_kind"], "cpu")
        for name in ("family", "shape"):
            split = report["splits"][name]
            self.assertEqual(split["folds"], 3)
            rows = split["aggregates"]
            policies = {row["policy"] for row in rows}
            self.assertEqual(policies, {"learned_ucb", "random", "heuristic", "no_context", "no_uncertainty", "bayesian_ucb"})
            complete = [row for row in rows if row["budget"] == 3]
            self.assertTrue(all(abs(row["mean_oracle_gap"]) < 1e-12 for row in complete))
            self.assertTrue(all(row["mean_cost"] > 0 for row in rows))
        self.assertEqual(evaluate_grouped(data, seed=7), report)
        self.assertEqual(evaluate_grouped([])["status"], "insufficient_data")
        self.assertEqual(evaluate_grouped(data[:3])["splits"]["family"]["status"], "insufficient_data")


if __name__ == "__main__":
    unittest.main()
