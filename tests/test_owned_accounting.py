"""Synthetic CPU privacy/parser checks, never GPU or performance evidence."""
import importlib.util
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "evidence/workload-2026-11-01/owned-shader-diagnostic/account_resources.py"
SPEC = importlib.util.spec_from_file_location("owned_resource_accounting", SOURCE)
accounting = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(accounting)


def fixture(rows):
    return ('<trace-query-result><node><schema name="fixture">'
            '<col><mnemonic>start</mnemonic></col>'
            '<col><mnemonic>process</mnemonic></col>'
            '<col><mnemonic>label</mnemonic></col></schema>' + rows +
            '</node></trace-query-result>').encode()


class OwnedAccountingTests(unittest.TestCase):
    def test_foreign_labels_and_different_row_width_are_never_emitted(self):
        content = fixture(
            '<row><time>1</time><process id="other"><pid>9</pid></process>'
            '<label>foreign-private-label</label><extra>foreign-backtrace</extra></row>'
            '<row><time>2</time><process id="own"><pid>47528</pid></process><label>owned</label></row>'
            '<row><time>3</time><process ref="other"/><label>foreign-ref</label></row>'
            '<row><time>4</time><process ref="own"/><label>owned-ref</label></row>')
        result = list(accounting.owned_rows(content, "fixture", ("start", "process", "label"), 47528))
        self.assertEqual([text(cells["label"]) for cells, text, _ in result], ["owned", "owned-ref"])
        self.assertEqual([accounting.integer(cells, text, "start") for cells, text, _ in result], [2, 4])

    def test_owned_width_and_schema_changes_fail_instead_of_partial_replay(self):
        content = fixture('<row><time>1</time><process><pid>47528</pid></process>'
                          '<label>owned</label><extra>unexpected</extra></row>')
        with self.assertRaises(ValueError):
            list(accounting.owned_rows(content, "fixture", ("start", "process", "label"), 47528))
        for schema, columns in (("other", ("start", "process", "label")),
                                ("fixture", ("start", "process", "changed"))):
            with self.subTest(schema=schema, columns=columns), self.assertRaises(ValueError):
                list(accounting.owned_rows(content, schema, columns, 47528))

    def test_missing_time_is_unknown_and_negative_sizes_are_rejected(self):
        cells = {"duration": "", "size": "-1", "zero": "0"}
        self.assertIsNone(accounting.integer(cells, str, "duration", optional=True))
        self.assertEqual(accounting.integer(cells, str, "zero"), 0)
        for name in ("duration", "size"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                accounting.integer(cells, str, name)

    def test_reads_are_bounded_before_open_and_detect_size_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.xml"
            path.write_bytes(b"abc")
            self.assertEqual(accounting.bounded_bytes(path), b"abc")
            with patch.object(Path, "stat", return_value=SimpleNamespace(st_size=20_000_001)), \
                    patch.object(Path, "open", side_effect=AssertionError("must reject before opening")):
                with self.assertRaises(ValueError):
                    accounting.bounded_bytes(path)
            with patch.object(Path, "stat", return_value=SimpleNamespace(st_size=3)), \
                    patch.object(Path, "open", return_value=io.BytesIO(b"abcd")):
                with self.assertRaises(ValueError):
                    accounting.bounded_bytes(path)
