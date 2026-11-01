"""Adapter/entry-point tests without actual DEEPX hardware or a Torch install."""
import argparse
import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dxm1_parseq_backend import recover_rgb
import alpr_auto_dxm1 as entry


class AdapterTests(unittest.TestCase):
    def test_normalized_rgb_round_trip_for_all_byte_values(self):
        raw = (np.arange(32*128*3) % 256).astype(np.uint8).reshape(32,128,3)
        tensor = (raw.transpose(2,0,1)[None].astype(np.float32)/255 - .5)/.5
        np.testing.assert_array_equal(recover_rgb(tensor), raw)

    def test_bad_transform_and_nonfinite_input_rejected(self):
        for value in [np.full((1,3,32,128), np.nan, np.float32),
                      np.full((1,3,32,128), 2, np.float32),
                      np.full((1,3,32,128), .12345, np.float32),
                      np.zeros((2,3,32,128), np.float32)]:
            with self.assertRaises(ValueError): recover_rgb(value)

    def fixture(self):
        base = types.ModuleType("alpr_auto")
        class Reader:
            def __init__(self, device="cpu", checkpoint="pretrained=parseq"):
                self.device = device
                self.model = types.SimpleNamespace(encode=lambda: "cpu")
            def read(self, crop, layout="auto"):
                return self.model.encode(), layout
        base.ParseqReader = Reader
        instances = []
        class Backend:
            def __init__(self, model, run, device):
                self.model, self.original = model, model.encode
                self.device, self.closed = device, False
                model.encode = lambda: "dxm1"
                instances.append(self)
            def report(self):
                return {"decoder_device": self.device, "successful_encoder_calls": 1,
                        "failed_encoder_calls": 0, "placement":{"npu_0":"NPU"}}
            def close(self):
                self.model.encode = self.original
                self.closed = True
        return base, Reader, Backend, instances

    def test_inherited_reader_and_cleanup_on_exception(self):
        base, original, backend, instances = self.fixture()
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with entry.reader_override(base, "run", backend):
                self.assertIs(base.ParseqReader.read, original.read)
                reader = base.ParseqReader()
                self.assertEqual(reader.read(None, "two"), ("dxm1", "two"))
                raise RuntimeError("fixture")
        self.assertIs(base.ParseqReader, original)
        self.assertTrue(instances[0].closed)
        self.assertEqual(instances[0].model.encode(), "cpu")

    def test_entrypoint_preserves_csv_and_annotates_actual_backend(self):
        base, original, backend, instances = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            run.mkdir()
            (run / "matmul_contract.json").write_text("{}")
            base_file = root / "alpr_auto.py"
            base_file.write_text("# stub baseline for interface test")
            base.__file__ = str(base_file)
            destination = root / "output"
            def main(argv):
                p = argparse.ArgumentParser()
                p.add_argument("--source");p.add_argument("--device",default="cpu");p.add_argument("--output")
                args = p.parse_args(argv)
                reader = base.ParseqReader(args.device)
                self.assertEqual(reader.read(None)[0], "dxm1")
                output = Path(args.output)
                output.mkdir()
                (output / "detections.csv").write_text("text\n30A33918\n")
                (output / "summary.json").write_text(json.dumps({"scope":"PC baseline", "status":"complete", "accuracy":None, "ocr_device":args.device}))
                return 0
            base.main = main
            with patch.dict(sys.modules, {"alpr_auto":base}), patch.object(entry, "Dxm1Encoder", backend), \
                    contextlib.redirect_stdout(io.StringIO()):
                code = entry.main(["--dx-run",str(run),"--source","image.jpg","--output",str(destination)])
            self.assertEqual(code,0)
            self.assertEqual((destination / "detections.csv").read_text(),"text\n30A33918\n")
            report = json.loads((destination / "summary.json").read_text())
            self.assertEqual(report["ocr_encoder_backend"],"DX-M1")
            self.assertEqual(report["ocr_decoder_device"],"cpu")
            self.assertIsNone(report["accuracy"])
            self.assertEqual(report["status"],"complete")
            self.assertIn("alpr_base_sha256",report)
        self.assertIs(base.ParseqReader,original)
        self.assertTrue(instances[0].closed)


if __name__ == "__main__":
    unittest.main()
