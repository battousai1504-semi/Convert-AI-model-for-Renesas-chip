"""Evidence collection rejects conflicting provenance and empty or mismatched observations."""
import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('dxm1_evidence',
    Path(__file__).resolve().parents[1] / 'scripts/collect_dxm1_evidence.py')
evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ['status'])
        writer.writeheader()
        writer.writerows(rows)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        run = self.root / 'run'
        run.mkdir()
        for name in ('model.dxnn', 'original_reference.onnx', 'parseq_rawpatch_matmul.onnx'):
            (run / name).write_bytes(name.encode())
        self.hash = evidence.digest(run / 'model.dxnn')
        self.original = evidence.digest(run / 'original_reference.onnx')
        placement = {'projection_nodes': ['project'], 'partitions': {'npu_0': 'NPU'}}
        write_json(run / 'matmul_contract.json', {'status':'compiled', 'dxnn':'model.dxnn',
            'dxnn_sha256':self.hash, 'original_sha256':self.original,
            'matmul_sha256':evidence.digest(run/'parseq_rawpatch_matmul.onnx'), 'placement':placement})
        ocr = self.root / 'ocr'
        write_json(ocr / 'summary.json', {'dxnn_sha256':self.hash, 'original_sha256':self.original,
            'images':6, 'text_matches':6, 'errors':0, 'all_texts_match':True})
        write_csv(ocr/'comparison.csv', [{'status':'match','pytorch_text':str(i),
            'dxm1_hybrid_text':str(i)} for i in range(6)])
        for name, count, text, calls in [('photo',1,'30A33918',2), ('video',30,'59V184625',61)]:
            root = self.root / name
            texts = [text] * count
            write_json(root/'smoke_comparison.json', {'detection_text_and_box_agreement':True,
                'dxm1_ocr_errors':0,'cpu_detection_texts':texts,'dxm1_detection_texts':texts,
                'successful_encoder_calls_including_warmup':calls,
                'expected_single_image_text':text if name=='photo' else None,
                'expected_text_seen_cpu':name=='photo','expected_text_seen_dxm1':name=='photo'})
            for backend in ('cpu','dxm1'):
                folder = root / backend
                write_json(folder/'summary.json', {'status':'complete', 'processed_frames':count,
                    'detections':count,'events':1,'ocr_errors':0,'ocr_encoder_backend':'DX-M1',
                    'ocr_decoder_device':'cpu','deepx_encoder':{'dxnn_sha256':self.hash,
                    'original_encoder_sha256':self.original, 'successful_encoder_calls':calls,
                    'failed_encoder_calls':0}})
                write_csv(folder/'detections.csv', [{'source':name,'frame_index':str(i),
                    'x1':'1','y1':'2','x2':'3','y2':'4','text':text,'status':'ok'} for i in range(count)])
                write_csv(folder/'events.csv', [{'text':text}])
                write_csv(folder/'frames.csv', [{'status':'ok'} for _ in range(count)])

    def collect(self):
        return evidence.collect(self.root, 'run', 'ocr', 'photo', 'video')

    def test_collects_counts_and_hashes_without_inventing_accuracy_or_fps(self):
        result = self.collect()
        self.assertEqual(result['observations']['video']['text_counts_each_backend'], {'59V184625':30})
        self.assertEqual(result['dxnn_sha256'], self.hash)
        self.assertIsNone(result['ground_truth_test_accuracy'])
        self.assertIsNone(result['end_to_end_fps'])
        self.assertEqual(len(result['source_report_sha256']),21)
        self.assertNotIn(str(self.root), json.dumps(result))

    def test_rejects_other_model_even_if_smoke_claims_pass(self):
        path = self.root/'video/dxm1/summary.json'
        report = evidence.read_json(path)
        report['deepx_encoder']['dxnn_sha256'] = '0'*64
        write_json(path, report)
        with self.assertRaisesRegex(ValueError, 'hash differs'):
            self.collect()

    def test_rejects_empty_csv_even_if_summary_claims_pass(self):
        write_csv(self.root/'video/dxm1/detections.csv', [])
        with self.assertRaisesRegex(ValueError, 'CSV counts'):
            self.collect()

    def test_rechecks_actual_boxes_instead_of_trusting_agreement_flag(self):
        path = self.root/'video/dxm1/detections.csv'
        rows = evidence.read_rows(path)
        rows[0]['x1'] = '99'
        write_csv(path, rows)
        with self.assertRaisesRegex(ValueError, 'boxes/text differ'):
            self.collect()


if __name__ == '__main__':
    unittest.main()
