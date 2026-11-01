"""DX-M1 full-encoder adapter for normalized PARSeq image tensors (batch=1)."""
from pathlib import Path
from importlib.metadata import version

import numpy as np

from compare_dxm1_matmul_ocr import load_run, check_runtime
from probe_parseq_dxm1 import ort_session
from probe_dxm1_matmul_head import pack_pixels


def recover_rgb(images):
    """Invert the standard PIL ToTensor/Normalize(0.5,0.5) transform exactly."""
    if images.shape != (1, 3, 32, 128) or not np.isfinite(images).all():
        raise ValueError("Expected finite normalized PARSeq tensor [1,3,32,128]")
    raw = (images[0].transpose(1, 2, 0) * np.float32(0.5) + np.float32(0.5)) * np.float32(255)
    rounded = np.rint(raw)
    if raw.min() < -1e-3 or raw.max() > 255.001 or np.max(np.abs(raw - rounded)) > 1e-3:
        raise ValueError("DX adapter requires the standard normalized uint8-image transform")
    return np.ascontiguousarray(np.clip(rounded, 0, 255), dtype=np.uint8)


class Dxm1Encoder:
    def __init__(self, model, run, decoder_device="cpu"):
        import torch
        from dx_engine import InferenceEngine
        from dx_engine.inference_option import InferenceOption
        self.torch, self.model = torch, model
        self.core = model.model
        self.original_encode = self.core.encode
        self.engine = None
        self.closed = False
        self.calls, self.failures = 0, 0
        if list(model.hparams.img_size) != [32,128]:
            raise ValueError("DX-M1 adapter supports standard PARSeq 32x128 only")
        manifest, dxnn, original = load_run(run)
        reference = ort_session(original)
        # Check checkpoint identity before replacing the encoder. The local
        # generator avoids changing the application's global random state.
        generator = torch.Generator(device="cpu").manual_seed(20261101)
        with torch.inference_mode():
            probes = [torch.zeros((1,3,32,128), dtype=torch.float32),
                      torch.rand((1,3,32,128), generator=generator, dtype=torch.float32) * 2 - 1]
            for sample in probes:
                actual = self.original_encode(sample.to(decoder_device)).detach().float().cpu().numpy()
                expected = reference.run(None, {reference.get_inputs()[0].name: sample.numpy()})[0]
                if (actual.shape != expected.shape or not np.isfinite(actual).all()
                        or not np.isfinite(expected).all()
                        or not np.allclose(actual, expected, rtol=1e-3, atol=1e-4)):
                    raise ValueError("ALPR checkpoint does not match this compiled encoder")
        del reference, probes
        options = InferenceOption()
        options.use_ort = True
        try:
            self.engine = InferenceEngine(str(dxnn), options)
            if not options.use_ort:
                raise RuntimeError("DX-RT disabled ORT; CPU tasks could be skipped")
            shape, info = check_runtime(self.engine)
            self.feed = np.empty(shape, dtype=np.float32)
            self.metadata = {
                "encoder_backend": "DX-M1", "decoder_device": decoder_device,
                "run": str(Path(run).resolve()), "dxnn": str(dxnn),
                "dxnn_sha256": manifest["dxnn_sha256"],
                "original_encoder_sha256": manifest["original_sha256"],
                "placement": manifest["placement"], "runtime_io": info,
                "dx_engine": version("dx-engine"),
                "checkpoint_check": "matched on zero and seeded random normalized inputs",
                "cpu_preprocessing": "existing RGB/resize transform + raw patch packing",
                "npu_calls_include_warmup": True,
            }
            self.core.encode = self.encode
        except BaseException:
            if self.engine is not None:
                self.engine.dispose()
            raise

    def encode(self, images):
        if self.closed:
            raise RuntimeError("DX-M1 encoder is closed")
        try:
            normalized = np.ascontiguousarray(images.detach().float().cpu().numpy(), dtype=np.float32)
            pixels = recover_rgb(normalized)
            np.copyto(self.feed.reshape(1,128,96), pack_pixels(pixels))
            outputs = self.engine.run([self.feed])
            if len(outputs) != 1 or outputs[0].dtype != np.float32:
                raise ValueError("Unexpected NPU output count/dtype")
            memory = np.array(outputs[0], copy=True)
            if memory.size != 128*384 or memory.shape[-2:] != (128,384) or not np.isfinite(memory).all():
                raise ValueError(f"Invalid NPU encoder memory: {memory.shape}")
            result = self.torch.from_numpy(memory.reshape(1,128,384)).to(
                device=images.device, dtype=images.dtype)
            self.calls += 1
            return result
        except Exception:
            self.failures += 1
            raise

    def report(self):
        return {**self.metadata, "successful_encoder_calls": self.calls,
                "failed_encoder_calls": self.failures}

    def close(self):
        if not self.closed:
            self.core.encode = self.original_encode
            self.closed = True
            if self.engine is not None:
                self.engine.dispose()
