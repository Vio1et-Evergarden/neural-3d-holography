import csv
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from plmholo.config import DEFAULTS, ROOT, merge, validate
from plmholo.propagation import ASM
from plmholo.quantization import PhaseLUT
from plmholo.schedule import Schedule
from plmholo.temporal import TemporalModel, integrate
from plmholo.losses import loss_value
from plmholo.motion import EyeMotion
from plmholo.targets import resample, load_target


def configuration(**updates):
    c = merge(DEFAULTS, {"threads": 1,
        "optics": {"shape": [9, 12], "colors": ["G"], "wavelengths_m": [517.7e-9]},
        "schedule": {"cycles": 4, "order": ["G"], "gains": [1.0], "window_cycles": 2},
        "optimization": {"iterations": 3}, "output": str(ROOT/".cache"/"test-output")})
    return validate(merge(c, updates))


class PropagationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(4)

    def test_zero_distance_padding_odd_rectangle_and_crop(self):
        c = configuration(optics={"depths_m": [0.0], "crop": [7, 8]})
        p = torch.randn(2, 9, 12, dtype=torch.float64)
        field = ASM(c["optics"])(p, [0, 0])
        torch.testing.assert_close(field[:, 0], torch.exp(1j*p[:, 1:8, 2:10]), rtol=1e-12, atol=1e-12)

    def test_plane_wave_phase_and_power(self):
        c = configuration(optics={"shape": [12, 16], "crop": [12, 16],
                                 "computational_shape": [12, 16], "depths_m": [0.0044]})
        y, x = torch.meshgrid(torch.arange(12, dtype=torch.float64), torch.arange(16, dtype=torch.float64), indexing="ij")
        phase = (2*math.pi*(2*x/16+y/12))[None]
        prop = ASM(c["optics"])
        result = prop(phase, [0])[0, 0]
        factor = prop.kernel(0, phase.device, phase.dtype)[0, 1, 2]
        torch.testing.assert_close(result, torch.exp(1j*phase[0])*factor, rtol=1e-11, atol=1e-11)
        self.assertAlmostEqual(result.abs().square().sum().item(), 192, places=9)

    def test_wavelength_depth_separation_and_cache(self):
        c = configuration(optics={"colors": ["R", "G", "B"], "wavelengths_m": [640e-9, 516e-9, 455.5e-9]},
                          schedule={"order": ["G", "R", "B"], "gains": [3, 3, 3]})
        p = torch.randn(1, 9, 12).expand(3, -1, -1)
        prop = ASM(c["optics"])
        result = prop(p, [0, 1, 2])
        self.assertEqual(tuple(result.shape), (3, 3, 9, 12))
        self.assertGreater((result[0].abs()-result[1].abs()).abs().mean().item(), 1e-3)
        for color in range(3):
            torch.testing.assert_close(result[color:color+1], prop(p[:1], [color]))
        self.assertEqual(len(prop._cache), 3)

    def test_gradient_finite_difference(self):
        c = configuration()
        prop = ASM(c["optics"])
        p = torch.randn(1, 9, 12, dtype=torch.float64, requires_grad=True)
        target = torch.rand(1, 3, 9, 12, dtype=torch.float64)
        direction = torch.randn_like(p)
        def loss(value):
            return (prop(value, [0]).abs().square()-target).square().mean()
        gradient, = torch.autograd.grad(loss(p), p)
        epsilon = 1e-5
        numerical = (loss(p.detach()+epsilon*direction)-loss(p.detach()-epsilon*direction))/(2*epsilon)
        torch.testing.assert_close(numerical, (gradient*direction).sum(), rtol=1e-6, atol=1e-9)

    def test_evanescent_exclusion_and_na(self):
        c = configuration(optics={"pitch_m": [1e-7, 1e-7], "numerical_aperture": 0.2})
        prop = ASM(c["optics"])
        h = prop.kernel(0, "cpu", torch.float32)
        self.assertTrue(torch.isfinite(h).all())
        self.assertTrue((h.abs() == 0).any())
        self.assertTrue((h.abs() > 0.9).any())


class TemporalTests(unittest.TestCase):
    def test_weighted_prefix_windows_no_wrap(self):
        values = torch.tensor([1., 4., 8., 2.]).reshape(4, 1, 1, 1, 1)
        dt = torch.tensor([1., 2., 1., 3.])
        actual = integrate(values, dt, 2).flatten()
        torch.testing.assert_close(actual, torch.tensor([1., 3., 16/3, 3.5]))

    def test_rgb_duty_compensation_and_global_gradient(self):
        c = validate(merge(DEFAULTS, {"optics": {"shape": [2, 2]}, "schedule": {"cycles": 3, "window_cycles": 2}}))
        schedule = Schedule.from_config(c)
        raw = torch.ones(9, 3, 2, 2, requires_grad=True)
        predicted, _ = TemporalModel(schedule).reconstruct(raw, torch.arange(9))
        torch.testing.assert_close(predicted[5:], torch.ones_like(predicted[5:]))
        predicted.square().mean().backward()
        self.assertTrue((raw.grad.abs().sum((1, 2, 3)) > 0).all())

    def test_pi_shift_does_not_cancel_temporal_intensity(self):
        c = configuration()
        phase = torch.rand(1, 9, 12)*6.28
        field = ASM(c["optics"])(torch.cat([phase, phase+math.pi]), [0, 0])
        torch.testing.assert_close(field[0].abs().square(), field[1].abs().square(), atol=2e-6, rtol=2e-5)
        self.assertLess(field.sum(0).abs().square().max().item(), 1e-10)

    def test_motion_zero_identity_translation_and_mask(self):
        video = torch.zeros(2, 1, 1, 5, 7)
        video[:, 0, 0, 2, 2] = 1
        dt = torch.ones(2)
        motion = EyeMotion(torch.zeros(2, 2))
        result, mask = motion.perceive(video, dt, 2, torch.arange(2))
        self.assertTrue(torch.equal(result, integrate(video, dt, 2)))
        self.assertTrue((mask == 1).all())
        motion = EyeMotion(torch.tensor([[0., 0.], [1., 0.]]))
        shifted, valid = motion.warp(video, torch.arange(2))
        self.assertAlmostEqual(shifted[0, 0, 0, 2, 3].item(), 1, places=6)
        self.assertEqual(valid[0, 0, 2, 0].item(), 0)

    def test_equal_timestamp_weight_despite_mask_areas(self):
        pred = torch.tensor([1., 3.])[:, None, None, None, None].expand(2, 1, 1, 2, 2)
        mask = torch.ones_like(pred)
        mask[1, :, :, 0, :] = 0
        self.assertAlmostEqual(loss_value(pred, torch.zeros_like(pred), mask).item(), 5.0)


class QuantizationTests(unittest.TestCase):
    def test_circular_wrap_and_physical_codes(self):
        phases = torch.arange(16, dtype=torch.float64)*2*math.pi/16
        # Table order and physical IDs deliberately do not follow phase order.
        lut = PhaseLUT(["G"], phases=phases.flip(0)[None], codes=torch.arange(16)[None])
        p = torch.tensor([[[-0.01, 2*math.pi+0.01, math.pi]]])
        quantized, codes = lut.hard(p, [0])
        self.assertEqual(codes.tolist(), [[[15, 15, 7]]])
        torch.testing.assert_close(lut.decode(codes, [0]), quantized)

    def test_nonuniform_csv_surrogates_are_hard_forward(self):
        cache = ROOT/".cache"
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as tmp:
            path = Path(tmp)/"lut.csv"
            with path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["color", "code", "phase_rad"])
                for code in range(16):
                    writer.writerow(["G", code, (code/16)**1.3*2*math.pi])
            lut = PhaseLUT(["G"], path)
            for method in ("ste", "smooth"):
                p = torch.tensor([[[0.12, 0.35, 1.2, 5.4]]], requires_grad=True)
                hard, _ = lut.hard(p, [0])
                forward = lut.forward(p, [0], method, tau=.15)
                self.assertTrue(torch.equal(forward, hard))
                forward.sin().sum().backward()
                self.assertTrue(torch.isfinite(p.grad).all())
                self.assertGreater(p.grad.abs().sum().item(), 0)

    def test_time_interpolation_and_static_broadcast(self):
        v = torch.tensor([[0.], [10.]])
        actual = resample(v, [0., 1.], torch.tensor([-.1, .25, 2.], dtype=torch.float64))
        torch.testing.assert_close(actual[:, 0], torch.tensor([0., 2.5, 10.]))
        torch.testing.assert_close(resample(v[:1], None, torch.arange(3)), torch.zeros(3, 1))


if __name__ == "__main__":
    unittest.main()
