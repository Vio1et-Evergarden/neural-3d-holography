"""Stride changes optimization sampling/commits, never physical time integration."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from plmholo.config import DEFAULTS, ROOT, merge, validate
from plmholo.schedule import Schedule
from plmholo.temporal import TemporalModel
from plmholo.optimization import prepare, optimize
from plmholo.losses import loss_value
from plmholo.artifacts import save_run
from plmholo.__main__ import benchmark


def config(method="causal", quant="smooth", motion=False):
    return validate(merge(DEFAULTS, {
        "threads": 1, "optics": {"shape": [6, 8]},
        "schedule": {"cycles": 7, "window_cycles": 4, "stride_cycles": 2,
                     "durations_s": [.001, .002, .0015]*7},
        "optimization": {"method": method, "iterations": 2, "log_every": 0},
        "quantization": {"method": quant},
        "motion": {"enabled": motion, "velocity_px_s": [15, 0]},
    }))


class StrideTests(unittest.TestCase):
    def test_w8_s4_rgb_and_mono_windows(self):
        for mono in (False, True):
            c = config()
            c["schedule"].update(cycles=16, window_cycles=8, stride_cycles=4, durations_s=None)
            if mono:
                c["optics"].update(colors=["G"], wavelengths_m=[516e-9])
                c["schedule"].update(order=["G"], gains=[1])
            s = Schedule.from_config(validate(c))
            scale = 1 if mono else 3
            self.assertEqual(s.optimization_ends(), [8*scale, 12*scale, 16*scale])
            self.assertEqual([(end-s.window, end) for end in s.optimization_ends()],
                             [(0, 8*scale), (4*scale, 12*scale), (8*scale, 16*scale)])
            self.assertEqual(list(s.causal_blocks()),
                             [(0, 8*scale), (8*scale, 12*scale), (12*scale, 16*scale)])

    def test_tail_short_sequence_nonoverlap_and_legacy_startup(self):
        c = config()
        s = Schedule.from_config(c)
        self.assertEqual(s.optimization_ends(), [12, 18, 21])
        self.assertEqual(list(s.causal_blocks()), [(0, 12), (12, 18), (18, 21)])
        c["schedule"]["window_cycles"] = 8
        self.assertEqual(Schedule.from_config(c).optimization_ends(), [21])
        c["schedule"].update(window_cycles=2, stride_cycles=2)
        self.assertEqual(Schedule.from_config(c).optimization_ends(), [6, 12, 18, 21])
        c["schedule"]["stride_cycles"] = None
        self.assertEqual(Schedule.from_config(c).optimization_ends(), list(range(1, 22)))
        # A one-cycle stride is NOT legacy one-physical-exposure mode in RGB.
        c["schedule"]["stride_cycles"] = 1
        self.assertEqual(Schedule.from_config(c).optimization_ends(), [6, 9, 12, 15, 18, 21])

    def test_invalid_stride_and_old_config_upgrade(self):
        for value in (0, -1, 5, 1.5, 2.0, True, "2"):
            c = config()
            c["schedule"]["stride_cycles"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "stride_cycles"):
                validate(c)
        old = config()
        del old["schedule"]["stride_cycles"]
        self.assertIsNone(validate(old)["schedule"]["stride_cycles"])
        self.assertEqual(Schedule.from_config(old).optimization_ends(), list(range(1, 22)))

    def test_exposure_integral_unchanged_and_global_gradients_cover_every_frame(self):
        c = config("global")
        s = Schedule.from_config(c)
        indices = torch.arange(len(s))
        raw = torch.arange(1, len(s)+1, dtype=torch.float32)[:, None, None, None].requires_grad_()
        model = TemporalModel(s)
        perceived, mask = model.reconstruct(raw, indices)
        legacy = deepcopy(c)
        legacy["schedule"]["stride_cycles"] = None
        original, _ = TemporalModel(Schedule.from_config(legacy)).reconstruct(raw, indices)
        self.assertTrue(torch.equal(perceived, original))
        endpoints = s.endpoint_indices()
        # Explicit windows: physical exposures 1..12, 7..18, and 10..21.
        for out_index, (begin, end) in zip(endpoints, [(0, 12), (6, 18), (9, 21)]):
            for color in range(3):
                active = s.colors[begin:end] == color
                expected = (raw[begin:end, 0, 0, 0][active]*s.durations[begin:end][active]).sum()
                expected = expected*s.gains[color]/s.durations[begin:end].sum()
                torch.testing.assert_close(perceived[out_index, color, 0, 0, 0], expected)
        perceived[endpoints].sum().backward()
        self.assertTrue((raw.grad > 0).all())
        self.assertEqual(perceived.shape[0], 21)  # Evaluation is never subsampled.

    def test_all_methods_quantizers_and_motion_with_stride(self):
        for method in ("independent", "joint", "global", "causal"):
            for quant in ("continuous", "post_quantized", "ste", "smooth"):
                for motion in (False, True):
                    with self.subTest(method=method, quant=quant, motion=motion):
                        c = config(method, quant, motion)
                        if quant in ("continuous", "ste"):
                            c["target"]["scene"] = "static"
                        ctx = prepare(c)
                        r = optimize(ctx, False)
                        self.assertTrue(torch.isfinite(r["perceived"]).all())
                        self.assertEqual(len(r["phase"]), 21)
                        self.assertEqual(len(r["perceived"]), 21)
                        self.assertEqual(r["propagation_counts"]["phase_propagations"], 21*6)
                        torch.testing.assert_close(ctx.prop(r["phase"], ctx.schedule.colors).abs().square(),
                                                   r["raw_intensity"])
                        if quant != "continuous":
                            self.assertTrue(torch.equal(ctx.lut.decode(r["codes"], ctx.schedule.colors), r["phase"]))
                        if method == "global":
                            e = ctx.schedule.endpoint_indices()
                            score = loss_value(r["perceived"][e], r["target_perceived"][e], r["mask"][e]).item()
                            self.assertAlmostEqual(score, r["history"][-1]["best_loss"], places=6)
                        if method == "causal":
                            rows = [row for row in r["history"] if row["update"] == 2]
                            self.assertEqual([row["stage"] for row in rows], ["block_0_12", "block_12_18", "block_18_21"])
                            for endpoint, row in zip(ctx.schedule.endpoint_indices(), rows):
                                score = loss_value(r["perceived"][endpoint], r["target_perceived"][endpoint], r["mask"][endpoint]).item()
                                self.assertAlmostEqual(score, row["best_loss"], places=6)

    def test_causal_committed_blocks_unchanged_by_future_targets_or_appending_frames(self):
        base = prepare(config("causal", "continuous", True))
        original = optimize(base, False)
        for boundary in (12, 18):
            changed = prepare(config("causal", "continuous", True))
            changed.target[boundary:] = 1-changed.target[boundary:]
            new = optimize(changed, False)
            self.assertTrue(torch.equal(original["phase"][:boundary], new["phase"][:boundary]))
            self.assertFalse(torch.equal(original["phase"][boundary:], new["phase"][boundary:]))
            # Optimizing a shorter sequence must yield identical already committed blocks.
            c = config("causal", "continuous", True)
            c["schedule"].update(cycles=boundary//3, durations_s=c["schedule"]["durations_s"][:boundary])
            short = prepare(c)
            short.target = base.target[:boundary].clone()  # Same supplied physical target prefix.
            partial = optimize(short, False)
            self.assertTrue(torch.equal(original["phase"][:boundary], partial["phase"]))

    def test_stride_does_not_change_independent_or_joint(self):
        for method in ("independent", "joint"):
            c = config(method)
            a = optimize(prepare(c), False)
            c["schedule"]["stride_cycles"] = None
            b = optimize(prepare(c), False)
            self.assertTrue(torch.equal(a["phase"], b["phase"]))
            self.assertTrue(torch.equal(a["perceived"], b["perceived"]))
            self.assertEqual(a["history"], b["history"])

    def test_zero_motion_identity_sgd_amplitude_and_short_tail(self):
        for method in ("causal", "global"):
            c = config(method, "ste")
            c["optimization"].update(optimizer="sgd", loss="amplitude_mse", lr=.1)
            a = optimize(prepare(c), False)
            c["motion"].update(enabled=True, velocity_px_s=[0, 0])
            b = optimize(prepare(c), False)
            self.assertTrue(torch.equal(a["phase"], b["phase"]))
            self.assertTrue(torch.equal(a["perceived"], b["perceived"]))

    def test_block_fresh_process_reload_and_selected_checkpoint_loss(self):
        (ROOT/".cache").mkdir(exist_ok=True)
        for method in ("global", "causal"):
            with self.subTest(method=method), tempfile.TemporaryDirectory(dir=ROOT/".cache") as tmp:
                c = config(method, motion=True)
                c["output"] = tmp
                ctx = prepare(c)
                r = optimize(ctx, False)
                saved = save_run(ctx, r)
                proc = subprocess.run([sys.executable, "-m", "plmholo", "evaluate", "--run", tmp],
                                      cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                checked = json.loads((Path(tmp)/"evaluation.json").read_text())
                self.assertTrue(checked["reload_verified"])
                self.assertEqual(checked["selected_window_endpoints"], [11, 17, 20])
                self.assertAlmostEqual(checked["configured_loss_selected_windows"],
                                       saved["configured_loss_selected_windows"], places=7)
                self.assertEqual(checked["full_window_count"], 10)
                if method == "global":
                    self.assertAlmostEqual(checked["configured_loss_selected_windows"], r["history"][-1]["best_loss"], places=6)

    def test_resume_old_benchmark_ignores_only_missing_default_stride(self):
        (ROOT/".cache").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/".cache") as tmp:
            c = config("global")
            c["schedule"]["stride_cycles"] = None
            c["output"] = tmp
            c["benchmark"] = {"methods": ["global"], "scenes": ["static"], "seeds": [0]}
            saved_c = deepcopy(c)
            saved_c.update(name="static_seed0_global", output=str(Path(tmp)/"static_seed0_global"))
            saved_c["target"] = {"kind": "synthetic", "scene": "static", "path": None}
            del saved_c["schedule"]["stride_cycles"]
            ctx = prepare(saved_c)
            save_run(ctx, optimize(ctx, False))
            # Simulate an old resolved config written before stride existed.
            (Path(saved_c["output"])/"config.json").write_text(json.dumps(saved_c))
            with patch("plmholo.__main__.run_one", side_effect=AssertionError("Must reuse old run")):
                benchmark(c, resume=True)
                c["schedule"]["stride_cycles"] = 2
                with self.assertRaisesRegex(ValueError, "changed configuration"):
                    benchmark(c, resume=True)


if __name__ == "__main__":
    unittest.main()
