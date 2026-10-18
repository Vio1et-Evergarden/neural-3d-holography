import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from plmholo.config import ROOT, DEFAULTS, merge, validate, load_config
from plmholo.optimization import prepare, optimize
from plmholo.artifacts import save_run
from plmholo.evaluation import evaluate_run


def tiny(method="global", quantization="smooth", motion=False):
    return validate(merge(DEFAULTS, {
        "threads": 1, "optics": {"shape": [8, 10]},
        "schedule": {"cycles": 3, "window_cycles": 2},
        "quantization": {"method": quantization},
        "optimization": {"method": method, "iterations": 2, "log_every": 0},
        "motion": {"enabled": motion, "velocity_px_s": [80, 0]},
        "output": str(ROOT/".cache"/"test-run"),
    }))


class OptimizationTests(unittest.TestCase):
    def test_all_methods_and_quantizers(self):
        for method in ("independent", "joint", "global", "causal"):
            for quant in ("continuous", "post_quantized", "ste", "smooth"):
                with self.subTest(method=method, quantization=quant):
                    ctx = prepare(tiny(method, quant))
                    result = optimize(ctx, verbose=False)
                    self.assertTrue(torch.isfinite(result["perceived"]).all())
                    expected = len(ctx.schedule)*(2+2*ctx.config["optimization"]["iterations"])
                    self.assertEqual(result["propagation_counts"]["phase_propagations"], expected)
                    fresh = ctx.prop(result["phase"], ctx.schedule.colors).abs().square()
                    torch.testing.assert_close(fresh, result["raw_intensity"])
                    if quant != "continuous":
                        decoded = ctx.lut.decode(result["codes"], ctx.schedule.colors)
                        self.assertTrue(torch.equal(decoded, result["phase"]))
                    for row in result["history"]:
                        self.assertLessEqual(row["best_loss"], row["evaluated_loss"]+1e-12)

    def test_causal_future_targets_cannot_change_committed_history(self):
        first = prepare(tiny("causal", "continuous"))
        second = prepare(tiny("causal", "continuous"))
        second.target[5:] = 1-second.target[5:]
        a, b = optimize(first, False), optimize(second, False)
        self.assertTrue(torch.equal(a["phase"][:5], b["phase"][:5]))
        self.assertGreater((a["phase"][5:]-b["phase"][5:]).abs().sum().item(), 0)

    def test_optional_motion_and_fresh_process_reload(self):
        cache = ROOT/".cache"
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as tmp:
            c = tiny("causal", "smooth", motion=True)
            c["output"] = str(Path(tmp)/"run")
            ctx = prepare(c)
            result = optimize(ctx, False)
            saved = save_run(ctx, result)
            proc = subprocess.run([sys.executable, "-m", "plmholo", "evaluate", "--run", c["output"]],
                                  cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            evaluation = json.loads((Path(c["output"])/"evaluation.json").read_text())
            self.assertTrue(evaluation["reload_verified"])
            self.assertAlmostEqual(evaluation["full_window_intensity_mse"], saved["final"]["full_window_intensity_mse"], places=7)

    def test_sgd_amplitude_loss_and_nonuniform_exposures(self):
        c = tiny("global", "continuous")
        c["optimization"].update(optimizer="sgd", loss="amplitude_mse", lr=.1)
        c["schedule"]["durations_s"] = [.001, .002, .003]*3
        result = optimize(prepare(c), False)
        self.assertTrue(torch.isfinite(result["perceived"]).all())

    def test_configs_and_output_boundary(self):
        for file in (ROOT/"configs").glob("*.json"):
            load_config(file)
        c = tiny()
        c["output"] = str(ROOT.parent/"must-not-write")
        with self.assertRaises(ValueError):
            validate(c)

    def test_motion_off_does_not_import_optional_module(self):
        code = ("import sys; from plmholo.config import load_config; "
                "from plmholo.optimization import prepare; "
                "prepare(load_config('configs/smoke.json')); "
                "assert 'plmholo.motion' not in sys.modules")
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_checkpoint_loss_matches_actual_global_state(self):
        from plmholo.losses import loss_value
        ctx = prepare(tiny("global", "smooth"))
        result = optimize(ctx, False)
        loss = loss_value(result["perceived"], result["target_perceived"], result["mask"]).item()
        self.assertAlmostEqual(loss, result["history"][-1]["best_loss"], places=7)

    def test_replacing_external_lut_does_not_change_saved_run(self):
        import csv
        cache = ROOT/".cache"
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as tmp:
            tmp = Path(tmp)
            lut = tmp/"measured.csv"
            with lut.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["color", "code", "phase_rad"])
                for ci, color in enumerate("RGB"):
                    for code in range(16):
                        writer.writerow([color, code, 2*np.pi*((15-code)/16)**(1.05+.1*ci)])
            c = tiny("global", "smooth")
            c["quantization"]["lut_path"] = str(lut)
            c["output"] = str(tmp/"run")
            ctx = prepare(c)
            result = optimize(ctx, False)
            saved = save_run(ctx, result)
            # An old run must use its embedded calibration even after the source changes.
            lut.write_text("this external table has been replaced\n")
            checked = evaluate_run(c["output"])
            self.assertAlmostEqual(checked["full_window_intensity_mse"], saved["final"]["full_window_intensity_mse"], places=7)


if __name__ == "__main__":
    unittest.main()
