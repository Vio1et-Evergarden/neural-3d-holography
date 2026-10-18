import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
import imageio.v3 as iio
from plmholo.config import ROOT, DEFAULTS, merge, validate, load_config
from plmholo.schedule import Schedule
from plmholo.targets import load_target


class InputTests(unittest.TestCase):
    def setUp(self):
        (ROOT/".cache").mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT/".cache")
        self.path = Path(self.tmp.name)
        self.c = validate(merge(DEFAULTS, {"optics": {"shape": [4, 6]},
                                          "schedule": {"cycles": 2, "window_cycles": 1}}))
        self.schedule = Schedule.from_config(self.c)

    def tearDown(self):
        self.tmp.cleanup()

    def test_static_npz_broadcast_and_dynamic_interpolation(self):
        path = self.path/"targets.npz"
        np.savez(path, intensity=np.full((1, 3, 3, 8, 12), .25, dtype=np.float32))
        self.c["target"] = {"kind": "npz", "path": str(path)}
        target = load_target(self.c, self.schedule)
        self.assertEqual(tuple(target.shape), (6, 3, 3, 4, 6))
        torch.testing.assert_close(target, torch.full_like(target, .25))
        np.savez(path, intensity=np.stack([np.zeros((3, 3, 4, 6)), np.ones((3, 3, 4, 6))]),
                 times_s=np.array([0., .01]))
        target = load_target(self.c, self.schedule)
        torch.testing.assert_close(target[:, 0, 0, 0, 0], (self.schedule.times/.01).float())

    def test_manifest_srgb_and_relative_image_paths(self):
        iio.imwrite(self.path/"gray.png", np.full((6, 8, 3), 128, dtype=np.uint8))
        manifest = self.path/"manifest.json"
        manifest.write_text(json.dumps({"encoding": "srgb", "frames": [{"depth_images": ["gray.png"]*3}]}))
        self.c["target"] = {"kind": "manifest", "path": str(manifest)}
        target = load_target(self.c, self.schedule)
        expected = ((128/255+.055)/1.055)**2.4
        torch.testing.assert_close(target, torch.full_like(target, expected))

    def test_bad_target_and_missing_timestamps(self):
        path = self.path/"bad.npz"
        self.c["target"] = {"kind": "npz", "path": str(path)}
        np.savez(path, intensity=np.zeros((2, 3, 3, 4, 6)))
        with self.assertRaisesRegex(ValueError, "times_s"):
            load_target(self.c, self.schedule)
        np.savez(path, intensity=-np.ones((1, 3, 3, 4, 6)))
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            load_target(self.c, self.schedule)

    def test_inherited_paths_belong_to_defining_file(self):
        parent = self.path/"parent"
        child = self.path/"child"
        parent.mkdir()
        child.mkdir()
        (parent/"base.json").write_text(json.dumps({"target": {"kind": "npz", "path": "target.npz"},
                                                    "output": "out"}))
        (child/"config.json").write_text(json.dumps({"extends": "../parent/base.json", "seed": 3}))
        c = load_config(child/"config.json")
        self.assertEqual(c["target"]["path"], str(parent/"target.npz"))
        self.assertEqual(c["output"], str(parent/"out"))
        (parent/"base.json").write_text(json.dumps({"extends": "../child/config.json"}))
        with self.assertRaisesRegex(ValueError, "Circular"):
            load_config(child/"config.json")

    def test_configuration_typo_rejected(self):
        c = merge(self.c, {"schedule": {"window_size": 99}})
        with self.assertRaisesRegex(ValueError, "Unknown"):
            validate(c)


if __name__ == "__main__":
    unittest.main()
