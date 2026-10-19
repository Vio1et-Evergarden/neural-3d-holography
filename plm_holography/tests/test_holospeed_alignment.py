"""Physical invariants of the deterministic Holospeed-style perception operator."""
import unittest
import torch

from plmholo.config import ROOT, load_config
from plmholo.motion import EyeMotion
from plmholo.schedule import Schedule
from plmholo.temporal import integrate


class HolospeedAlignmentTests(unittest.TestCase):
    def test_tracking_keeps_moving_target_sharp_while_fixed_eye_integrates_a_trail(self):
        # A point on each of three planes moves right by one pixel per exposure.
        video = torch.zeros(6, 1, 3, 9, 20)
        for time in range(6):
            for depth in range(3):
                video[time, 0, depth, 4, 4+depth+time] = 1
        position = torch.stack([torch.arange(6), torch.zeros(6)], dim=1).float()
        tracked, valid = EyeMotion(position).perceive(video, torch.ones(6), 6, torch.arange(6))
        fixed = integrate(video, torch.ones(6), 6)
        torch.testing.assert_close(tracked[-1]*valid[-1], video[-1]*valid[-1], rtol=0, atol=1e-6)
        for depth in range(3):
            self.assertAlmostEqual(tracked[-1, 0, depth].max().item(), 1., places=6)
            self.assertAlmostEqual(fixed[-1, 0, depth].max().item(), 1/6, places=6)
            self.assertEqual(torch.count_nonzero(fixed[-1, 0, depth]).item(), 6)

    def test_known_tracking_profiles_match_synthetic_motion_and_use_every_physical_endpoint(self):
        for method in ("causal", "global"):
            c = load_config(ROOT/f"configs/holospeed_known_motion_{method}.json")
            s = Schedule.from_config(c)
            eye = EyeMotion.from_config(c, s)
            fraction = (s.times-s.times[0])/(s.times[-1]-s.times[0])
            expected = .45*(c["optics"]["crop"][1]-1)/2*fraction
            torch.testing.assert_close(eye.positions[:, 0], expected.float(), atol=2e-6, rtol=1e-6)
            self.assertTrue((eye.positions[:, 1] == 0).all())
            self.assertTrue(torch.equal(s.endpoint_indices(), torch.arange(len(s))))
            self.assertIsNone(c["schedule"]["stride_cycles"])
            self.assertEqual(c["optimization"]["method"], method)


if __name__ == "__main__":
    unittest.main()
