"""Regression checks for phase/loss correspondence, using the real ASM operator."""
import unittest

import torch

from algorithms import gradient_descent
from prop_ideal import Propagation


class PhaseSnapshotTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)
        self.phase = torch.rand(1, 1, 12, 16) * (2 * torch.pi) - torch.pi
        self.target = torch.rand_like(self.phase)
        self.prop = Propagation(4.4e-3, 517.7e-9, (6.4e-6, 6.4e-6), F_aperture=0.5)

    def check_result(self, result, count):
        with torch.no_grad():
            amp = self.prop(result['final_phase']).abs()
            scale = (amp * self.target).mean() / amp.square().mean()
            loss = ((scale * amp - self.target) ** 2).mean().item()
        self.assertAlmostEqual(loss, result['best_loss'], places=7)
        self.assertEqual(len(result['loss_vals']), count)
        self.assertAlmostEqual(min(result['loss_vals']), result['best_loss'], places=7)
        torch.testing.assert_close(scale * amp, result['recon_amp'])
        self.assertFalse(result['final_phase'].requires_grad)
        self.assertFalse(result['recon_amp'].requires_grad)

    def test_one_evaluation_snapshots_phase_before_update(self):
        result = gradient_descent(self.phase.clone(), self.target,
                                  forward_prop=self.prop, num_iters=1, lr=0.2)
        self.check_result(result, 1)
        torch.testing.assert_close(result['final_phase'], self.phase)

    def test_best_phase_survives_later_updates(self):
        result = gradient_descent(self.phase.clone(), self.target,
                                  forward_prop=self.prop, num_iters=12, lr=3.0)
        self.check_result(result, 12)
        self.assertLess(result['best_iter'], 12)


if __name__ == '__main__':
    unittest.main()
