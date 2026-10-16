"""Small CPU experiment using the repository's ASM and Adam phase optimizer.

No hardware, neural model, temporal frames, or phase quantization in the loss.
See ASM_WALKTHROUGH.md for the physical interpretation and commands.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import time

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).parent / 'results/.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import numpy as np
from skimage.transform import resize
import torch

import algorithms
from prop_ideal import Propagation
import utils


def load_target(path, channel, roi):
    """Same sRGB -> intensity -> amplitude order as TargetLoader.load_image."""
    rgb = imageio.imread(path)
    gray = rgb if rgb.ndim == 2 else rgb[..., channel]
    amplitude = np.sqrt(utils.srgb_gamma2lin(utils.im2float(gray, dtype=np.float64)))
    # Match image_loader.resize_keep_aspect: center crop, then resize amplitude.
    h, w = amplitude.shape
    crop_shape = (int(np.ceil(w * roi[0] / roi[1])),
                  int(np.ceil(h * roi[1] / roi[0])))
    amplitude = utils.crop_image(amplitude, crop_shape, pytorch=False)
    amplitude = resize(amplitude, roi, order=1, mode='reflect', anti_aliasing=True)
    return torch.from_numpy(amplitude.astype(np.float32))[None, None]


def evaluate(prop, phase, target, roi):
    amplitude = utils.crop_image(prop(phase), roi, stacked_complex=False).abs()
    scale = (amplitude * target).mean() / amplitude.square().mean()
    reconstruction = scale * amplitude
    return reconstruction, scale, (reconstruction - target).square().mean()


def display_intensity(amplitude):
    """Display sRGB-encoded intensity with fixed [0, 1] clipping, not auto contrast."""
    intensity = np.clip(amplitude ** 2, 0, 1)
    return utils.srgb_lin2gamma(intensity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, default=Path('data/test/rgb/0.png'))
    parser.add_argument('--out', type=Path, default=Path('results/asm_cpu'))
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.03)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--shape', nargs=2, type=int, default=[144, 256], metavar=('H', 'W'))
    parser.add_argument('--roi', nargs=2, type=int, default=[128, 224], metavar=('H', 'W'))
    parser.add_argument('--channel', type=int, choices=[0, 1, 2], default=1)
    parser.add_argument('--wavelength-nm', type=float, default=517.7)
    parser.add_argument('--pitch-um', type=float, default=6.4)
    parser.add_argument('--distance-mm', type=float, default=4.4)
    parser.add_argument('--aperture', type=float, default=0.5)
    args = parser.parse_args()
    if args.iterations < 1 or args.threads < 1 or args.lr <= 0:
        parser.error('iterations, threads, and lr must be positive')
    if any(r < 1 or s < r for s, r in zip(args.shape, args.roi)):
        parser.error('shape must contain the positive ROI dimensions')
    if min(args.wavelength_nm, args.pitch_um, args.distance_mm) <= 0 or not 0 < args.aperture <= 1:
        parser.error('optical lengths must be positive and aperture must be in (0, 1]')
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    dev = torch.device('cpu')
    print(f'Python {platform.python_version()}, torch {torch.__version__}, device={dev}', flush=True)
    print(f'CUDA={torch.cuda.is_available()}, MPS={torch.backends.mps.is_available()}', flush=True)

    # 1. Target amplitude and a phase-only, unit-amplitude incident field.
    target = load_target(args.image, args.channel, args.roi).to(dev)
    phase0 = (2 * torch.pi * (torch.rand(1, 1, *args.shape, device=dev) - 0.5))
    # 2. The original padded FFT -> H multiplication -> inverse FFT operator.
    prop = Propagation(args.distance_mm * 1e-3, args.wavelength_nm * 1e-9,
                       (args.pitch_um * 1e-6,) * 2, F_aperture=args.aperture).to(dev)
    started = time.perf_counter()
    with torch.no_grad():
        initial_amp, _, initial_loss = evaluate(prop, phase0, target, args.roi)

    # 3. A small directional derivative check: does autograd predict loss change?
    probe = phase0.detach().clone().requires_grad_(True)
    _, _, probe_loss = evaluate(prop, probe, target, args.roi)
    gradient, = torch.autograd.grad(probe_loss, probe)
    grad_rms = gradient.square().mean().sqrt()
    if not torch.isfinite(gradient).all() or grad_rms <= 0:
        raise RuntimeError('Initial phase gradient must be finite and nonzero')
    direction = gradient / grad_rms
    epsilon = 1e-3  # radians RMS
    with torch.no_grad():
        loss_plus = evaluate(prop, phase0 + epsilon * direction, target, args.roi)[2]
        loss_minus = evaluate(prop, phase0 - epsilon * direction, target, args.roi)[2]
    numerical = ((loss_plus - loss_minus) / (2 * epsilon)).item()
    analytical = (gradient * direction).sum().item()
    derivative_error = abs(numerical - analytical) / max(abs(analytical), 1e-12)
    if derivative_error > 0.05 or not loss_minus < initial_loss:
        raise RuntimeError(f'Gradient check failed: relative error={derivative_error}')
    print(f'Gradient check: relative error={derivative_error:.3g}; '
          f'loss(phi-eps*d)={loss_minus.item():.6g} < {initial_loss.item():.6g}', flush=True)

    history = []
    def record(iteration, stats):
        history.append({'iteration': iteration, **stats})
        if iteration == 1 or iteration % 50 == 0 or iteration == args.iterations:
            print(f'iteration {iteration:4d}: loss={stats["loss"]:.6g}, '
                  f'gradient_rms={stats["gradient_rms"]:.3g}', flush=True)

    # 4. Only the phase tensor is optimized; the propagation model is fixed.
    result = algorithms.gradient_descent(
        phase0.clone(), target, forward_prop=prop, num_iters=args.iterations,
        roi_res=args.roi, lr=args.lr, iteration_callback=record)
    phase = result['final_phase']
    with torch.no_grad():
        reconstruction, scale, final_loss = evaluate(prop, phase, target, args.roi)
    torch.testing.assert_close(reconstruction, result['recon_amp'])
    if not np.isclose(final_loss.item(), result['best_loss'], rtol=1e-5, atol=1e-8):
        raise RuntimeError('Saved phase and best loss do not match')

    # 5. Floating-point phase is the scientific result; PNG is just a preview.
    def array(tensor):
        return tensor.detach().cpu().squeeze().numpy()
    phase_array = array(phase)
    wrapped = (phase_array + np.pi) % (2 * np.pi) - np.pi
    np.save(args.out / 'phase_radians.npy', phase_array)
    np.save(args.out / 'target_amplitude.npy', array(target))
    np.save(args.out / 'reconstruction_amplitude.npy', array(reconstruction))
    np.save(args.out / 'reconstruction_intensity.npy', array(reconstruction) ** 2)
    np.save(args.out / 'initial_phase_radians.npy', array(phase0))
    imageio.imwrite(args.out / 'phase_preview.png', utils.phasemap_8bit(phase, inverted=False))
    for name, amplitude in [('target', target), ('initial_reconstruction', initial_amp),
                            ('reconstruction', reconstruction)]:
        display = display_intensity(array(amplitude))
        imageio.imwrite(args.out / f'{name}.png', np.round(display * 255).astype(np.uint8))
    with (args.out / 'loss.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    # Reload the saved float phase into a fresh propagation operator.
    reloaded = torch.from_numpy(np.load(args.out / 'phase_radians.npy'))[None, None]
    fresh_prop = Propagation(args.distance_mm * 1e-3, args.wavelength_nm * 1e-9,
                             (args.pitch_um * 1e-6,) * 2, F_aperture=args.aperture)
    with torch.no_grad():
        checked = evaluate(fresh_prop, reloaded, target, args.roi)[0]
    torch.testing.assert_close(checked, reconstruction)
    metadata = {
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
        'device': str(dev), 'image_sha256': hashlib.sha256(args.image.read_bytes()).hexdigest(),
        'initial_loss': initial_loss.item(), 'best_loss': final_loss.item(),
        'best_iteration': result['best_iter'], 'amplitude_scale': scale.item(),
        'gradient_check_relative_error': derivative_error,
        'negative_gradient_probe_loss': loss_minus.item(),
        'saved_phase_reconstruction_max_error': (checked - reconstruction).abs().max().item(),
        'phase_change_rms_rad': (phase - phase0).square().mean().sqrt().item(),
        'elapsed_seconds': time.perf_counter() - started,
        'phase_encoding': 'phase_radians.npy: continuous radians; PNG: non-inverted 8-bit preview only',
        'reconstruction_encoding': 'NPY: scaled amplitude / linear intensity; PNG: sRGB intensity clipped to [0,1]',
        'iteration_convention': 'Loss and best phase evaluated before each Adam step; best phase is returned.',
    }
    (args.out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.8), constrained_layout=True)
    for ax, title, amplitude in zip(axes[0], ['Target intensity', 'Initial reconstruction', 'Optimized reconstruction'],
                                     [target, initial_amp, reconstruction]):
        ax.imshow(display_intensity(array(amplitude)), cmap='gray', vmin=0, vmax=1)
        ax.set_title(title + ' (sRGB)')
        ax.axis('off')
    im = axes[1, 0].imshow(wrapped, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[1, 0].set_title('Continuous phase (rad)')
    axes[1, 0].axis('off')
    fig.colorbar(im, ax=axes[1, 0], shrink=0.7)
    axes[1, 1].semilogy([r['iteration'] for r in history], result['loss_vals'])
    axes[1, 1].set(xlabel='Evaluation / iteration', ylabel='Amplitude MSE', title='Optimization loss')
    axes[1, 1].grid(alpha=0.3)
    error = np.abs(array(reconstruction) - array(target))
    im = axes[1, 2].imshow(error, cmap='magma', vmin=0)
    axes[1, 2].set_title('Absolute amplitude error')
    axes[1, 2].axis('off')
    fig.colorbar(im, ax=axes[1, 2], shrink=0.7)
    fig.savefig(args.out / 'overview.png', dpi=160)
    plt.close(fig)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    main()
