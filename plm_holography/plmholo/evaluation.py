"""Metrics and fresh-process reconstruction using embedded run inputs."""
from pathlib import Path
import json
import math
import numpy as np
import torch
from .propagation import ASM
from .quantization import PhaseLUT
from .schedule import Schedule
from .temporal import TemporalModel
from .losses import loss_value


def metrics(perceived, target, mask, window):
    squared = (perceived-target).square()
    valid = torch.broadcast_to(mask, squared.shape)
    denominator = valid.sum(dim=(-2, -1))
    if (denominator == 0).any():
        raise ValueError("No valid pixels in one or more evaluated frames")
    per = (squared*valid).sum(dim=(-2, -1))/denominator
    first = window-1
    full = per[first:]
    score = full.mean().item() if len(full) else None
    # Peak value 1 is fixed in our relative linear-intensity units, never fitted.
    psnr = -10*math.log10(max(score, 1e-30)) if score is not None else None
    error = perceived-target
    temporal_rmse = None
    if len(full) > 1:
        delta = error[first+1:]-error[first:-1]
        common = mask[first+1:]*mask[first:-1]
        temporal_rmse = math.sqrt(loss_value(delta, torch.zeros_like(delta), common).item())
    return {
        "all_timestamps_intensity_mse": per.mean().item(),
        "full_window_intensity_mse": score,
        "full_window_psnr_db_peak1": psnr,
        "startup_intensity_mse": per[:first].mean().item() if first and len(per[:first]) else None,
        "full_window_count": len(full),
        "per_color_depth_mse": full.mean(0).tolist() if len(full) else None,
        "temporal_error_difference_rms": temporal_rmse,
        "per_timestamp_mse": per.mean(dim=(1, 2)).tolist(),
    }


def evaluate_run(run, device="cpu"):
    """Read the immutable run snapshot; no dependence on original target/LUT files."""
    run = Path(run)
    cfg = json.loads((run/"config.json").read_text())
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(cfg["threads"])
    schedule = Schedule.from_config(cfg, device)
    with np.load(run/"arrays.npz", allow_pickle=False) as data:
        def tensor(key, dtype=torch.float32):
            return torch.tensor(data[key].copy(), device=device, dtype=dtype)
        saved_phase = tensor("phase_radians")
        lut = PhaseLUT(schedule.names, phases=data["lut_phase_rad"].copy(), codes=data["lut_codes"].copy())
        if "codes" in data:
            phase = lut.decode(tensor("codes", torch.long), schedule.colors)
            phase_error = (phase-saved_phase).abs().max().item()
            torch.testing.assert_close(phase, saved_phase, rtol=0, atol=1e-6)
        else:
            phase, phase_error = saved_phase, 0.0
        target = tensor("target_intensity")
        expected = tensor("perceived_intensity")
        expected_raw = tensor("raw_intensity")
        expected_mask = tensor("valid_mask")
        positions = tensor("motion_positions_px")
        np.testing.assert_allclose(data["times_s"], schedule.times.cpu().numpy(), rtol=0, atol=1e-12)
        np.testing.assert_array_equal(data["color_indices"], schedule.colors.cpu().numpy())
    motion = None
    if cfg["motion"]["enabled"]:
        from .motion import EyeMotion
        motion = EyeMotion(positions)
    model = TemporalModel(schedule, motion)
    prop = ASM(cfg["optics"])
    indices = torch.arange(len(schedule), device=device)
    with torch.no_grad():
        raw = prop(phase, schedule.colors).abs().square()
        perceived, mask = model.reconstruct(raw, indices)
        goal, target_mask = model.perceive(target, indices)
        mask = mask*target_mask
    torch.testing.assert_close(raw, expected_raw, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(perceived, expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(mask, expected_mask, rtol=0, atol=0)
    result = metrics(perceived, goal, mask, schedule.window)
    result.update(phase_decode_max_error=phase_error,
                  raw_reconstruction_max_error=(raw-expected_raw).abs().max().item(),
                  perceived_reconstruction_max_error=(perceived-expected).abs().max().item(),
                  configured_loss_all_timestamps=loss_value(perceived, goal, mask, cfg["optimization"]["loss"]).item(),
                  device=device, reload_verified=True)
    recorded = json.loads((run/"metadata.json").read_text())
    for key in ("all_timestamps_intensity_mse", "full_window_intensity_mse"):
        expected_score = recorded["final"][key]
        if expected_score is not None and not math.isclose(result[key], expected_score, rel_tol=2e-5, abs_tol=2e-7):
            raise AssertionError(f"Reloaded {key} differs from the reported run score")
    if not math.isclose(result["configured_loss_all_timestamps"], recorded["configured_loss_all_timestamps"],
                        rel_tol=2e-5, abs_tol=2e-7):
        raise AssertionError("Reloaded configured loss differs from the reported run loss")
    (run/"evaluation.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return result
