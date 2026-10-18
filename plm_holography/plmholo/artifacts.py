"""Self-contained scientific run snapshots and fixed-scale visualizations."""
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import numpy as np
import torch
from .config import ROOT
from .evaluation import metrics
from .losses import loss_value
from . import __version__


def array(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def digest(x):
    x = np.ascontiguousarray(array(x))
    return hashlib.sha256(str(x.shape).encode()+str(x.dtype).encode()+x.tobytes()).hexdigest()


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def plot_run(ctx, result, output):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT/".cache"/"matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v3 as iio
    cfg, schedule = ctx.config, ctx.schedule
    ndepth = len(cfg["optics"]["depths_m"])
    endpoint = len(schedule)-1
    target = array(result["target_perceived"])
    initial = array(result["initial_perceived"])
    final = array(result["perceived"])

    def rgb(frame):
        if len(schedule.names) == 1:
            return frame[0]
        return np.stack([frame[schedule.names.index(c)] for c in "RGB"], axis=-1)

    fig, axes = plt.subplots(ndepth, 3, figsize=(9, 3*ndepth), squeeze=False, constrained_layout=True)
    for d, z in enumerate(cfg["optics"]["depths_m"]):
        for col, values in enumerate((target, initial, final)):
            frame = rgb(values[endpoint, :, d])
            axes[d, col].imshow(np.clip(frame, 0, 1), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            axes[d, col].set_xticks([])
            axes[d, col].set_yticks([])
            if col == 0:
                axes[d, col].set_ylabel(f"{z*1e3:g} mm")
            if d == 0:
                axes[d, col].set_title(("Perceived target", "Initial", "Final")[col])
    fig.suptitle("Linear intensity; fixed display range [0,1]; values above 1 clipped only in preview")
    fig.savefig(output/"comparison.png", dpi=140)
    plt.close(fig)
    scores = metrics(result["perceived"], result["target_perceived"], result["mask"], schedule.window)
    before = metrics(result["initial_perceived"], result["target_perceived"], result["mask"], schedule.window)
    fig, ax = plt.subplots(figsize=(8, 4))
    t = schedule.times.cpu().numpy()*1000
    ax.plot(t, before["per_timestamp_mse"], label="Initial")
    ax.plot(t, scores["per_timestamp_mse"], label="Final")
    if schedule.window <= len(schedule):
        ax.axvline(t[schedule.window-1], linestyle="--", color="gray", label="First full window")
    ax.set(xlabel="Physical time (ms)", ylabel="Integrated intensity MSE")
    ax.legend()
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(output/"temporal_error.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    records = result["history"]
    # Break lines between local objectives; values from different stages are not a global loss curve.
    for stage in dict.fromkeys(r["stage"] for r in records):
        rows = [r for r in records if r["stage"] == stage]
        positions = [i for i, r in enumerate(records) if r["stage"] == stage]
        ax.plot(positions, [r["evaluated_loss"] for r in rows], color="C0", alpha=.65)
        ax.plot(positions, [r["best_loss"] for r in rows], color="C1", linewidth=1)
    ax.set(xlabel="Evaluation record (each stage has its own objective)", ylabel="Stage loss",
           title="Blue: evaluated state; orange: best within that stage")
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(output/"optimization.png", dpi=140)
    plt.close(fig)
    # GIF is a deliberately slowed preview, not device playback or calibrated colorimetry.
    frames = []
    for i in range(len(schedule)):
        rows = []
        for d in range(ndepth):
            parts = []
            for values in (target, final):
                frame = rgb(values[i, :, d])
                if frame.ndim == 2:
                    frame = np.repeat(frame[..., None], 3, axis=-1)
                parts.append(np.uint8(np.clip(frame, 0, 1)*255+.5))
            rows.append(np.concatenate(parts, axis=1))
        frames.append(np.concatenate(rows, axis=0))
    iio.imwrite(output/"sequence_preview.gif", np.stack(frames), duration=100, loop=0)


def save_run(ctx, result):
    output = Path(ctx.config["output"])
    if (output/"metadata.json").exists():
        raise FileExistsError(f"Run already exists: {output}; choose another output directory")
    output.mkdir(parents=True, exist_ok=True)
    schedule = ctx.schedule
    arrays = {
        "phase_radians": array(result["phase"]),
        "initial_phase_radians": array(result["initial_phase"]),
        "raw_intensity": array(result["raw_intensity"]),
        "perceived_intensity": array(result["perceived"]),
        "initial_perceived_intensity": array(result["initial_perceived"]),
        "target_intensity": array(result["target"]),
        "perceived_target_intensity": array(result["target_perceived"]),
        "valid_mask": array(result["mask"]),
        "color_indices": array(schedule.colors), "times_s": array(schedule.times),
        "durations_s": array(schedule.durations), "source_gains": array(schedule.gains),
        "lut_phase_rad": array(ctx.lut.phases), "lut_codes": array(ctx.lut.codes),
        "motion_positions_px": array(result["motion_positions"]),
    }
    if result["codes"] is not None:
        arrays["codes"] = array(result["codes"]).astype(np.uint8)
    np.savez_compressed(output/"arrays.npz", **arrays)
    json_write(output/"config.json", ctx.config)
    with (output/"loss.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result["history"][0]))
        writer.writeheader()
        writer.writerows(result["history"])
    final = metrics(result["perceived"], result["target_perceived"], result["mask"], schedule.window)
    initial = metrics(result["initial_perceived"], result["target_perceived"], result["mask"], schedule.window)
    metadata = {
        "library_version": __version__, "python": platform.python_version(),
        "torch": torch.__version__, "numpy": np.__version__, "platform": platform.platform(),
        "device": str(ctx.device), "seed": ctx.config["seed"],
        "elapsed_optimization_seconds": result["elapsed_seconds"],
        "propagation_counts": result["propagation_counts"],
        "optimized_phase_updates": result["optimized_phase_updates"],
        "target_sha256": digest(result["target"]), "lut_sha256": digest(ctx.lut.phases),
        "lut_codes_sha256": digest(ctx.lut.codes), "initial_phase_sha256": digest(result["initial_phase"]),
        "motion_sha256": digest(result["motion_positions"]),
        "final": final, "initial": initial,
        "configured_loss_all_timestamps": loss_value(result["perceived"], result["target_perceived"], result["mask"], ctx.config["optimization"]["loss"]).item(),
        "preview_clipped_fraction": (result["perceived"] > 1).float().mean().item(),
        "power_units": "relative linear intensity; gains fixed, RGB duty compensation explicit",
        "checkpoint_selection": "Best actual evaluated state within each optimizer stage; initialization included",
        "phase_model": "ideal or supplied 16-state phase-only LUT; no calibrated hardware response claimed",
    }
    plot_run(ctx, result, output)
    json_write(output/"metadata.json", metadata)
    return metadata
