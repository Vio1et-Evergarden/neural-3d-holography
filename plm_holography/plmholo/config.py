"""JSON configuration, relative path resolution, and explicit physical defaults."""
from copy import deepcopy
import json
from pathlib import Path
import math

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {
    "name": "rgb_demo", "seed": 0, "device": "cpu", "threads": 2,
    "optics": {"shape": [64, 64], "crop": None, "computational_shape": None,
               "pad_factor": 2, "pitch_m": [10.8e-6, 10.8e-6],
               "colors": ["R", "G", "B"],
               "wavelengths_m": [640e-9, 516e-9, 455.5e-9],
               "depths_m": [3.4e-3, 4.4e-3, 5.7e-3],
               "frequency_grid": "fft", "numerical_aperture": None,
               "propagation_batch": 4},
    "schedule": {"cycles": 16, "order": ["G", "R", "B"], "rate_hz": 1440.0,
                 "durations_s": None, "window_cycles": 8, "gains": [3., 3., 3.]},
    "target": {"kind": "synthetic", "scene": "moving", "path": None},
    "motion": {"enabled": False, "path": None, "velocity_px_s": [0., 0.]},
    "quantization": {"method": "smooth", "levels": 16, "lut_path": None,
                     "tau_start_steps": 0.5, "tau_end_steps": 0.1},
    "optimization": {"method": "global", "optimizer": "adam", "iterations": 200,
                     "lr": None, "loss": "intensity_mse", "log_every": 50},
    "output": "results/rgb_demo",
    "benchmark": {"methods": ["independent", "joint", "global", "causal"],
                  "scenes": ["static", "moving"], "seeds": [0, 1, 2]},
}


def merge(base, update):
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _read(path, ancestors=()):
    path = Path(path).resolve()
    if path in ancestors:
        raise ValueError("Circular configuration inheritance")
    data = json.loads(path.read_text())
    parent = data.pop("extends", None)
    # Each file owns the relative paths it introduces, including inherited paths.
    for section, key in [("target", "path"), ("motion", "path"),
                         ("quantization", "lut_path")]:
        if data.get(section, {}).get(key):
            data[section][key] = str((path.parent / data[section][key]).resolve())
    if "output" in data:
        data["output"] = str((path.parent / data["output"]).resolve())
    inherited = _read(path.parent / parent, (*ancestors, path)) if parent else {}
    return merge(inherited, data)


def load_config(path):
    return validate(merge(DEFAULTS, _read(path)))


def validate(config):
    c = deepcopy(config)
    for section, value in c.items():
        if section not in DEFAULTS:
            raise ValueError(f"Unknown configuration section: {section}")
        if isinstance(DEFAULTS[section], dict):
            unknown = set(value)-set(DEFAULTS[section])
            if unknown:
                raise ValueError(f"Unknown {section} options: {sorted(unknown)}")
    o, s, q, opt = (c[k] for k in ("optics", "schedule", "quantization", "optimization"))
    colors = o["colors"]
    if len(colors) not in (1, 3) or len(set(colors)) != len(colors) or any(x not in ("R", "G", "B") for x in colors):
        raise ValueError("Use one monochrome channel or all three unique R,G,B channels")
    if len(o["wavelengths_m"]) != len(colors) or len(s["gains"]) != len(colors):
        raise ValueError("One wavelength and fixed gain are required per color")
    if sorted(s["order"]) != sorted(colors):
        raise ValueError("schedule.order must contain each configured color once")
    if any(not math.isfinite(x) or x <= 0 for x in [*o["wavelengths_m"], *o["pitch_m"],
                                                    s["rate_hz"], *s["gains"]]):
        raise ValueError("Wavelengths, pitch, rate and gains must be finite and positive")
    if not o["depths_m"] or any(not math.isfinite(z) or z < 0 for z in o["depths_m"]):
        raise ValueError("Depths must be finite and nonnegative")
    for key in ("shape", "crop", "computational_shape"):
        value = o[key]
        if value is not None and (len(value) != 2 or any(int(v) != v or v < 1 for v in value)):
            raise ValueError(f"optics.{key} must contain two positive integers")
    o["crop"] = o["crop"] or list(o["shape"])
    if any(a > b for a, b in zip(o["crop"], o["shape"])):
        raise ValueError("Output crop cannot exceed the phase shape")
    if int(o["pad_factor"]) != o["pad_factor"] or o["pad_factor"] < 1:
        raise ValueError("pad_factor must be a positive integer")
    o["computational_shape"] = o["computational_shape"] or [o["pad_factor"] * n for n in o["shape"]]
    if any(a < b for a, b in zip(o["computational_shape"], o["shape"])):
        raise ValueError("Computational grid must contain the phase grid")
    for value in [s["cycles"], s["window_cycles"], opt["iterations"], c["threads"], o["propagation_batch"]]:
        if int(value) != value or value < 1:
            raise ValueError("Counts must be positive integers")
    if o["frequency_grid"] not in ("fft", "legacy"):
        raise ValueError("frequency_grid must be fft or legacy")
    if o["numerical_aperture"] is not None and not 0 < o["numerical_aperture"] <= 1:
        raise ValueError("numerical_aperture must be in (0,1] or null")
    if q["levels"] != 16:
        raise ValueError("This PLM release uses exactly 16 states")
    if q["method"] not in ("continuous", "post_quantized", "ste", "smooth"):
        raise ValueError("Unknown quantization method")
    if not 0 < q["tau_end_steps"] <= q["tau_start_steps"]:
        raise ValueError("Smoothing widths must satisfy 0 < end <= start")
    if opt["method"] not in ("independent", "joint", "global", "causal"):
        raise ValueError("Unknown temporal optimization method")
    if opt["optimizer"] not in ("adam", "sgd") or opt["loss"] not in ("intensity_mse", "amplitude_mse"):
        raise ValueError("Unknown optimizer or loss")
    if opt["lr"] is None:
        opt["lr"] = 0.001 if q["method"] in ("ste", "smooth") else 0.01
    if not math.isfinite(opt["lr"]) or opt["lr"] <= 0:
        raise ValueError("Learning rate must be positive")
    if c["device"] not in ("cpu", "cuda", "auto"):
        raise ValueError("Supported devices: cpu, cuda, auto")
    if c["target"]["kind"] not in ("synthetic", "npz", "manifest"):
        raise ValueError("Unknown target kind")
    out = Path(c["output"])
    out = out.resolve() if out.is_absolute() else (ROOT / out).resolve()
    if not out.is_relative_to(ROOT):
        raise ValueError("All new experiment outputs must stay inside plm_holography/")
    c["output"] = str(out)
    return c
