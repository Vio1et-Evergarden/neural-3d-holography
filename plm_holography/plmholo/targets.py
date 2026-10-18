"""Linear-intensity focal stacks and timestamped resampling."""
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn.functional as F
import imageio.v3 as iio


def srgb_to_linear(x):
    return np.where(x <= 0.04045, x/12.92, ((x+0.055)/1.055)**2.4)


def resample(values, source_times, destination_times):
    """Linear interpolation; endpoints held outside the supplied interval."""
    if len(values) == 1:
        return values.expand(len(destination_times), *values.shape[1:]).clone()
    t = torch.as_tensor(source_times, device=values.device, dtype=torch.float64)
    if t.shape != (len(values),) or not torch.isfinite(t).all() or not (t[1:] > t[:-1]).all():
        raise ValueError("Dynamic sample times must be finite and strictly increasing")
    dest = destination_times.to(t).clamp(t[0], t[-1])
    right = torch.searchsorted(t.contiguous(), dest.contiguous()).clamp(1, len(t)-1)
    left = right-1
    alpha = ((dest-t[left])/(t[right]-t[left])).to(values.dtype)
    alpha = alpha.reshape(-1, *([1]*(values.ndim-1)))
    return values[left]*(1-alpha)+values[right]*alpha


def resize_stack(values, shape):
    """Center crop to target aspect, then antialiased resize of linear intensity."""
    h, w = values.shape[-2:]
    oh, ow = shape
    if h*ow > w*oh:
        nh = max(1, round(w*oh/ow))
        values = values[..., (h-nh)//2:(h-nh)//2+nh, :]
    else:
        nw = max(1, round(h*ow/oh))
        values = values[..., :, (w-nw)//2:(w-nw)//2+nw]
    leading = values.shape[:-2]
    resized = F.interpolate(values.reshape(-1, 1, *values.shape[-2:]), size=shape,
                            mode="bilinear", align_corners=False, antialias=True)
    return resized.reshape(*leading, *shape)


def synthetic(config, schedule, device):
    o = config["optics"]
    h, w = o["crop"]
    y, x = torch.meshgrid(torch.linspace(-1, 1, h, device=device),
                          torch.linspace(-1, 1, w, device=device), indexing="ij")
    t = schedule.times.to(dtype=torch.float32)
    fraction = (t-t[0])/(t[-1]-t[0]).clamp_min(1e-12)
    scene = config["target"].get("scene", "moving")
    if scene not in ("static", "moving"):
        raise ValueError("Synthetic scene must be static or moving")
    shift = 0.45*fraction if scene == "moving" else torch.zeros_like(fraction)
    stacks = []
    for ci, _ in enumerate(o["colors"]):
        depths = []
        for di, _ in enumerate(o["depths_m"]):
            xx = x[None]-shift[:, None, None]-0.12*(di-(len(o["depths_m"])-1)/2)
            yy = y[None]+0.12*(ci-1)
            texture = 0.10*torch.sin(14*xx+ci)*torch.cos(11*yy+di)
            spot = 0.30*torch.exp(-((xx+0.25)**2+(yy-0.10)**2)/0.10)
            ring = 0.12*torch.exp(-((torch.sqrt((xx-0.3)**2+(yy+0.25)**2)-0.28)/0.08)**2)
            depths.append((0.45+texture+spot+ring).clamp(0, 1))
        stacks.append(torch.stack(depths, dim=1))
    return torch.stack(stacks, dim=1)


def load_target(config, schedule, device="cpu"):
    tc, o = config["target"], config["optics"]
    if tc["kind"] == "synthetic":
        return synthetic(config, schedule, device)
    path = Path(tc["path"])
    if tc["kind"] == "npz":
        with np.load(path, allow_pickle=False) as data:
            values = torch.as_tensor(data["intensity"].copy(), dtype=torch.float32, device=device)
            times = data["times_s"].copy() if "times_s" in data else None
    else:
        manifest = json.loads(path.read_text())
        if not manifest.get("frames"):
            raise ValueError("Image manifest must contain at least one frame")
        frames, times = [], []
        for frame in manifest["frames"]:
            if len(frame["depth_images"]) != len(o["depths_m"]):
                raise ValueError("Manifest must supply one RGB image for each depth")
            planes = []
            for name in frame["depth_images"]:
                image = iio.imread(path.parent / name)
                if image.dtype.kind in "ui":
                    image = image.astype(np.float32)/np.iinfo(image.dtype).max
                else:
                    image = image.astype(np.float32)
                if image.ndim == 2:
                    if len(o["colors"]) != 1:
                        raise ValueError("RGB configurations require RGB image files")
                    image = image[..., None]
                else:
                    image = image[..., ["RGB".index(c) for c in o["colors"]]]
                if manifest.get("encoding", "srgb") == "srgb":
                    image = srgb_to_linear(image)
                elif manifest["encoding"] != "linear":
                    raise ValueError("Manifest encoding must be srgb or linear")
                plane = torch.as_tensor(image.transpose(2, 0, 1), device=device)
                planes.append(resize_stack(plane, o["crop"]))
            frames.append(torch.stack(planes, dim=1))
            times.append(frame.get("time_s", 0.0))
        values = torch.stack(frames).float()
    expected = (len(o["colors"]), len(o["depths_m"]))
    if values.ndim != 5 or tuple(values.shape[1:3]) != expected or len(values) < 1:
        raise ValueError(f"Target must have shape [T,{expected[0]},{expected[1]},H,W]")
    if not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError("Target intensities must be finite and nonnegative")
    if len(values) > 1 and times is None:
        raise ValueError("Dynamic NPZ targets require times_s")
    values = resize_stack(values, o["crop"])
    return resample(values, times, schedule.times)
