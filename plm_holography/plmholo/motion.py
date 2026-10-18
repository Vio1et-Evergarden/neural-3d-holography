"""Optional retinal translations, not a pupil or optical-flow model.

Positive x eye displacement moves a stationary image left on the retina.
For endpoint e, grid_sample reads source coordinate x + eye_i - eye_e.
"""
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn.functional as F
from .targets import resample
from .temporal import integrate


class EyeMotion:
    def __init__(self, positions):
        if positions.ndim != 2 or positions.shape[1] != 2 or not torch.isfinite(positions).all():
            raise ValueError("Eye positions must be finite [time,2] coordinates in x,y pixels")
        self.positions = positions
        self._grids = {}

    @classmethod
    def from_config(cls, config, schedule):
        m = config["motion"]
        if m.get("path"):
            path = Path(m["path"])
            if path.suffix == ".npz":
                with np.load(path, allow_pickle=False) as data:
                    positions = torch.tensor(data["positions_px"].copy(), device=schedule.times.device, dtype=torch.float32)
                    times = data["times_s"].copy()
            else:
                data = json.loads(path.read_text())
                positions = torch.tensor(data["positions_px"], device=schedule.times.device, dtype=torch.float32)
                times = data["times_s"]
            positions = resample(positions, times, schedule.times)
        else:
            velocity = torch.tensor(m["velocity_px_s"], device=schedule.times.device, dtype=torch.float32)
            positions = (schedule.times-schedule.times[0]).float()[:, None]*velocity
        return cls(positions)

    def warp(self, video, absolute_indices):
        n, c, d, h, w = video.shape
        key = (tuple(int(x) for x in absolute_indices), h, w, str(video.device), video.dtype)
        if key not in self._grids:
            positions = self.positions[absolute_indices].to(video)
            delta = positions-positions[-1]
            y, x = torch.meshgrid(torch.arange(h, device=video.device, dtype=video.dtype),
                                  torch.arange(w, device=video.device, dtype=video.dtype), indexing="ij")
            sx = x[None]+delta[:, 0, None, None]
            sy = y[None]+delta[:, 1, None, None]
            valid = ((sx >= 0) & (sx <= w-1) & (sy >= 0) & (sy <= h-1)).all(0)
            grid = torch.stack([2*(sx+0.5)/w-1, 2*(sy+0.5)/h-1], dim=-1)
            self._grids[key] = (grid, valid[None, None], bool((delta == 0).all()))
        grid, valid, stationary = self._grids[key]
        warped = video if stationary else F.grid_sample(video.reshape(n, c*d, h, w), grid,
                                                        mode="bilinear", padding_mode="zeros", align_corners=False).reshape_as(video)
        return warped, valid.to(video.dtype)

    def perceive(self, video, durations, window, indices):
        if (self.positions[indices] == self.positions[indices][0]).all():
            return integrate(video, durations, window), video.new_ones((len(video), 1, 1, *video.shape[-2:]))
        outputs, masks = [], []
        for end in range(1, len(video)+1):
            start = max(0, end-window)
            warped, valid = self.warp(video[start:end], indices[start:end])
            weights = durations[start:end].to(video)
            outputs.append((warped*weights[:, None, None, None, None]).sum(0)/weights.sum())
            masks.append(valid)
        return torch.stack(outputs), torch.stack(masks)
