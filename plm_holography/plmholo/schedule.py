"""One physical PLM update carries one color, one phase, and one exposure."""
from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class Schedule:
    colors: torch.Tensor
    times: torch.Tensor
    durations: torch.Tensor
    gains: torch.Tensor
    window: int
    names: tuple

    @classmethod
    def from_config(cls, config, device="cpu"):
        s, o = config["schedule"], config["optics"]
        names = tuple(o["colors"])
        color = [names.index(c) for c in s["order"]]*s["cycles"]
        durations = s.get("durations_s")
        dt = np.full(len(color), 1/s["rate_hz"]) if durations is None else np.asarray(durations, dtype=float)
        if dt.shape != (len(color),) or not np.isfinite(dt).all() or (dt <= 0).any():
            raise ValueError("durations_s must give one finite positive duration per physical subframe")
        return cls(torch.tensor(color, device=device, dtype=torch.long),
                   torch.tensor(dt.cumsum()-dt/2, device=device, dtype=torch.float64),
                   torch.tensor(dt, device=device, dtype=torch.float32),
                   torch.tensor(s["gains"], device=device, dtype=torch.float32),
                   s["window_cycles"]*len(names), names)

    def __len__(self):
        return len(self.colors)

    def dense(self, raw_intensity, indices):
        colors = self.colors[indices]
        selector = torch.nn.functional.one_hot(colors, len(self.names)).to(raw_intensity.dtype)
        return raw_intensity[:, None]*selector[:, :, None, None, None]*self.gains[None, :, None, None, None]

    def duty(self, index):
        # Actual exposure fraction of this color in its RGB cycle.
        c = len(self.names)
        start = (index//c)*c
        return self.durations[index]/self.durations[start:start+c].sum()
