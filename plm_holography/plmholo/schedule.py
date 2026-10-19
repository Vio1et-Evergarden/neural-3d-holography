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
    stride: int | None = None

    @classmethod
    def from_config(cls, config, device="cpu"):
        s, o = config["schedule"], config["optics"]
        names = tuple(o["colors"])
        color = [names.index(c) for c in s["order"]]*s["cycles"]
        durations = s.get("durations_s")
        dt = np.full(len(color), 1/s["rate_hz"]) if durations is None else np.asarray(durations, dtype=float)
        if dt.shape != (len(color),) or not np.isfinite(dt).all() or (dt <= 0).any():
            raise ValueError("durations_s must give one finite positive duration per physical subframe")
        stride_cycles = s.get("stride_cycles")
        stride = None if stride_cycles is None else stride_cycles*len(names)
        return cls(torch.tensor(color, device=device, dtype=torch.long),
                   torch.tensor(dt.cumsum()-dt/2, device=device, dtype=torch.float64),
                   torch.tensor(dt, device=device, dtype=torch.float32),
                   torch.tensor(s["gains"], device=device, dtype=torch.float32),
                   s["window_cycles"]*len(names), names, stride)

    def __len__(self):
        return len(self.colors)

    def optimization_ends(self):
        """Exclusive window ends. Explicit stride starts at the first full window.

        Always include the last exposure, even if the final step is shorter.
        None retains legacy per-exposure endpoints, including startup prefixes.
        """
        if self.stride is None:
            return list(range(1, len(self)+1))
        ends = list(range(min(self.window, len(self)), len(self)+1, self.stride))
        if ends[-1] != len(self):
            ends.append(len(self))
        return ends

    def endpoint_indices(self):
        return torch.tensor(self.optimization_ends(), device=self.colors.device, dtype=torch.long)-1

    def causal_blocks(self):
        """Disjoint new variables; overlapping history is never re-optimized."""
        start = 0
        for end in self.optimization_ends():
            yield start, end
            start = end

    def dense(self, raw_intensity, indices):
        colors = self.colors[indices]
        selector = torch.nn.functional.one_hot(colors, len(self.names)).to(raw_intensity.dtype)
        return raw_intensity[:, None]*selector[:, :, None, None, None]*self.gains[None, :, None, None, None]

    def duty(self, index):
        # Actual exposure fraction of this color in its RGB cycle.
        c = len(self.names)
        start = (index//c)*c
        return self.durations[index]/self.durations[start:start+c].sum()
