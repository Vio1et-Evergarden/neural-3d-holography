"""Physical code -> phase LUT; hard forward propagation with optional surrogates."""
import csv
import math
import torch


class PhaseLUT:
    def __init__(self, names, path=None, phases=None, codes=None):
        self.names = tuple(names)
        if phases is not None:
            self.phases = torch.as_tensor(phases, dtype=torch.float64)
            self.codes = torch.as_tensor(codes, dtype=torch.long)
        elif path:
            with open(path, newline="") as handle:
                rows = list(csv.DictReader(handle))
            tables, identifiers = [], []
            for color in names:
                selected = [r for r in rows if r["color"] == color]
                tables.append([float(r["phase_rad"]) for r in selected])
                identifiers.append([int(r["code"]) for r in selected])
            if any(len(row) != 16 for row in tables):
                raise ValueError("LUT CSV needs 16 rows for each configured color")
            self.phases = torch.tensor(tables, dtype=torch.float64)
            self.codes = torch.tensor(identifiers, dtype=torch.long)
        else:
            self.phases = (torch.arange(16, dtype=torch.float64)*2*math.pi/16).repeat(len(names), 1)
            self.codes = torch.arange(16).repeat(len(names), 1)
        if self.phases.shape != (len(names), 16) or self.codes.shape != self.phases.shape:
            raise ValueError("LUT shape must be [colors,16]")
        if not torch.isfinite(self.phases).all():
            raise ValueError("Non-finite LUT phase")
        if any(sorted(row.tolist()) != list(range(16)) for row in self.codes):
            raise ValueError("Each color must contain physical codes 0..15 exactly once")
        # Keep physical IDs, not a sorted-by-phase table. First row breaks ties.

    def hard(self, phase, colors):
        outputs, codes = [], []
        for p, color in zip(phase, colors):
            table = self.phases[int(color)].to(p)
            ids = self.codes[int(color)].to(device=p.device)
            delta = table[:, None, None]-p[None]
            delta = torch.atan2(torch.sin(delta), torch.cos(delta))
            index = delta.abs().argmin(0)
            outputs.append(table[index])
            codes.append(ids[index])
        return torch.stack(outputs), torch.stack(codes)

    def decode(self, codes, colors, dtype=torch.float32):
        values = []
        for code, color in zip(codes, colors):
            ids = self.codes[int(color)].to(code.device)
            match = code[None] == ids[:, None, None]
            if not match.any(0).all():
                raise ValueError("Saved code map contains an unknown physical code")
            values.append(self.phases[int(color)].to(device=code.device, dtype=dtype)[match.long().argmax(0)])
        return torch.stack(values)

    def forward(self, phase, colors, method, tau=None):
        if method in ("continuous", "post_quantized"):
            return phase
        hard, _ = self.hard(phase.detach(), colors)
        if method == "ste":
            return hard + (phase-phase.detach())
        if method != "smooth" or tau is None or tau <= 0:
            raise ValueError("Smooth quantization needs a positive smoothing width")
        result = []
        for p, color, actual in zip(phase, colors, hard):
            table = self.phases[int(color)].to(p)
            delta = table[:, None, None]-p[None]
            delta = torch.atan2(torch.sin(delta), torch.cos(delta))
            weights = torch.softmax(-delta.square()/(2*tau*tau), dim=0)
            soft = p+(weights*delta).sum(0)
            result.append(actual+(soft-soft.detach()))
        return torch.stack(result)

    def evaluated(self, phase, colors, method):
        if method == "continuous":
            return phase.detach(), None
        return self.hard(phase.detach(), colors)
