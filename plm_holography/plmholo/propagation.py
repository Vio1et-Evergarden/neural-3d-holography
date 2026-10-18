"""Phase-only illumination -> padded angular-spectrum propagation -> output crop."""
import math
import torch
import torch.nn.functional as F


def center_crop(x, shape):
    h, w = x.shape[-2:]
    oh, ow = shape
    return x[..., (h-oh)//2:(h-oh)//2+oh, (w-ow)//2:(w-ow)//2+ow]


class ASM:
    def __init__(self, optics):
        self.shape = tuple(optics["shape"])
        self.crop = tuple(optics.get("crop") or self.shape)
        self.grid = tuple(optics.get("computational_shape") or [optics.get("pad_factor", 2)*n for n in self.shape])
        self.pitch = tuple(optics["pitch_m"])
        self.wavelengths = tuple(optics["wavelengths_m"])
        self.depths = tuple(optics["depths_m"])
        self.frequency_grid = optics.get("frequency_grid", "fft")
        self.na = optics.get("numerical_aperture")
        self.batch_size = optics.get("propagation_batch", 4)
        self._cache = {}
        self.calls = 0
        self.phase_propagations = 0

    def kernel(self, color, device, dtype):
        key = (color, str(device), dtype, self.grid, self.pitch, self.depths,
               self.wavelengths, self.frequency_grid, self.na)
        if key not in self._cache:
            ny, nx = self.grid
            dy, dx = self.pitch
            wavelength = self.wavelengths[color]
            # Build accurate optical phases in float64, then cast the final kernel.
            if self.frequency_grid == "fft":
                fy = torch.fft.fftfreq(ny, dy, device=device, dtype=torch.float64)
                fx = torch.fft.fftfreq(nx, dx, device=device, dtype=torch.float64)
            else:
                # Compatibility with the original inclusive-endpoint, centered grid.
                fy = torch.linspace(-1/(2*dy), 1/(2*dy), ny, device=device, dtype=dtype)
                fx = torch.linspace(-1/(2*dx), 1/(2*dx), nx, device=device, dtype=dtype)
            yy, xx = torch.meshgrid(fy, fx, indexing="ij")
            f2 = xx.square() + yy.square()
            mask = f2 <= 1/wavelength**2
            if self.na is not None:
                mask = mask & (f2 <= (self.na/wavelength)**2)
            kz = 2*math.pi*torch.sqrt(torch.clamp(1/wavelength**2-f2, min=0))
            z = torch.tensor(self.depths, device=device, dtype=kz.dtype)[:, None, None]
            h = mask * torch.exp(1j*z*kz)
            if self.frequency_grid == "legacy":
                h = torch.fft.ifftshift(h, dim=(-2, -1))
            self._cache[key] = h.to(torch.complex128 if dtype == torch.float64 else torch.complex64)
        return self._cache[key]

    def __call__(self, phases, colors):
        if phases.ndim != 3 or tuple(phases.shape[-2:]) != self.shape:
            raise ValueError("Phases must have shape [physical_subframe,H,W]")
        if len(colors) != len(phases):
            raise ValueError("Each physical subframe needs a color")
        self.calls += 1
        self.phase_propagations += len(phases)
        h, w = self.shape
        ph, pw = self.grid[0]-h, self.grid[1]-w
        outputs = []
        for start in range(0, len(phases), self.batch_size):
            p = phases[start:start+self.batch_size]
            field = F.pad(torch.exp(1j*p), (pw//2, pw-pw//2, ph//2, ph-ph//2))
            spectrum = torch.fft.fft2(field, norm="ortho")
            kernels = torch.stack([self.kernel(int(c), p.device, p.dtype)
                                   for c in colors[start:start+self.batch_size]])
            propagated = torch.fft.ifft2(spectrum[:, None]*kernels, norm="ortho")
            # Undo input padding first, then crop the requested observation ROI.
            propagated = propagated[..., ph//2:ph//2+h, pw//2:pw//2+w]
            outputs.append(center_crop(propagated, self.crop))
        return torch.cat(outputs, dim=0)

    def counts(self):
        return {"calls": self.calls, "phase_propagations": self.phase_propagations,
                "depth_propagations": self.phase_propagations*len(self.depths)}
