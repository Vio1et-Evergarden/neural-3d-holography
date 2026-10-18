"""Exposure-weighted causal box integration; no phase interference across time."""
import torch


def integrate(video, durations, window):
    weights = durations.to(video.dtype)
    sums = (video*weights[:, None, None, None, None]).cumsum(0)
    total = weights.cumsum(0)
    # Prefix zero makes an absent history explicit, without wrapping the sequence.
    padded = torch.cat([torch.zeros_like(sums[:1]), sums], dim=0)
    weight_prefix = torch.cat([total.new_zeros(1), total])
    ends = torch.arange(1, len(video)+1, device=video.device)
    starts = (ends-window).clamp_min(0)
    return (padded[ends]-padded[starts])/(weight_prefix[ends]-weight_prefix[starts])[:, None, None, None, None]


class TemporalModel:
    def __init__(self, schedule, motion=None):
        self.schedule = schedule
        self.motion = motion

    def perceive(self, video, indices):
        dt = self.schedule.durations[indices]
        if self.motion is None:
            perceived = integrate(video, dt, self.schedule.window)
            mask = video.new_ones((len(video), 1, 1, *video.shape[-2:]))
            return perceived, mask
        return self.motion.perceive(video, dt, self.schedule.window, indices)

    def reconstruct(self, raw_intensity, indices):
        return self.perceive(self.schedule.dense(raw_intensity, indices), indices)

    def final(self, video, indices):
        """One window endpoint, without computing its unused partial endpoints."""
        indices = indices[-self.schedule.window:]
        video = video[-len(indices):]
        mask = video.new_ones((1, 1, *video.shape[-2:]))
        if self.motion is not None:
            video, mask = self.motion.warp(video, indices)
        weights = self.schedule.durations[indices].to(video)
        return (video*weights[:, None, None, None, None]).sum(0)/weights.sum(), mask

    def reconstruct_final(self, raw_intensity, indices):
        return self.final(self.schedule.dense(raw_intensity, indices), indices)
