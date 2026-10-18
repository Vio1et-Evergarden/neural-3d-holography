"""Losses operate on integrated linear intensities, with fixed gains."""
import torch


def squared_error(prediction, target, kind="intensity_mse"):
    if kind == "amplitude_mse":
        # epsilon keeps the derivative finite at black pixels.
        prediction = torch.sqrt(prediction.clamp_min(1e-12))
        target = torch.sqrt(target.clamp_min(1e-12))
    return (prediction-target).square()


def loss_value(prediction, target, mask=None, kind="intensity_mse"):
    error = squared_error(prediction, target, kind)
    if mask is None:
        return error.mean()
    mask = torch.broadcast_to(mask, error.shape)
    denominator = mask.sum(dim=(-2, -1))
    if (denominator == 0).any():
        raise ValueError("Motion leaves no shared valid pixels in the evaluation region")
    return ((error*mask).sum(dim=(-2, -1))/denominator).mean()
