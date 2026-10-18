"""Four temporal objectives sharing the same propagation and discrete evaluation."""
from dataclasses import dataclass
import math
import time
import torch
from .config import validate
from .propagation import ASM
from .schedule import Schedule
from .targets import load_target
from .quantization import PhaseLUT
from .temporal import TemporalModel
from .losses import loss_value


@dataclass
class Context:
    config: dict
    schedule: Schedule
    prop: ASM
    lut: PhaseLUT
    temporal: TemporalModel
    target: torch.Tensor
    device: torch.device


def prepare(config):
    config = validate(config)
    device_name = config["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available; choose device=cpu")
    device = torch.device(device_name)
    torch.set_num_threads(config["threads"])
    schedule = Schedule.from_config(config, device)
    target = load_target(config, schedule, device)
    motion = None
    if config["motion"]["enabled"]:
        from .motion import EyeMotion
        motion = EyeMotion.from_config(config, schedule)
    return Context(config, schedule, ASM(config["optics"]),
                   PhaseLUT(schedule.names, config["quantization"]["lut_path"]),
                   TemporalModel(schedule, motion), target, device)


def initial_phases(ctx):
    # CPU generator makes the initial tensor identical before CPU/CUDA transfer.
    generator = torch.Generator(device="cpu").manual_seed(ctx.config["seed"])
    return ((torch.rand((len(ctx.schedule), *ctx.config["optics"]["shape"]), generator=generator)*2-1)*math.pi).to(ctx.device)


def optimize(ctx, verbose=True):
    cfg, schedule = ctx.config, ctx.schedule
    opt, quant = cfg["optimization"], cfg["quantization"]
    method, quant_method = opt["method"], quant["method"]
    initial = initial_phases(ctx)
    indices_all = torch.arange(len(schedule), device=ctx.device)
    started = time.perf_counter()
    history = []
    with torch.no_grad():
        initial_evaluated, _ = ctx.lut.evaluated(initial, schedule.colors, quant_method)
        initial_raw = ctx.prop(initial_evaluated, schedule.colors).abs().square()
        target_perceived, target_mask = ctx.temporal.perceive(ctx.target, indices_all)

    def solve(starting, indices, objective, stage):
        latent = starting.detach().clone().requires_grad_(True)
        colors = schedule.colors[indices]
        factory = torch.optim.Adam if opt["optimizer"] == "adam" else torch.optim.SGD
        optimizer = factory([latent], lr=opt["lr"])
        with torch.no_grad():
            actual, _ = ctx.lut.evaluated(latent, colors, quant_method)
            raw = ctx.prop(actual, colors).abs().square()
            best_loss = objective(raw).item()
            best_phase, best_raw = actual.clone(), raw.clone()
        if not math.isfinite(best_loss):
            raise FloatingPointError("Non-finite initial objective")
        history.append(dict(stage=stage, update=0, training_loss=best_loss,
                            evaluated_loss=best_loss, best_loss=best_loss, gradient_rms=0.0))
        for step in range(1, opt["iterations"]+1):
            fraction = (step-1)/max(opt["iterations"]-1, 1)
            tau = 2*math.pi/16*quant["tau_start_steps"]*(quant["tau_end_steps"]/quant["tau_start_steps"])**fraction
            optimizer.zero_grad()
            forward_phase = ctx.lut.forward(latent, colors, quant_method, tau)
            training_loss = objective(ctx.prop(forward_phase, colors).abs().square())
            training_loss.backward()
            if not torch.isfinite(training_loss) or not torch.isfinite(latent.grad).all():
                raise FloatingPointError(f"Non-finite gradient/loss in {stage}, step {step}")
            gradient_rms = latent.grad.square().mean().sqrt().item()
            optimizer.step()
            with torch.no_grad():
                actual, _ = ctx.lut.evaluated(latent, colors, quant_method)
                raw = ctx.prop(actual, colors).abs().square()
                score = objective(raw).item()
                if not math.isfinite(score):
                    raise FloatingPointError("Non-finite evaluated loss")
                if score < best_loss:
                    best_loss = score
                    best_phase, best_raw = actual.clone(), raw.clone()
            history.append(dict(stage=stage, update=step, training_loss=training_loss.item(),
                                evaluated_loss=score, best_loss=best_loss, gradient_rms=gradient_rms))
            if verbose and method == "global" and opt["log_every"] and (step == 1 or step % opt["log_every"] == 0):
                print(f"  {stage} {step}/{opt['iterations']}: loss={score:.7g}, best={best_loss:.7g}", flush=True)
        return best_phase.detach(), best_raw.detach()

    kind = opt["loss"]
    if method == "global":
        def objective(raw):
            perceived, mask = ctx.temporal.reconstruct(raw, indices_all)
            return loss_value(perceived, target_perceived, mask*target_mask, kind)
        final_phase, final_raw = solve(initial, indices_all, objective, "global")
    elif method == "joint":
        phases, raws = [], []
        for begin in range(0, len(schedule), schedule.window):
            end = min(begin+schedule.window, len(schedule))
            indices = indices_all[begin:end]
            with torch.no_grad():
                goal, target_valid = ctx.temporal.final(ctx.target[indices], indices)
            def objective(raw):
                perceived, valid = ctx.temporal.reconstruct_final(raw, indices)
                return loss_value(perceived, goal, valid*target_valid, kind)
            phase, raw = solve(initial[indices], indices, objective, f"group_{begin}")
            phases.append(phase)
            raws.append(raw)
        final_phase, final_raw = torch.cat(phases), torch.cat(raws)
    else:
        phases, raws = [], []
        for frame in range(len(schedule)):
            indices = indices_all[frame:frame+1]
            if method == "independent":
                color = int(schedule.colors[frame])
                coefficient = schedule.gains[color]*schedule.duty(frame)
                def objective(raw):
                    return loss_value(coefficient*raw[0], ctx.target[frame, color], kind=kind)
            else:
                begin = max(0, frame-schedule.window+1)
                window_indices = indices_all[begin:frame+1]
                # Frozen tensors, not aliases to optimizer parameters or their graphs.
                past = torch.cat(raws[begin:frame], dim=0) if frame > begin else initial_raw[:0]
                past = past.detach()
                goal, target_valid = target_perceived[frame], target_mask[frame]
                def objective(raw):
                    combined = torch.cat([past, raw], dim=0)
                    perceived, valid = ctx.temporal.reconstruct_final(combined, window_indices)
                    return loss_value(perceived, goal, valid*target_valid, kind)
            phase, raw = solve(initial[indices], indices, objective, f"frame_{frame}")
            phases.append(phase)
            raws.append(raw)
            if verbose and (frame == 0 or (frame+1) % schedule.window == 0 or frame+1 == len(schedule)):
                print(f"  committed {frame+1}/{len(schedule)} phases", flush=True)
        final_phase, final_raw = torch.cat(phases), torch.cat(raws)

    with torch.no_grad():
        perceived, mask = ctx.temporal.reconstruct(final_raw, indices_all)
        initial_perceived, _ = ctx.temporal.reconstruct(initial_raw, indices_all)
        codes = None
        if quant_method != "continuous":
            _, codes = ctx.lut.hard(final_phase, schedule.colors)
        positions = ctx.temporal.motion.positions if ctx.temporal.motion is not None else torch.zeros((len(schedule), 2), device=ctx.device)
    return dict(phase=final_phase, codes=codes, raw_intensity=final_raw,
                perceived=perceived, target=ctx.target, target_perceived=target_perceived,
                mask=mask*target_mask, initial_phase=initial,
                initial_perceived=initial_perceived, history=history,
                elapsed_seconds=time.perf_counter()-started,
                propagation_counts=ctx.prop.counts(), motion_positions=positions,
                optimized_phase_updates=len(schedule)*opt["iterations"])
