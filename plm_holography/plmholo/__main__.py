"""Command line interface. Run from the plm_holography directory."""
import argparse
import csv
from copy import deepcopy
import json
from pathlib import Path
import time
from .config import load_config, validate, ROOT


def run_one(config, verbose=True):
    from .optimization import prepare, optimize
    from .artifacts import save_run
    config = validate(config)
    if (Path(config["output"])/"metadata.json").exists():
        raise FileExistsError(f"Output already contains a run: {config['output']}")
    ctx = prepare(config)
    print(f"{config['name']}: {config['optimization']['method']}, {config['quantization']['method']}, "
          f"{len(ctx.schedule)} subframes, {ctx.device}", flush=True)
    result = optimize(ctx, verbose=verbose)
    meta = save_run(ctx, result)
    print(f"  full-window MSE={meta['final']['full_window_intensity_mse']}; "
          f"optimization={meta['elapsed_optimization_seconds']:.2f}s; {config['output']}", flush=True)
    return meta


def benchmark(config, resume=False):
    from .artifacts import json_write
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    started = time.perf_counter()
    for scene in config["benchmark"]["scenes"]:
        for seed in config["benchmark"]["seeds"]:
            for method in config["benchmark"]["methods"]:
                c = deepcopy(config)
                c["seed"] = seed
                c["target"] = {"kind": "synthetic", "scene": scene, "path": None}
                c["optimization"]["method"] = method
                c["name"] = f"{scene}_seed{seed}_{method}"
                c["output"] = str(output/c["name"])
                c = validate(c)
                saved = Path(c["output"])/"metadata.json"
                if resume and saved.exists():
                    if json.loads(saved.with_name("config.json").read_text()) != c:
                        raise ValueError(f"Cannot resume a run with changed configuration: {saved.parent}")
                    meta = json.loads(saved.read_text())
                else:
                    meta = run_one(c, verbose=False)
                rows.append({"scene": scene, "seed": seed, "method": method,
                             "quantization": c["quantization"]["method"],
                             "intensity_mse": meta["final"]["full_window_intensity_mse"],
                             "initial_intensity_mse": meta["initial"]["full_window_intensity_mse"],
                             "psnr_db": meta["final"]["full_window_psnr_db_peak1"],
                             "temporal_error_rms": meta["final"]["temporal_error_difference_rms"],
                             "seconds": meta["elapsed_optimization_seconds"],
                             "phase_propagations": meta["propagation_counts"]["phase_propagations"],
                             "depth_propagations": meta["propagation_counts"]["depth_propagations"],
                             "initial_phase_sha256": meta["initial_phase_sha256"],
                             "run": c["output"]})
                with (output/"summary.csv").open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
    groups = []
    import numpy as np
    for scene in config["benchmark"]["scenes"]:
        for method in config["benchmark"]["methods"]:
            group = [r for r in rows if r["scene"] == scene and r["method"] == method]
            values = [r["intensity_mse"] for r in group if r["intensity_mse"] is not None]
            groups.append({"scene": scene, "method": method,
                           "mse_mean": float(np.mean(values)) if values else None,
                           "mse_std": float(np.std(values)) if values else None,
                           "seconds_mean": float(np.mean([r["seconds"] for r in group]))})
    json_write(output/"summary.json", {"groups": groups, "runs": rows,
                                      "elapsed_wall_seconds": time.perf_counter()-started})
    lines = ["# CPU benchmark", "", "Fixed gains; linear intensity MSE over full windows. Paired seeds.", "",
             "| Scene | Method | MSE mean | MSE std | Seconds mean |", "|---|---|---:|---:|---:|"]
    for row in groups:
        lines.append(f"| {row['scene']} | {row['method']} | {row['mse_mean']} | {row['mse_std']} | {row['seconds_mean']:.2f} |")
    lines += ["", "No universal ranking is assumed. Global and causal methods have different information access.",
              "Runtime covers optimization, not plotting. Propagation counts include checkpoint evaluation."]
    (output/"REPORT.md").write_text("\n".join(lines)+"\n")
    print(f"Benchmark report: {output/'REPORT.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("optimize", "benchmark"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        if name == "benchmark":
            p.add_argument("--resume", action="store_true", help="Skip completed runs only when their configs match")
    p = sub.add_parser("evaluate")
    p.add_argument("--run", required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.command == "evaluate":
        from .evaluation import evaluate_run
        run = Path(args.run).resolve()
        if not run.is_relative_to(ROOT):
            parser.error("Run directory must be inside plm_holography/")
        result = evaluate_run(run, args.device)
        print(json.dumps(result, indent=2, allow_nan=False))
    else:
        config = load_config(args.config)
        if args.command == "optimize":
            run_one(config)
        else:
            benchmark(config, args.resume)


if __name__ == "__main__":
    main()
