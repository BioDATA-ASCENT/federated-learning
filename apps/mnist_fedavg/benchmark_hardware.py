"""Compare local CPU throughput for the MNIST FedAvg model across machines."""

from __future__ import annotations

import argparse
import os
import platform
import statistics
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from mnist_fedavg.task import MLP, train

APP_CONFIG = tomllib.loads(
    Path(__file__).with_name("pyproject.toml").read_text()
)["tool"]["flwr"]["app"]["config"]


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _thread_counts(value: str) -> list[int]:
    try:
        result = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use comma-separated positive integers") from exc
    if not result or any(count < 1 for count in result):
        raise argparse.ArgumentTypeError("thread counts must be positive integers")
    return list(dict.fromkeys(result))


def _default_thread_counts() -> list[int]:
    logical_cpus = os.cpu_count() or 1
    return list(dict.fromkeys(min(count, logical_cpus) for count in (1, 2, 4, 8)))


def _processor_name() -> str:
    if sys.platform == "darwin":
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            check=False,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    elif platform.system() == "Linux":
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            for line in cpuinfo.read_text().splitlines():
                if line.startswith(("model name", "Hardware", "Processor")):
                    return line.split(":", maxsplit=1)[-1].strip()
    return platform.processor() or "not reported"


def _run_once(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    momentum: float,
    seed: int,
) -> float:
    torch.manual_seed(seed)
    model = MLP()
    loader = DataLoader(
        TensorDataset(inputs, labels),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    started = time.perf_counter()
    train(model, loader, epochs, learning_rate, momentum)
    return time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the MNIST FedAvg CPU training workload locally."
    )
    parser.add_argument(
        "--threads",
        type=_thread_counts,
        default=_default_thread_counts(),
        help="comma-separated PyTorch CPU thread counts (default: 1,2,4,8 up to available CPUs)",
    )
    parser.add_argument(
        "--repetitions",
        type=_positive_int,
        default=3,
        help="measured runs per thread count, after one warm-up (default: 3)",
    )
    parser.add_argument(
        "--samples",
        type=_positive_int,
        default=10_000,
        help="synthetic examples per run (default: 10000)",
    )
    args = parser.parse_args()

    seed = int(APP_CONFIG["seed"])
    torch.manual_seed(seed)
    inputs = torch.randn(args.samples, 784)
    labels = torch.randint(0, 10, (args.samples,))
    batch_size = int(APP_CONFIG["batch-size"])
    epochs = int(APP_CONFIG["local-epochs"])

    print("Local CPU hardware check (synthetic MNIST-shaped data)")
    print(f"Machine: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Processor: {_processor_name()}")
    print(f"Logical CPUs: {os.cpu_count() or 'unknown'}")
    print(f"PyTorch: {torch.__version__}")
    print(
        f"Workload: MLP from MNIST FedAvg, {args.samples:,} examples, "
        f"batch size {batch_size}, {epochs} epoch(s)"
    )
    print("Results (median of measured runs; warm-up excluded):")
    print(f"{'Threads':>7}  {'Seconds':>9}  {'Examples/sec':>13}")

    for thread_count in args.threads:
        torch.set_num_threads(thread_count)
        _run_once(
            inputs,
            labels,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=float(APP_CONFIG["lr"]),
            momentum=float(APP_CONFIG["momentum"]),
            seed=seed,
        )
        durations = [
            _run_once(
                inputs,
                labels,
                batch_size=batch_size,
                epochs=epochs,
                learning_rate=float(APP_CONFIG["lr"]),
                momentum=float(APP_CONFIG["momentum"]),
                seed=seed,
            )
            for _ in range(args.repetitions)
        ]
        median_seconds = statistics.median(durations)
        examples_per_second = args.samples * epochs / median_seconds
        print(
            f"{thread_count:>7}  {median_seconds:>9.2f}  "
            f"{examples_per_second:>13,.0f}"
        )

    print(
        "\nCompare reports from different machines using the same sample count, "
        "thread counts, and PyTorch version. This is a CPU workload proxy, not "
        "a simulation of another processor or a full federation run."
    )


if __name__ == "__main__":
    main()
