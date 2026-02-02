"""Shared utilities for distributed PyTorch benchmarks."""

import os
import json
import socket
import time
import statistics
import argparse
from pathlib import Path

import torch
import torch.distributed as dist


def init_ranks():
    """Detect and return (local_rank, world_rank, world_size) from the
    available environment.  Supports mpi4py, torchrun, OpenMPI env vars,
    and Cray MPICH (PMI)."""
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        shmem_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)

        local_rank = shmem_comm.Get_rank()
        world_size = comm.Get_size()
        world_rank = comm.Get_rank()

        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = comm.bcast(
                socket.gethostbyname(socket.gethostname()), root=0
            )
        if "MASTER_PORT" not in os.environ:
            if world_rank == 0:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(("", 0))
                free_port = sock.getsockname()[1]
                sock.close()
            else:
                free_port = None
            free_port = comm.bcast(free_port, root=0)
            os.environ["MASTER_PORT"] = str(free_port)
        return local_rank, world_rank, world_size

    except ImportError:
        pass

    if "LOCAL_RANK" in os.environ:
        return (
            int(os.environ["LOCAL_RANK"]),
            int(os.environ["RANK"]),
            int(os.environ["WORLD_SIZE"]),
        )

    if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        return (
            int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]),
            int(os.environ["OMPI_COMM_WORLD_RANK"]),
            int(os.environ["OMPI_COMM_WORLD_SIZE"]),
        )

    if "PMI_RANK" in os.environ:
        return (
            int(os.environ["PMI_LOCAL_RANK"]),
            int(os.environ["PMI_RANK"]),
            int(os.environ["PMI_SIZE"]),
        )

    raise RuntimeError(
        "Could not determine ranks — run with torchrun, mpirun, or MPI + mpi4py"
    )


def init_dist(backend, local_rank, world_rank, world_size):
    """Initialise the process group and set the CUDA device when using NCCL."""
    dist.init_process_group(backend, rank=world_rank, world_size=world_size)
    if backend == "nccl":
        torch.cuda.set_device(local_rank)


def get_nccl_version_str():
    """Return NCCL version as a dotted string, or 'N/A'."""
    try:
        return ".".join(map(str, torch.cuda.nccl.version()))
    except Exception:
        return "N/A"


def print_env_info(world_rank, world_size, backend):
    """Print system/runtime information from rank 0."""
    if world_rank != 0:
        return

    cuda_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
    gpu_info = "N/A"
    gpu_memory = "N/A"
    if torch.cuda.is_available() and cuda_devices > 0:
        gpu_info = torch.cuda.get_device_name(0)
        gpu_memory = f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB"

    print("=" * 60)
    print("DISTRIBUTED COMMUNICATION BENCHMARK")
    print(f"Hostname         : {socket.gethostname()}")
    print(f"World Size       : {world_size}")
    print(f"Backend          : {backend}")
    print(f"Conda Env        : {os.environ.get('CONDA_DEFAULT_ENV', 'N/A')}")
    print(f"PyTorch Version  : {torch.__version__}")
    print(f"CUDA Version     : {torch.version.cuda}")
    print(f"CUDA Devices     : {cuda_devices}")
    print(f"GPU Model        : {gpu_info}")
    print(f"GPU Memory       : {gpu_memory}")
    if backend == "nccl":
        print(f"NCCL Version     : {get_nccl_version_str()}")
    print("=" * 60)


def make_tensor(size, backend, local_rank):
    """Create a float32 tensor of *size* elements on the appropriate device."""
    device = torch.device("cuda", local_rank) if backend == "nccl" else torch.device("cpu")
    return torch.ones(size, device=device)


def benchmark_op(op_fn, tensor, backend, warmup=5, iters=20):
    """Run *op_fn(tensor)* with warmup, return (warmup_times, benchmark_times).

    Times are measured on rank 0 only; other ranks get empty lists.
    Uses perf_counter with barrier+cuda sync for accurate timing.
    """
    use_cuda = backend == "nccl"

    def _timed(t):
        if use_cuda:
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()
        start = time.perf_counter()
        op_fn(t)
        if use_cuda:
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()
        return time.perf_counter() - start

    warmup_times = []
    for _ in range(warmup):
        elapsed = _timed(tensor.clone())
        if dist.get_rank() == 0:
            warmup_times.append(elapsed)

    bench_times = []
    for _ in range(iters):
        elapsed = _timed(tensor.clone())
        if dist.get_rank() == 0:
            bench_times.append(elapsed)

    return warmup_times, bench_times


def compute_bandwidth(size_bytes, time_sec, world_size, collective="all_reduce"):
    """Return algorithm bandwidth in GB/s (using 1024^3).

    For all_reduce the data moved is 2*(N-1)/N * size  (ring algorithm).
    For broadcast the data moved is size (simple).
    For all_gather / reduce the data moved is (N-1)/N * size.
    """
    n = world_size
    if collective == "all_reduce":
        factor = 2.0 * (n - 1) / n
    elif collective == "broadcast":
        factor = 1.0
    elif collective in ("all_gather", "reduce", "reduce_scatter"):
        factor = (n - 1) / n
    else:
        factor = 1.0
    algobw = (size_bytes * factor / (1024**3)) / time_sec
    return algobw


def default_tensor_sizes():
    """Return a default list of 1-D tensor lengths spanning ~4 KB to ~400 MB."""
    return [10**i for i in range(3, 9)]


def common_arg_parser(description="Distributed collective benchmark"):
    """Return an ArgumentParser pre-loaded with common flags."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--backend", choices=["nccl", "gloo", "mpi"], default="nccl",
                        help='Distributed backend ("nccl" for GPU, "gloo" for CPU)')
    parser.add_argument(
        "--tensor_sizes",
        nargs="+",
        type=int,
        default=None,
        help="1-D tensor sizes to sweep (default: 1e3 … 1e8)",
    )
    parser.add_argument("--warmup", type=int, default=5,
                        help="Number of warmup iterations")
    parser.add_argument("--iters", type=int, default=20,
                        help="Number of benchmark iterations")
    parser.add_argument("--output", type=str, default="results/benchmark_results.log",
                        help="Output CSV log file")
    return parser


def save_result(record, output_path):
    """Append a dict as a single JSON line to *output_path*."""
    if output_path is None:
        return
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def init_csv_log(output_path, backend, world_size, warmup, iters, tensor_sizes):
    """Write a metadata header and CSV column header to *output_path*."""
    if output_path is None:
        return
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    cuda_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
    gpu_info = "N/A"
    gpu_memory = "N/A"
    if torch.cuda.is_available() and cuda_devices > 0:
        gpu_info = torch.cuda.get_device_name(0)
        gpu_memory = f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB"

    nccl_version = get_nccl_version_str()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    with open(output_path, "a") as f:
        f.write(f"# ============================================================\n")
        f.write(f"# Timestamp             : {timestamp}\n")
        f.write(f"# Hostname              : {socket.gethostname()}\n")
        f.write(f"# World Size            : {world_size}\n")
        f.write(f"# Backend               : {backend}\n")
        if backend == "nccl":
            f.write(f"# NCCL Version          : {nccl_version}\n")
        f.write(f"# Conda Environment     : {os.environ.get('CONDA_DEFAULT_ENV', 'N/A')}\n")
        f.write(f"# PyTorch Version       : {torch.__version__}\n")
        f.write(f"# CUDA Version          : {torch.version.cuda}\n")
        f.write(f"# CUDA Devices          : {cuda_devices}\n")
        f.write(f"# GPU Model             : {gpu_info}\n")
        f.write(f"# GPU Memory per Device : {gpu_memory}\n")
        f.write(f"# Warmup Iterations     : {warmup}\n")
        f.write(f"# Benchmark Iterations  : {iters}\n")
        f.write(f"# Tensor Sizes          : {tensor_sizes}\n")
        f.write(f"# ------------------------------------------------------------\n")
        f.write("backend,operation,tensor_elements,tensor_bytes,tensor_mb,"
                "warmup_iterations,benchmark_iterations,"
                "warmup_mean_sec,warmup_std_sec,"
                "benchmark_mean_sec,benchmark_std_sec,benchmark_min_sec,benchmark_max_sec,"
                "bandwidth_gbps\n")


def log_csv_result(output_path, backend, operation, world_size,
                   tensor_elements, tensor_bytes, warmup_times, bench_times,
                   warmup_iters, bench_iters):
    """Append one CSV row of results to *output_path*."""
    if output_path is None or not bench_times:
        return

    warmup_mean = statistics.mean(warmup_times) if warmup_times else 0
    warmup_std = statistics.stdev(warmup_times) if len(warmup_times) > 1 else 0
    bench_mean = statistics.mean(bench_times)
    bench_std = statistics.stdev(bench_times) if len(bench_times) > 1 else 0
    bench_min = min(bench_times)
    bench_max = max(bench_times)

    bw = compute_bandwidth(tensor_bytes, bench_mean, world_size, collective=operation)
    tensor_mb = tensor_bytes / (1024**2)

    nccl_version = get_nccl_version_str()
    backend_label = f"{backend}-{nccl_version}" if backend == "nccl" else backend

    with open(output_path, "a") as f:
        f.write(f"{backend_label},{operation},{tensor_elements},{tensor_bytes},{tensor_mb:.6f},")
        f.write(f"{warmup_iters},{bench_iters},")
        f.write(f"{warmup_mean:.8f},{warmup_std:.8f},")
        f.write(f"{bench_mean:.8f},{bench_std:.8f},{bench_min:.8f},{bench_max:.8f},")
        f.write(f"{bw:.6f}\n")


def print_benchmark_header(idx, total, size, size_bytes, warmup, iters):
    """Print per-tensor-size header matching torch_comm_bench style."""
    print(f"\nBenchmark {idx + 1}/{total}")
    print(f"Tensor size: {size:,} elements ({size_bytes / (1024**2):.2f} MB)")
    print(f"Running {warmup} warmup + {iters} benchmark iterations")


def print_op_result(op_name, warmup_times, bench_times, size_bytes, world_size, collective):
    """Print a single-line result for one operation matching torch_comm_bench style."""
    warmup_avg = statistics.mean(warmup_times) if warmup_times else 0
    bench_avg = statistics.mean(bench_times)
    bench_std = statistics.stdev(bench_times) if len(bench_times) > 1 else 0
    bw = compute_bandwidth(size_bytes, bench_avg, world_size, collective=collective)

    label = op_name.ljust(12)
    print(f"{label} - Warmup: {warmup_avg:.6f}s, Benchmark: {bench_avg:.6f}"
          f"\u00b1{bench_std:.6f}s, BW: {bw:.2f} GB/s")


def format_size(nbytes):
    """Human-readable byte size string."""
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.2f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.2f} TB"
