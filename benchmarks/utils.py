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

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def init_ranks():
    """Detect and return (local_rank, world_rank, world_size) from the
    available environment.  Supports torchrun, mpi4py, OpenMPI env vars,
    and Cray MPICH (PMI).

    torchrun / elastic env vars are checked first because mpi4py may be
    importable even when the job was *not* launched via MPI, in which case
    MPI.COMM_WORLD reports world_size=1 and every process appears to be
    rank 0.
    """
    # --- torchrun / torch.distributed.launch ---
    if "LOCAL_RANK" in os.environ:
        return (
            int(os.environ["LOCAL_RANK"]),
            int(os.environ["RANK"]),
            int(os.environ["WORLD_SIZE"]),
        )

    # --- OpenMPI env vars (set by mpirun/mpiexec without mpi4py) ---
    if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        return (
            int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]),
            int(os.environ["OMPI_COMM_WORLD_RANK"]),
            int(os.environ["OMPI_COMM_WORLD_SIZE"]),
        )

    if "PMI_RANK" in os.environ:
        local_rank = int(os.environ["PMI_LOCAL_RANK"])
        world_rank = int(os.environ["PMI_RANK"])
        world_size = int(os.environ["PMI_SIZE"])
        if "MASTER_ADDR" not in os.environ or "MASTER_PORT" not in os.environ:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
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

    # --- mpi4py fallback (e.g. Cray MPICH without PMI env vars) ---
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        world_size = comm.Get_size()
        if world_size > 1:
            shmem_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
            local_rank = shmem_comm.Get_rank()
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

    raise RuntimeError(
        "Could not determine ranks — run with torchrun, mpirun, or MPI + mpi4py"
    )


def init_dist(backend, local_rank, world_rank, world_size):
    """Initialise the process group and set the CUDA device when using NCCL."""
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend, rank=world_rank, world_size=world_size,
                                device_id=device)
    else:
        dist.init_process_group(backend, rank=world_rank, world_size=world_size)


def get_nccl_version_str():
    """Return NCCL version as a dotted string, or 'N/A'."""
    try:
        return ".".join(map(str, torch.cuda.nccl.version()))
    except Exception:
        return "N/A"


def print_env_info(world_rank, world_size, backend, all_hostnames=None):
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
    if all_hostnames:
        num_nodes, node_list, ranks_per_node = _node_summary(all_hostnames)
        print(f"Nodes            : {num_nodes}")
        print(f"Node List        : {', '.join(node_list)}")
        rpn = ", ".join(f"{h} ({n})" for h, n in ranks_per_node.items())
        print(f"Ranks per Node   : {rpn}")
    else:
        print(f"Hostname         : {socket.gethostname()}")
    print(f"World Size       : {world_size}")
    print(f"Backend          : {backend}")
    print(f"Conda Env        : {os.environ.get('CONDA_DEFAULT_ENV', 'N/A')}")
    print(f"PyTorch Version  : {torch.__version__}")
    print(f"CUDA Version     : {torch.version.cuda}")
    if cuda_devices > 0 and all_hostnames:
        num_nodes = len(set(all_hostnames))
        print(f"CUDA Devices     : {cuda_devices} per node, {cuda_devices * num_nodes} total")
    else:
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

    Timing follows the nccl-tests methodology: all iterations run in a tight
    loop with a single cuda-synchronize at the end, so NCCL can pipeline
    operations.  The returned lists contain one entry — the per-iteration
    average — which keeps the downstream stats helpers working unchanged.

    warmup_times has one entry (the average warmup iteration time).
    benchmark_times has one entry (the average benchmark iteration time).
    """
    use_cuda = backend == "nccl"

    # ---------- warmup ----------
    if use_cuda:
        torch.cuda.synchronize()
    dist.barrier()
    start = time.perf_counter()
    for _ in range(warmup):
        op_fn(tensor)
    if use_cuda:
        torch.cuda.synchronize()
    warmup_total = time.perf_counter() - start

    warmup_times = []
    if dist.get_rank() == 0 and warmup > 0:
        warmup_times.append(warmup_total / warmup)

    # ---------- benchmark ----------
    if use_cuda:
        torch.cuda.synchronize()
    dist.barrier()
    start = time.perf_counter()
    for _ in range(iters):
        op_fn(tensor)
    if use_cuda:
        torch.cuda.synchronize()
    bench_total = time.perf_counter() - start

    bench_times = []
    if dist.get_rank() == 0 and iters > 0:
        bench_times.append(bench_total / iters)

    return warmup_times, bench_times


def _bus_factor(collective, world_size):
    """Return the nccl-tests bus bandwidth correction factor."""
    n = world_size
    if collective == "all_reduce":
        return 2.0 * (n - 1) / n
    elif collective == "broadcast":
        return 1.0
    elif collective in ("all_gather", "reduce", "reduce_scatter"):
        return (n - 1) / n
    return 1.0


def compute_bandwidth(size_bytes, time_sec, world_size, collective="all_reduce"):
    """Return (algbw, busbw) in GB/s (using 10^9, matching nccl-tests).

    algbw = size / time  (algorithm bandwidth — simple throughput).
    busbw = algbw * factor  (bus bandwidth — accounts for ring/tree data
    movement, comparable across different world sizes).

    Correction factors follow nccl-tests conventions:
        all_reduce    : 2*(N-1)/N
        broadcast     : 1
        all_gather / reduce / reduce_scatter : (N-1)/N
    """
    factor = _bus_factor(collective, world_size)
    algbw = (size_bytes / 1e9) / time_sec
    busbw = algbw * factor
    return algbw, busbw


# ---------------------------------------------------------------------------
# Dataloader benchmark utilities
# ---------------------------------------------------------------------------

class SyntheticImageDataset(torch.utils.data.Dataset):
    """Dataset of random image tensors for benchmarking (no filesystem I/O)."""

    def __init__(self, num_samples, num_channels=3, image_size=224, num_classes=1000):
        self.num_samples = num_samples
        self.shape = (num_channels, image_size, image_size)
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.randn(*self.shape)
        label = torch.randint(0, self.num_classes, (1,)).item()
        return image, label


class SyntheticERA5Dataset(torch.utils.data.Dataset):
    """Dataset mimicking ERA5 reanalysis fields: (lon, lat, levels).

    Default shape matches ERA5 high-resolution: 1440 lon × 721 lat × 137 levels
    (~567 MB per sample uncompressed at float32).
    """

    def __init__(self, num_samples, lon=1440, lat=721, num_levels=137, num_classes=10):
        self.num_samples = num_samples
        self.shape = (lon, lat, num_levels)
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        field = torch.randn(*self.shape)
        label = torch.randint(0, self.num_classes, (1,)).item()
        return field, label


class SimpleModel(torch.nn.Module):
    """Small CNN for dataloader benchmarking (not meant to train to convergence)."""

    def __init__(self, num_channels=3, num_classes=1000):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(num_channels, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = torch.nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class SimpleERA5Model(torch.nn.Module):
    """Small CNN for ERA5-shaped inputs (lon × lat × levels)."""

    def __init__(self, lon=1440, num_classes=10):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(lon, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = torch.nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def dataloader_arg_parser(description="Distributed dataloader benchmark"):
    """Return an ArgumentParser with dataloader-specific flags."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--backend", choices=["nccl", "gloo", "mpi"], default="nccl",
                        help='Distributed backend ("nccl" for GPU, "gloo" for CPU)')
    parser.add_argument("--warmup", type=int, default=5,
                        help="Number of warmup batches")
    parser.add_argument("--iters", type=int, default=50,
                        help="Number of benchmark batches to iterate")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV log file")
    parser.add_argument("--table", action="store_true", default=False,
                        help="Use rich tables for output formatting")
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[32, 64, 128],
                        help="Batch sizes to sweep")
    parser.add_argument("--num_workers", nargs="+", type=int, default=[0, 2, 4],
                        help="DataLoader num_workers values to sweep")
    parser.add_argument("--prefetch_factor", nargs="+", type=int, default=[2, 4],
                        help="Prefetch factor values to sweep (ignored when num_workers=0)")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Synthetic image spatial dimension (H=W)")
    parser.add_argument("--num_channels", type=int, default=3,
                        help="Number of image channels")
    parser.add_argument("--num_classes", type=int, default=1000,
                        help="Number of classes for synthetic labels")
    parser.add_argument("--dataset_size", type=int, default=10000,
                        help="Number of samples in the synthetic dataset")
    parser.add_argument("--pin_memory", action="store_true", default=True,
                        help="Use pinned memory for DataLoader")
    return parser


def benchmark_dataloader(dataloader, model, device, criterion, warmup=5, iters=50):
    """Time *iters* batches from *dataloader*, return timing breakdown.

    Returns dict with keys: batch_times, data_times, compute_times (lists of floats in seconds).
    """
    model.train()
    batch_times = []
    data_times = []
    compute_times = []
    dl_iter = iter(dataloader)
    total_batches = warmup + iters

    for i in range(total_batches):
        t0 = time.perf_counter()
        try:
            images, labels = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dataloader)
            images, labels = next(dl_iter)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        t1 = time.perf_counter()

        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        if i >= warmup:
            batch_times.append(t2 - t0)
            data_times.append(t1 - t0)
            compute_times.append(t2 - t1)

    return {
        "batch_times": batch_times,
        "data_times": data_times,
        "compute_times": compute_times,
    }


def init_dataloader_csv_log(output_path, backend, world_size, warmup, iters, config_info=""):
    """Write a metadata header and CSV column header for dataloader benchmarks."""
    if output_path is None:
        return
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    cuda_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
    gpu_info = "N/A"
    gpu_memory = "N/A"
    if torch.cuda.is_available() and cuda_devices > 0:
        gpu_info = torch.cuda.get_device_name(0)
        gpu_memory = f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB"

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    with open(output_path, "a") as f:
        f.write(f"# ============================================================\n")
        f.write(f"# Timestamp             : {timestamp}\n")
        f.write(f"# Hostname              : {socket.gethostname()}\n")
        f.write(f"# World Size            : {world_size}\n")
        f.write(f"# Backend               : {backend}\n")
        if backend == "nccl":
            f.write(f"# NCCL Version          : {get_nccl_version_str()}\n")
        f.write(f"# PyTorch Version       : {torch.__version__}\n")
        f.write(f"# CUDA Version          : {torch.version.cuda}\n")
        f.write(f"# CUDA Devices          : {cuda_devices}\n")
        f.write(f"# GPU Model             : {gpu_info}\n")
        f.write(f"# GPU Memory per Device : {gpu_memory}\n")
        f.write(f"# Warmup Batches        : {warmup}\n")
        f.write(f"# Benchmark Batches     : {iters}\n")
        if config_info:
            f.write(f"# Config                : {config_info}\n")
        f.write(f"# ------------------------------------------------------------\n")
        f.write("backend,benchmark_type,world_size,batch_size,num_workers,prefetch_factor,"
                "dataset_size,image_size,"
                "avg_batch_time_ms,std_batch_time_ms,min_batch_time_ms,max_batch_time_ms,"
                "avg_data_time_ms,avg_compute_time_ms,data_pct,"
                "samples_per_sec,batches_per_sec,global_samples_per_sec\n")


def log_dataloader_csv_result(output_path, backend, benchmark_type, world_size,
                              batch_size, num_workers, prefetch_factor,
                              dataset_size, image_size, timing):
    """Append one CSV row of dataloader benchmark results."""
    if output_path is None:
        return

    bt = timing["batch_times"]
    dt = timing["data_times"]
    ct = timing["compute_times"]

    avg_batch = statistics.mean(bt)
    std_batch = statistics.stdev(bt) if len(bt) > 1 else 0
    min_batch = min(bt)
    max_batch = max(bt)
    avg_data = statistics.mean(dt)
    avg_compute = statistics.mean(ct)
    data_pct = (avg_data / avg_batch * 100) if avg_batch > 0 else 0
    samples_per_sec = batch_size / avg_batch if avg_batch > 0 else 0
    batches_per_sec = 1.0 / avg_batch if avg_batch > 0 else 0
    global_samples = samples_per_sec * world_size

    nccl_version = get_nccl_version_str()
    backend_label = f"{backend}-{nccl_version}" if backend == "nccl" else backend

    with open(output_path, "a") as f:
        f.write(f"{backend_label},{benchmark_type},{world_size},{batch_size},"
                f"{num_workers},{prefetch_factor},{dataset_size},{image_size},"
                f"{avg_batch * 1e3:.4f},{std_batch * 1e3:.4f},"
                f"{min_batch * 1e3:.4f},{max_batch * 1e3:.4f},"
                f"{avg_data * 1e3:.4f},{avg_compute * 1e3:.4f},{data_pct:.1f},"
                f"{samples_per_sec:.1f},{batches_per_sec:.2f},{global_samples:.1f}\n")


def print_dataloader_table(title, results):
    """Print a rich table for dataloader benchmark results.

    *results* is a list of dicts with keys:
        batch_size, num_workers, prefetch_factor, avg_batch_ms, std_batch_ms,
        avg_data_ms, avg_compute_ms, data_pct, samples_per_sec, global_samples_per_sec
    """
    if not HAS_RICH or not results:
        return

    console = Console(width=130)
    table = Table(title=f"[bold bright_cyan]{title}[/bold bright_cyan]",
                  border_style="bright_blue", show_lines=False,
                  padding=(0, 1), header_style="bold bright_white")
    table.add_column("Batch Size", justify="right", style="cyan", no_wrap=True)
    table.add_column("Workers", justify="right", no_wrap=True)
    table.add_column("Prefetch", justify="right", no_wrap=True)
    table.add_column("Batch (ms)", justify="right", no_wrap=True)
    table.add_column("± Std (ms)", justify="right", style="dim", no_wrap=True)
    table.add_column("Data (ms)", justify="right", no_wrap=True)
    table.add_column("Compute (ms)", justify="right", no_wrap=True)
    table.add_column("Data %", justify="right", no_wrap=True)
    table.add_column("Samples/s", justify="right", no_wrap=True)
    table.add_column("Global Samples/s", justify="right", style="bold green", no_wrap=True)

    for r in results:
        data_pct = r["data_pct"]
        pct_style = "bold red" if data_pct > 50 else ("yellow" if data_pct > 25 else "green")
        pct_text = Text(f"{data_pct:.1f}%", style=pct_style)

        table.add_row(
            str(r["batch_size"]),
            str(r["num_workers"]),
            str(r["prefetch_factor"]),
            f"{r['avg_batch_ms']:.2f}",
            f"{r['std_batch_ms']:.2f}",
            f"{r['avg_data_ms']:.2f}",
            f"{r['avg_compute_ms']:.2f}",
            pct_text,
            f"{r['samples_per_sec']:.1f}",
            f"{r['global_samples_per_sec']:.1f}",
        )

    console.print()
    console.print(table)


def default_tensor_sizes():
    """Return a default list of 1-D tensor lengths spanning ~4 KB to ~2 GB."""
    return [10**i for i in range(3, 9)] + [200_000_000, 500_000_000]


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
        help="1-D tensor sizes to sweep (default: 1e3 … 5e8)",
    )
    parser.add_argument("--warmup", type=int, default=5,
                        help="Number of warmup iterations")
    parser.add_argument("--iters", type=int, default=20,
                        help="Number of benchmark iterations")
    parser.add_argument("--output", type=str, default="results/benchmark_results.log",
                        help="Output CSV log file")
    parser.add_argument("--table", action="store_true", default=False,
                        help="Use rich tables for output formatting")
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
                "benchmark_mean_ms,benchmark_std_ms,benchmark_min_ms,benchmark_max_ms,"
                "algbw_gibps,busbw_gibps\n")


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

    algbw, busbw = compute_bandwidth(tensor_bytes, bench_mean, world_size, collective=operation)
    tensor_mb = tensor_bytes / (1024**2)

    # Millisecond equivalents
    bench_mean_ms = bench_mean * 1e3
    bench_std_ms = bench_std * 1e3
    bench_min_ms = bench_min * 1e3
    bench_max_ms = bench_max * 1e3

    nccl_version = get_nccl_version_str()
    backend_label = f"{backend}-{nccl_version}" if backend == "nccl" else backend

    with open(output_path, "a") as f:
        f.write(f"{backend_label},{operation},{tensor_elements},{tensor_bytes},{tensor_mb:.6f},")
        f.write(f"{warmup_iters},{bench_iters},")
        f.write(f"{warmup_mean:.8f},{warmup_std:.8f},")
        f.write(f"{bench_mean:.8f},{bench_std:.8f},{bench_min:.8f},{bench_max:.8f},")
        f.write(f"{bench_mean_ms:.4f},{bench_std_ms:.4f},{bench_min_ms:.4f},{bench_max_ms:.4f},")
        f.write(f"{algbw:.6f},{busbw:.6f}\n")


def _ansi_bw(value):
    """Return *value* formatted with ANSI colour based on bandwidth magnitude."""
    s = f"{value:.2f}"
    if value >= 30:
        return f"\033[1;32m{s}\033[0m"   # bold green
    elif value >= 10:
        return f"\033[33m{s}\033[0m"      # yellow
    else:
        return f"\033[2m{s}\033[0m"       # dim


def print_benchmark_header(idx, total, size, size_bytes, warmup, iters):
    """Print per-tensor-size header matching torch_comm_bench style."""
    print(f"\n\033[1mBenchmark {idx + 1}/{total}\033[0m")
    print(f"Tensor size: \033[36m{size:,}\033[0m elements ({size_bytes / (1024**2):.2f} MB)")
    print(f"Running {warmup} warmup + {iters} benchmark iterations")


def print_op_result(op_name, warmup_times, bench_times, size_bytes, world_size, collective):
    """Print a single-line result for one operation matching torch_comm_bench style."""
    warmup_avg = statistics.mean(warmup_times) if warmup_times else 0
    bench_avg = statistics.mean(bench_times)
    bench_std = statistics.stdev(bench_times) if len(bench_times) > 1 else 0
    bench_avg_ms = bench_avg * 1e3
    algbw, busbw = compute_bandwidth(size_bytes, bench_avg, world_size, collective=collective)

    label = f"\033[1m{op_name.ljust(12)}\033[0m"
    print(f"{label} - Warmup: {warmup_avg:.6f}s, Benchmark: {bench_avg_ms:.3f}ms"
          f" (±{bench_std * 1e3:.3f}ms), AlgBW: {_ansi_bw(algbw)} GB/s, "
          f"BusBW: {_ansi_bw(busbw)} GB/s")


def format_size(nbytes):
    """Human-readable byte size string."""
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.2f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.2f} TB"


# ---------------------------------------------------------------------------
# Rich table output helpers
# ---------------------------------------------------------------------------

def _bw_style(bw):
    """Return a rich style string based on bandwidth value."""
    if bw >= 30:
        return "bold green"
    elif bw >= 10:
        return "yellow"
    else:
        return "dim"


def gather_hostnames(world_size):
    """Gather hostnames from all ranks. Returns list of hostnames on rank 0, None elsewhere."""
    hostname = socket.gethostname()
    all_hostnames = [None] * world_size
    dist.all_gather_object(all_hostnames, hostname)
    return all_hostnames


def _node_summary(all_hostnames):
    """Return (num_nodes, node_list, ranks_per_node) from gathered hostnames."""
    from collections import OrderedDict
    node_ranks = OrderedDict()
    for rank, host in enumerate(all_hostnames):
        node_ranks.setdefault(host, []).append(rank)
    num_nodes = len(node_ranks)
    node_list = list(node_ranks.keys())
    ranks_per_node = {host: len(ranks) for host, ranks in node_ranks.items()}
    return num_nodes, node_list, ranks_per_node


def print_rich_env_info(world_rank, world_size, backend, all_hostnames=None):
    """Print system info as a rich panel from rank 0."""
    if world_rank != 0 or not HAS_RICH:
        return

    console = Console(width=120)
    cuda_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
    gpu_info = "N/A"
    gpu_memory = "N/A"
    if torch.cuda.is_available() and cuda_devices > 0:
        gpu_info = torch.cuda.get_device_name(0)
        gpu_memory = f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB"

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold cyan")
    table.add_column()

    if all_hostnames:
        num_nodes, node_list, ranks_per_node = _node_summary(all_hostnames)
        table.add_row("Nodes", f"{num_nodes}")
        table.add_row("Node List", ", ".join(node_list))
        rpn = ", ".join(f"{h} ({n})" for h, n in ranks_per_node.items())
        table.add_row("Ranks per Node", rpn)
    else:
        table.add_row("Hostname", socket.gethostname())

    table.add_row("World Size", str(world_size))
    table.add_row("Backend", backend)
    table.add_row("Conda Env", os.environ.get("CONDA_DEFAULT_ENV", "N/A"))
    table.add_row("PyTorch Version", torch.__version__)
    table.add_row("CUDA Version", str(torch.version.cuda))
    table.add_row("CUDA Devices", f"{cuda_devices} per node, {cuda_devices * (num_nodes if all_hostnames else 1)} total"
                   if cuda_devices > 0 else "0")
    table.add_row("GPU Model", gpu_info)
    table.add_row("GPU Memory", gpu_memory)
    if backend == "nccl":
        table.add_row("NCCL Version", get_nccl_version_str())

    console.print(Panel(table, title="[bold]Distributed Communication Benchmark[/bold]",
                        border_style="bright_blue", width=90))


def print_collective_table(label, results):
    """Print a rich table for collective benchmark results.

    *results* is a list of dicts with keys:
        size, size_bytes, warmup_avg, bench_avg, bench_std, algbw, busbw
    """
    if not HAS_RICH or not results:
        return

    # Collectives where AlgBW == BusBW (correction factor = 1)
    _SAME_BW = {"Broadcast"}
    # Per-collective BusBW correction factor labels
    _BW_FACTOR = {
        "All-reduce":     "2·(N−1)/N",
        "All-gather":     "(N−1)/N",
        "Reduce-scatter": "(N−1)/N",
    }
    same_bw = label in _SAME_BW

    console = Console(width=120)
    table = Table(title=f"[bold bright_cyan]{label}[/bold bright_cyan]",
                  border_style="bright_blue", show_lines=False,
                  padding=(0, 1), header_style="bold bright_white")
    table.add_column("Elements", justify="right", style="cyan", no_wrap=True)
    table.add_column("Size", justify="right", no_wrap=True)
    table.add_column("Warmup (ms)", justify="right", no_wrap=True)
    table.add_column("Latency (ms)", justify="right", no_wrap=True)
    table.add_column("± Std (ms)", justify="right", style="dim", no_wrap=True)
    if same_bw:
        table.add_column("BW (GB/s)", justify="right", no_wrap=True)
    else:
        table.add_column("AlgBW (GB/s)", justify="right", no_wrap=True)
        table.add_column("BusBW (GB/s)", justify="right", no_wrap=True)

    for r in results:
        algbw_text = Text(f"{r['algbw']:.2f}", style=_bw_style(r["algbw"]))
        busbw_text = Text(f"{r['busbw']:.2f}", style=_bw_style(r["busbw"]))
        bench_ms = r["bench_avg"] * 1e3
        std_ms = r["bench_std"] * 1e3
        warmup_ms = r["warmup_avg"] * 1e3
        row = [
            f"{r['size']:,}",
            f"{r['size_bytes'] / (1024**2):.2f} MB",
            f"{warmup_ms:.3f}",
            f"{bench_ms:.3f}",
            f"{std_ms:.3f}",
        ]
        if same_bw:
            row.append(algbw_text)
        else:
            row.extend([algbw_text, busbw_text])
        table.add_row(*row)

    console.print()
    console.print(table)

    # Show BW explanation notes
    if label in _BW_FACTOR:
        factor = _BW_FACTOR[label]
        console.print(f"  [dim]Notes:[/dim]")
        console.print(
            f"  [dim]• AlgBW = Size / Time         — raw algorithm throughput (GB/s, decimal 10⁹, matching nccl-tests)[/dim]"
        )
        console.print(
            f"  [dim]• BusBW = AlgBW × [bright_magenta]{factor}[/bright_magenta]"
            f"   — normalised per-link bus utilisation (nccl-tests convention);[/dim]"
        )
        console.print(
            f"  [dim]                                  correction factor accounts for ring/tree data movement so BusBW[/dim]"
        )
        console.print(
            f"  [dim]                                  is comparable across different world sizes (N = number of ranks)[/dim]"
        )
    elif same_bw:
        console.print(f"  [dim]Notes:[/dim]")
        console.print(
            f"  [dim]• BW = Size / Time — raw algorithm throughput (GB/s, decimal 10⁹, matching nccl-tests)[/dim]"
        )


def print_p2p_table(results):
    """Print a rich table for point-to-point benchmark results.

    *results* is a list of dicts with keys:
        peer, size, size_bytes, warmup_avg, bench_avg, bench_std, half_rt, bw
    """
    if not HAS_RICH or not results:
        return

    console = Console(width=120)

    # Group by peer
    peers = sorted(set(r["peer"] for r in results))
    for peer in peers:
        peer_results = [r for r in results if r["peer"] == peer]
        table = Table(
            title=f"[bold bright_cyan]Send/Recv  rank 0 ↔ rank {peer}[/bold bright_cyan]",
            border_style="bright_blue", show_lines=False,
            padding=(0, 1), header_style="bold bright_white")
        table.add_column("Elements", justify="right", style="cyan", no_wrap=True)
        table.add_column("Size", justify="right", no_wrap=True)
        table.add_column("Warmup (ms)", justify="right", no_wrap=True)
        table.add_column("RTT (ms)", justify="right", no_wrap=True)
        table.add_column("± Std (ms)", justify="right", style="dim", no_wrap=True)
        table.add_column("One-way (ms)", justify="right", no_wrap=True)
        table.add_column("BW (GB/s)", justify="right", no_wrap=True)

        for r in peer_results:
            bw_text = Text(f"{r['algbw']:.2f}", style=_bw_style(r["algbw"]))
            rtt_ms = r["bench_avg"] * 1e3
            std_ms = r["bench_std"] * 1e3
            half_rt_ms = r["half_rt"] * 1e3
            warmup_ms = r["warmup_avg"] * 1e3
            table.add_row(
                f"{r['size']:,}",
                f"{r['size_bytes'] / (1024**2):.2f} MB",
                f"{warmup_ms:.3f}",
                f"{rtt_ms:.3f}",
                f"{std_ms:.3f}",
                f"{half_rt_ms:.3f}",
                bw_text,
            )

        console.print()
        console.print(table)


def print_summary_table(all_results):
    """Print a final summary table showing peak BW per collective.

    *all_results* is a dict mapping collective label -> list of result dicts.
    """
    if not HAS_RICH or not all_results:
        return

    console = Console(width=120)
    table = Table(title="Summary — Peak Bandwidth", title_style="bold bright_yellow",
                  border_style="bright_yellow", padding=(0, 1),
                  header_style="bold bright_white")
    table.add_column("Collective", style="bold cyan", no_wrap=True)
    table.add_column("Peak AlgBW (GB/s)", justify="right", no_wrap=True)
    table.add_column("Peak BusBW (GB/s)", justify="right", no_wrap=True)
    table.add_column("At Size", justify="right", no_wrap=True)

    for label, results in all_results.items():
        if not results:
            continue
        best = max(results, key=lambda r: r["busbw"])
        algbw_text = Text(f"{best['algbw']:.2f}", style=_bw_style(best["algbw"]))
        busbw_text = Text(f"{best['busbw']:.2f}", style=_bw_style(best["busbw"]))
        table.add_row(label, algbw_text, busbw_text, f"{best['size']:,}")

    console.print()
    console.print(table)
