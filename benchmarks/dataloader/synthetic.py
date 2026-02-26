"""Synthetic dataloader benchmark — sweep num_workers × prefetch_factor.

Reads ERA5-shaped data from a Zarr v3 store (1440 lon × 721 lat × 137 levels)
to benchmark data loading throughput at realistic climate-data scales.
Each sample is ~567 MB uncompressed at float32.

The 3-D parameter grid (batch_sizes × num_workers × prefetch_factor)
shows how worker count and prefetching affect throughput.

Usage:
    # First create the Zarr store:
    python scripts/create_era5_zarr.py --output /glade/derecho/scratch/negins/era5_bench.zarr

    # Then run the benchmark:
    torchrun --nproc_per_node=4 -m benchmarks.dataloader.synthetic \
        --zarr /glade/derecho/scratch/negins/era5_bench.zarr --backend nccl --table

    # Or single-GPU:
    python -m benchmarks.dataloader.synthetic \
        --zarr /glade/derecho/scratch/negins/era5_bench.zarr --table
"""

import os
import statistics
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import zarr

from benchmarks.utils import (
    init_ranks,
    init_dist,
    print_env_info,
    SimpleERA5Model,
    dataloader_arg_parser,
    benchmark_dataloader,
    init_dataloader_csv_log,
    log_dataloader_csv_result,
    print_dataloader_table,
    get_lustre_stripe_info,
    format_stripe_info,
)


class ZarrERA5Dataset(Dataset):
    """Dataset that reads ERA5-shaped data from a Zarr v3 store."""

    def __init__(self, zarr_path, num_classes=10):
        self.store = zarr.open_group(zarr_path, mode="r")
        self.data = self.store["data"]
        self.num_samples = self.data.shape[0]
        self.shape = self.data.shape[1:]  # (lon, lat, levels)
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Read from Zarr and convert to torch tensor
        arr = self.data[idx]
        field = torch.from_numpy(np.asarray(arr)).float()
        label = torch.randint(0, self.num_classes, (1,)).item()
        return field, label


def _is_distributed():
    """Return True if a distributed environment is detected."""
    return any(k in os.environ for k in (
        "LOCAL_RANK", "OMPI_COMM_WORLD_RANK", "PMI_RANK",
    ))


def _era5_arg_parser():
    """Return an ArgumentParser with ERA5/Zarr-specific options."""
    parser = dataloader_arg_parser(
        "ERA5 Zarr dataloader benchmark (sweep num_workers × prefetch_factor)")
    parser.set_defaults(
        batch_sizes=[1, 2],
        num_workers=[0, 2, 4, 8],
    )
    parser.add_argument("--zarr", type=str, required=True,
                        help="Path to ERA5 Zarr v3 store")
    return parser


def bench(backend, batch_sizes, num_workers_list, prefetch_factors,
          warmup, iters, output,
          local_rank, world_rank, world_size,
          zarr_path, num_classes, pin_memory, use_table=False):
    """Run Zarr ERA5 dataloader benchmark over batch_sizes × num_workers × prefetch_factor."""
    use_cuda = backend == "nccl" and torch.cuda.is_available()
    device = torch.device("cuda", local_rank) if use_cuda else torch.device("cpu")

    dataset = ZarrERA5Dataset(zarr_path, num_classes)
    lon, lat, num_levels = dataset.shape
    sample_bytes = lon * lat * num_levels * 4  # float32
    model = SimpleERA5Model(lon, num_classes).to(device)
    if world_size > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank] if use_cuda else None)
    criterion = torch.nn.CrossEntropyLoss()

    results = []
    for batch_size in batch_sizes:
        for nw in num_workers_list:
            # When num_workers=0, prefetch_factor is ignored by PyTorch
            pf_list = [None] if nw == 0 else prefetch_factors
            for pf in pf_list:
                if world_size > 1:
                    sampler = DistributedSampler(dataset, num_replicas=world_size,
                                                 rank=world_rank, shuffle=True)
                    sampler.set_epoch(0)
                else:
                    sampler = None

                loader_kwargs = dict(
                    batch_size=batch_size,
                    num_workers=nw,
                    pin_memory=pin_memory and use_cuda,
                    drop_last=True,
                    shuffle=(sampler is None),
                )
                if sampler is not None:
                    loader_kwargs["sampler"] = sampler
                if pf is not None:
                    loader_kwargs["prefetch_factor"] = pf
                loader = DataLoader(dataset, **loader_kwargs)

                model.zero_grad()
                timing = benchmark_dataloader(loader, model, device, criterion,
                                              warmup=warmup, iters=iters)

                pf_display = pf if pf is not None else "-"

                if world_rank == 0:
                    bt = timing["batch_times"]
                    dt = timing["data_times"]
                    ct = timing["compute_times"]
                    avg_batch = statistics.mean(bt)
                    std_batch = statistics.stdev(bt) if len(bt) > 1 else 0
                    avg_data = statistics.mean(dt)
                    avg_compute = statistics.mean(ct)
                    data_pct = (avg_data / avg_batch * 100) if avg_batch > 0 else 0
                    samples_per_sec = batch_size / avg_batch if avg_batch > 0 else 0
                    global_samples = samples_per_sec * world_size

                    if not use_table:
                        print(f"  batch_size={batch_size:>4d}  workers={nw}  "
                              f"prefetch={pf_display}  "
                              f"batch={avg_batch * 1e3:.2f}ms  data={avg_data * 1e3:.2f}ms  "
                              f"compute={avg_compute * 1e3:.2f}ms  data%={data_pct:.1f}%  "
                              f"samples/s={samples_per_sec:.1f}  "
                              f"global_samples/s={global_samples:.1f}")

                    log_dataloader_csv_result(
                        output, backend, "zarr_era5", world_size,
                        batch_size, nw, pf if pf is not None else 0,
                        len(dataset), f"{lon}x{lat}x{num_levels}", timing)

                    if use_table:
                        results.append({
                            "batch_size": batch_size,
                            "num_workers": nw,
                            "prefetch_factor": pf_display,
                            "avg_batch_ms": avg_batch * 1e3,
                            "std_batch_ms": std_batch * 1e3,
                            "avg_data_ms": avg_data * 1e3,
                            "avg_compute_ms": avg_compute * 1e3,
                            "data_pct": data_pct,
                            "samples_per_sec": samples_per_sec,
                            "global_samples_per_sec": global_samples,
                        })

                if world_size > 1:
                    dist.barrier()
    return results


def run(backend, batch_sizes, num_workers_list, prefetch_factors,
        warmup, iters, output, zarr_path, num_classes, pin_memory,
        use_table=False):
    """Standalone entry: initialise (optionally distributed), run benchmark, clean up."""
    distributed = _is_distributed()

    if distributed:
        local_rank, world_rank, world_size = init_ranks()
        init_dist(backend, local_rank, world_rank, world_size)
        print_env_info(world_rank, world_size, backend)
    else:
        local_rank, world_rank, world_size = 0, 0, 1
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        print("=" * 60)
        print("ZARR ERA5 DATALOADER BENCHMARK (single-GPU)")
        print(f"PyTorch Version  : {torch.__version__}")
        if torch.cuda.is_available():
            print(f"GPU Model        : {torch.cuda.get_device_name(0)}")
        print("=" * 60)

    # Open Zarr to get metadata
    store = zarr.open_group(zarr_path, mode="r")
    data = store["data"]
    num_samples, lon, lat, num_levels = data.shape
    sample_bytes = lon * lat * num_levels * 4
    stripe_info = get_lustre_stripe_info(zarr_path)

    if world_rank == 0:
        print(f"Zarr Store       : {zarr_path}")
        print(f"Dataset Size     : {num_samples} samples")
        print(f"Sample Shape     : ({lon}, {lat}, {num_levels})  "
              f"[lon × lat × levels]")
        print(f"Sample Size      : {sample_bytes / (1024**2):.1f} MB (float32)")
        print(f"Lustre Stripe    : {format_stripe_info(stripe_info)}")
        print(f"Batch Sizes      : {batch_sizes}")
        print(f"Num Workers      : {num_workers_list}")
        print(f"Prefetch Factors : {prefetch_factors}")
        print("=" * 60)
        if output:
            stripe_str = format_stripe_info(stripe_info)
            init_dataloader_csv_log(
                output, backend, world_size, warmup, iters,
                f"zarr={zarr_path}, shape=({lon},{lat},{num_levels}), "
                f"lustre_stripe=({stripe_str}), "
                f"batch_sizes={batch_sizes}, num_workers={num_workers_list}, "
                f"prefetch_factor={prefetch_factors}")

    results = bench(backend, batch_sizes, num_workers_list, prefetch_factors,
                    warmup, iters, output,
                    local_rank, world_rank, world_size,
                    zarr_path, num_classes, pin_memory, use_table=use_table)

    if world_rank == 0:
        if use_table:
            print_dataloader_table(
                f"Zarr ERA5 Dataloader ({lon}×{lat}×{num_levels})",
                results)
        print("\nBenchmark completed!")
        if output:
            print(f"Results saved to: {output}")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


def main():
    args = _era5_arg_parser().parse_args()
    run(args.backend, args.batch_sizes, args.num_workers, args.prefetch_factor,
        args.warmup, args.iters, args.output, args.zarr, args.num_classes,
        args.pin_memory, use_table=args.table)


if __name__ == "__main__":
    main()
