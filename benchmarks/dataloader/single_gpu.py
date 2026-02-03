"""Single-GPU dataloader benchmark (baseline, no distributed).

Measures data loading + forward/backward throughput on a single device.
Sweeps batch_sizes × num_workers to show how worker count affects throughput.

Usage:
    python -m benchmarks.dataloader.single_gpu --table
    python -m benchmarks.dataloader.single_gpu --batch_sizes 32 64 --num_workers 0 2 4
"""

import statistics
import torch
from torch.utils.data import DataLoader

from benchmarks.utils import (
    SyntheticImageDataset,
    SimpleModel,
    dataloader_arg_parser,
    benchmark_dataloader,
    init_dataloader_csv_log,
    log_dataloader_csv_result,
    print_dataloader_table,
    print_env_info,
)


def bench(batch_sizes, num_workers_list, warmup, iters, output,
          local_rank, image_size, num_channels, num_classes,
          dataset_size, pin_memory, use_table=False):
    """Run single-GPU dataloader benchmark (no distributed)."""
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    dataset = SyntheticImageDataset(dataset_size, num_channels, image_size, num_classes)
    model = SimpleModel(num_channels, num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()

    results = []
    for batch_size in batch_sizes:
        for nw in num_workers_list:
            loader_kwargs = dict(
                batch_size=batch_size,
                num_workers=nw,
                pin_memory=pin_memory and torch.cuda.is_available(),
                drop_last=True,
            )
            if nw > 0:
                loader_kwargs["prefetch_factor"] = 2
            loader = DataLoader(dataset, **loader_kwargs)

            model.zero_grad()
            timing = benchmark_dataloader(loader, model, device, criterion,
                                          warmup=warmup, iters=iters)

            bt = timing["batch_times"]
            dt = timing["data_times"]
            ct = timing["compute_times"]
            avg_batch = statistics.mean(bt)
            std_batch = statistics.stdev(bt) if len(bt) > 1 else 0
            avg_data = statistics.mean(dt)
            avg_compute = statistics.mean(ct)
            data_pct = (avg_data / avg_batch * 100) if avg_batch > 0 else 0
            samples_per_sec = batch_size / avg_batch if avg_batch > 0 else 0

            if not use_table:
                print(f"  batch_size={batch_size:>4d}  workers={nw}  "
                      f"batch={avg_batch * 1e3:.2f}ms  data={avg_data * 1e3:.2f}ms  "
                      f"compute={avg_compute * 1e3:.2f}ms  data%={data_pct:.1f}%  "
                      f"samples/s={samples_per_sec:.1f}")

            log_dataloader_csv_result(
                output, "cpu" if not torch.cuda.is_available() else "nccl",
                "single_gpu", 1, batch_size, nw, 2,
                dataset_size, image_size, timing)

            if use_table:
                results.append({
                    "batch_size": batch_size,
                    "num_workers": nw,
                    "prefetch_factor": 2,
                    "avg_batch_ms": avg_batch * 1e3,
                    "std_batch_ms": std_batch * 1e3,
                    "avg_data_ms": avg_data * 1e3,
                    "avg_compute_ms": avg_compute * 1e3,
                    "data_pct": data_pct,
                    "samples_per_sec": samples_per_sec,
                    "global_samples_per_sec": samples_per_sec,
                })
    return results


def run(batch_sizes, num_workers_list, warmup, iters, output,
        image_size, num_channels, num_classes, dataset_size, pin_memory,
        use_table=False):
    """Standalone entry: set up device, run benchmark, print results."""
    local_rank = 0
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    print("=" * 60)
    print("SINGLE-GPU DATALOADER BENCHMARK")
    print(f"Hostname         : {__import__('socket').gethostname()}")
    print(f"PyTorch Version  : {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU Model        : {torch.cuda.get_device_name(0)}")
    print(f"Dataset Size     : {dataset_size}")
    print(f"Image Size       : {image_size}×{image_size}×{num_channels}")
    print(f"Batch Sizes      : {batch_sizes}")
    print(f"Num Workers      : {num_workers_list}")
    print("=" * 60)

    if output:
        init_dataloader_csv_log(output, "nccl", 1, warmup, iters,
                                f"batch_sizes={batch_sizes}, num_workers={num_workers_list}")

    results = bench(batch_sizes, num_workers_list, warmup, iters, output,
                    local_rank, image_size, num_channels, num_classes,
                    dataset_size, pin_memory, use_table=use_table)

    if use_table:
        print_dataloader_table("Single-GPU Dataloader", results)

    print("\nBenchmark completed!")
    if output:
        print(f"Results saved to: {output}")


def main():
    parser = dataloader_arg_parser("Single-GPU dataloader benchmark (no distributed)")
    args = parser.parse_args()
    run(args.batch_sizes, args.num_workers, args.warmup, args.iters, args.output,
        args.image_size, args.num_channels, args.num_classes, args.dataset_size,
        args.pin_memory, use_table=args.table)


if __name__ == "__main__":
    main()
