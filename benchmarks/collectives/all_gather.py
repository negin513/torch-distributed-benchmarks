"""All-gather bandwidth and latency benchmark.

Usage:
    torchrun --nproc_per_node=4 -m benchmarks.collectives.all_gather --backend nccl
    mpirun -np 4 python -m benchmarks.collectives.all_gather --backend nccl
"""

import time
import torch
import torch.distributed as dist

from benchmarks.utils import (
    init_ranks,
    init_dist,
    print_env_info,
    make_tensor,
    default_tensor_sizes,
    common_arg_parser,
    init_csv_log,
    log_csv_result,
    print_benchmark_header,
    print_op_result,
)


def bench(backend, tensor_sizes, warmup, iters, output, local_rank, world_rank, world_size,
          use_table=False):
    """Run all-gather benchmark loop (assumes process group is already initialised)."""
    import statistics as _stats
    from benchmarks.utils import compute_bandwidth
    use_cuda = backend == "nccl"

    results = []
    for idx, size in enumerate(tensor_sizes):
        send_tensor = make_tensor(size, backend, local_rank)
        per_rank_bytes = send_tensor.element_size() * send_tensor.numel()
        total_elems = size * world_size
        total_bytes = per_rank_bytes * world_size
        # Report per-rank input size (the "package" each rank contributes)
        report_elems = size
        report_bytes = per_rank_bytes

        # Pre-allocate a single contiguous output buffer (matches nccl-tests)
        device = torch.device("cuda", local_rank) if use_cuda else torch.device("cpu")
        output_tensor = torch.empty(total_elems, dtype=torch.float32, device=device)

        if world_rank == 0 and not use_table:
            print_benchmark_header(idx, len(tensor_sizes), report_elems, report_bytes, warmup, iters)

        # Warmup — tight loop, single sync (nccl-tests style)
        if use_cuda:
            torch.cuda.synchronize()
        dist.barrier()
        warmup_start = time.perf_counter()
        for _ in range(warmup):
            dist.all_gather_into_tensor(output_tensor, send_tensor)
        if use_cuda:
            torch.cuda.synchronize()
        warmup_total = time.perf_counter() - warmup_start

        warmup_times = []
        if world_rank == 0 and warmup > 0:
            warmup_times.append(warmup_total / warmup)

        # Benchmark — tight loop, single sync (nccl-tests style)
        if use_cuda:
            torch.cuda.synchronize()
        dist.barrier()
        bench_start = time.perf_counter()
        for _ in range(iters):
            dist.all_gather_into_tensor(output_tensor, send_tensor)
        if use_cuda:
            torch.cuda.synchronize()
        bench_total = time.perf_counter() - bench_start

        bench_times = []
        if world_rank == 0 and iters > 0:
            bench_times.append(bench_total / iters)

        if world_rank == 0:
            if not use_table:
                print_op_result("All-gather", warmup_times, bench_times,
                                total_bytes, world_size, "all_gather")
            log_csv_result(output, backend, "all_gather", world_size,
                           total_elems, total_bytes, warmup_times, bench_times, warmup, iters)
            if use_table and bench_times:
                bench_avg = _stats.mean(bench_times)
                algbw, busbw = compute_bandwidth(total_bytes, bench_avg, world_size, "all_gather")
                results.append({
                    "size": report_elems,
                    "size_bytes": report_bytes,
                    "warmup_avg": _stats.mean(warmup_times) if warmup_times else 0,
                    "bench_avg": bench_avg,
                    "bench_std": _stats.stdev(bench_times) if len(bench_times) > 1 else 0,
                    "algbw": algbw,
                    "busbw": busbw,
                })
    return results


def run(backend, tensor_sizes, warmup, iters, output):
    local_rank, world_rank, world_size = init_ranks()
    init_dist(backend, local_rank, world_rank, world_size)
    print_env_info(world_rank, world_size, backend)

    if world_rank == 0:
        init_csv_log(output, backend, world_size, warmup, iters, tensor_sizes)

    bench(backend, tensor_sizes, warmup, iters, output, local_rank, world_rank, world_size)

    if world_rank == 0:
        print(f"\nBenchmark completed!")
        if output:
            print(f"Results saved to: {output}")

    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = common_arg_parser("All-gather benchmark")
    args = parser.parse_args()
    sizes = args.tensor_sizes or default_tensor_sizes()
    run(args.backend, sizes, args.warmup, args.iters, args.output)


if __name__ == "__main__":
    main()
