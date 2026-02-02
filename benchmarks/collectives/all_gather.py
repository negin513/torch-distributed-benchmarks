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


def bench(backend, tensor_sizes, warmup, iters, output, local_rank, world_rank, world_size):
    """Run all-gather benchmark loop (assumes process group is already initialised)."""
    use_cuda = backend == "nccl"

    for idx, size in enumerate(tensor_sizes):
        send_tensor = make_tensor(size, backend, local_rank)
        size_bytes = send_tensor.element_size() * send_tensor.numel()

        # Pre-allocate the gather output list
        gather_list = [torch.empty_like(send_tensor) for _ in range(world_size)]

        if world_rank == 0:
            print_benchmark_header(idx, len(tensor_sizes), size, size_bytes, warmup, iters)

        # Warmup
        warmup_times = []
        for _ in range(warmup):
            if use_cuda:
                torch.cuda.synchronize()
            dist.barrier()
            start = time.perf_counter()
            dist.all_gather(gather_list, send_tensor.clone())
            if use_cuda:
                torch.cuda.synchronize()
            dist.barrier()
            elapsed = time.perf_counter() - start
            if world_rank == 0:
                warmup_times.append(elapsed)

        # Benchmark
        bench_times = []
        for _ in range(iters):
            if use_cuda:
                torch.cuda.synchronize()
            dist.barrier()
            start = time.perf_counter()
            dist.all_gather(gather_list, send_tensor.clone())
            if use_cuda:
                torch.cuda.synchronize()
            dist.barrier()
            elapsed = time.perf_counter() - start
            if world_rank == 0:
                bench_times.append(elapsed)

        if world_rank == 0:
            print_op_result("All-gather", warmup_times, bench_times,
                            size_bytes, world_size, "all_gather")
            log_csv_result(output, backend, "all_gather", world_size,
                           size, size_bytes, warmup_times, bench_times, warmup, iters)


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
