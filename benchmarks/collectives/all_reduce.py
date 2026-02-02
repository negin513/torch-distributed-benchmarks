"""All-reduce bandwidth and latency benchmark.

Usage:
    torchrun --nproc_per_node=4 -m benchmarks.collectives.all_reduce --backend nccl
    mpirun -np 4 python -m benchmarks.collectives.all_reduce --backend nccl
"""

import torch.distributed as dist

from benchmarks.utils import (
    init_ranks,
    init_dist,
    print_env_info,
    make_tensor,
    benchmark_op,
    default_tensor_sizes,
    common_arg_parser,
    init_csv_log,
    log_csv_result,
    print_benchmark_header,
    print_op_result,
)


def bench(backend, tensor_sizes, warmup, iters, output, local_rank, world_rank, world_size):
    """Run all-reduce benchmark loop (assumes process group is already initialised)."""
    for idx, size in enumerate(tensor_sizes):
        tensor = make_tensor(size, backend, local_rank)
        size_bytes = tensor.element_size() * tensor.numel()

        if world_rank == 0:
            print_benchmark_header(idx, len(tensor_sizes), size, size_bytes, warmup, iters)

        warmup_times, bench_times = benchmark_op(
            lambda t: dist.all_reduce(t),
            tensor,
            backend,
            warmup=warmup,
            iters=iters,
        )

        if world_rank == 0:
            print_op_result("All-reduce", warmup_times, bench_times,
                            size_bytes, world_size, "all_reduce")
            log_csv_result(output, backend, "all_reduce", world_size,
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
    parser = common_arg_parser("All-reduce benchmark")
    args = parser.parse_args()
    sizes = args.tensor_sizes or default_tensor_sizes()
    run(args.backend, sizes, args.warmup, args.iters, args.output)


if __name__ == "__main__":
    main()
