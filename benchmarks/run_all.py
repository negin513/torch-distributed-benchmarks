"""Run all collective benchmarks sequentially in a single launch.

Uses a single process group init/destroy cycle to avoid NCCL issues
with repeated initialisation in the same process.

Usage:
    torchrun --nproc_per_node=4 -m benchmarks.run_all --backend nccl
    mpiexec -np 8 --cpu-bind none python -m benchmarks.run_all --backend nccl
    mpiexec -np 8 --cpu-bind none python -m benchmarks.run_all --backend nccl --table
"""

import gc
import torch
import torch.distributed as dist

from benchmarks.collectives import all_reduce, all_gather, broadcast, reduce_scatter, point_to_point
from benchmarks.utils import (
    init_ranks,
    init_dist,
    gather_hostnames,
    print_env_info,
    print_rich_env_info,
    print_collective_table,
    print_p2p_table,
    print_summary_table,
    init_csv_log,
    common_arg_parser,
    default_tensor_sizes,
)


COLLECTIVES = [
    ("All-reduce", all_reduce),
    ("All-gather", all_gather),
    ("Broadcast", broadcast),
    ("Reduce-scatter", reduce_scatter),
    ("Point-to-point", point_to_point),
]


def main():
    parser = common_arg_parser("Run all collective benchmarks")
    args = parser.parse_args()
    sizes = args.tensor_sizes or default_tensor_sizes()
    use_table = args.table

    local_rank, world_rank, world_size = init_ranks()
    init_dist(args.backend, local_rank, world_rank, world_size)

    all_hostnames = gather_hostnames(world_size)

    if use_table:
        print_rich_env_info(world_rank, world_size, args.backend, all_hostnames)
    else:
        print_env_info(world_rank, world_size, args.backend, all_hostnames)

    if world_rank == 0:
        init_csv_log(args.output, args.backend, world_size, args.warmup, args.iters, sizes)

    all_results = {}

    for label, module in COLLECTIVES:
        if world_rank == 0 and not use_table:
            print(f"\n{'=' * 60}")
            print(f"  {label}")
            print(f"{'=' * 60}")

        results = module.bench(args.backend, sizes, args.warmup, args.iters, args.output,
                               local_rank, world_rank, world_size, use_table=use_table)

        if world_rank == 0 and use_table:
            if label == "Point-to-point":
                print_p2p_table(results or [])
                # For summary, use peak BW across all peers
                if results:
                    all_results[label] = results
            else:
                print_collective_table(label, results or [])
                if results:
                    all_results[label] = results

        dist.barrier()

    if world_rank == 0:
        if use_table:
            print_summary_table(all_results)
        print(f"\nAll benchmarks completed!")
        if args.output:
            print(f"Results saved to: {args.output}")

    dist.barrier()
    gc.collect()
    torch.cuda.empty_cache()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
