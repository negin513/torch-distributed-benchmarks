"""Run all collective benchmarks sequentially in a single launch.

Uses a single process group init/destroy cycle to avoid NCCL issues
with repeated initialisation in the same process.

Usage:
    torchrun --nproc_per_node=4 -m benchmarks.run_all --backend nccl
    mpiexec -np 8 --cpu-bind none python -m benchmarks.run_all --backend nccl
"""

import torch.distributed as dist

from benchmarks.collectives import all_reduce, all_gather, broadcast, reduce_scatter, point_to_point
from benchmarks.utils import (
    init_ranks,
    init_dist,
    print_env_info,
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

    local_rank, world_rank, world_size = init_ranks()
    init_dist(args.backend, local_rank, world_rank, world_size)
    print_env_info(world_rank, world_size, args.backend)

    if world_rank == 0:
        init_csv_log(args.output, args.backend, world_size, args.warmup, args.iters, sizes)

    for label, module in COLLECTIVES:
        if world_rank == 0:
            print(f"\n{'=' * 60}")
            print(f"  {label}")
            print(f"{'=' * 60}")

        module.bench(args.backend, sizes, args.warmup, args.iters, args.output,
                     local_rank, world_rank, world_size)

        dist.barrier()

    if world_rank == 0:
        print(f"\nAll benchmarks completed!")
        if args.output:
            print(f"Results saved to: {args.output}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
