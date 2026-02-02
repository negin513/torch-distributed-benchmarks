"""Point-to-point (send/recv) latency benchmark.

Measures half-round-trip latency between rank 0 and every other rank,
which is useful for diagnosing driver or interconnect issues.

Usage:
    torchrun --nproc_per_node=4 -m benchmarks.collectives.point_to_point --backend nccl
    mpirun -np 4 python -m benchmarks.collectives.point_to_point --backend nccl
"""

import time
import statistics
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
)


def _ping_pong(tensor, peer, backend):
    """Rank 0 sends to *peer*, then *peer* sends back.  Returns round-trip
    time on rank 0, None on other ranks."""
    rank = dist.get_rank()
    use_cuda = backend == "nccl"

    if use_cuda:
        torch.cuda.synchronize()

    rt = None
    if rank == 0:
        start = time.perf_counter()
        dist.send(tensor, dst=peer)
        dist.recv(tensor, src=peer)
        if use_cuda:
            torch.cuda.synchronize()
        rt = time.perf_counter() - start

    elif rank == peer:
        dist.recv(tensor, src=0)
        dist.send(tensor, dst=0)
        if use_cuda:
            torch.cuda.synchronize()

    # All ranks sync so non-participating ranks don't race ahead
    dist.barrier()
    return rt


def bench(backend, tensor_sizes, warmup, iters, output, local_rank, world_rank, world_size):
    """Run point-to-point benchmark loop (assumes process group is already initialised)."""
    if world_size < 2:
        if world_rank == 0:
            print("Point-to-point benchmark requires at least 2 ranks. Skipping.")
        return

    bench_idx = 0
    total = (world_size - 1) * len(tensor_sizes)

    for peer in range(1, world_size):
        for size in tensor_sizes:
            tensor = make_tensor(size, backend, local_rank)
            size_bytes = tensor.element_size() * tensor.numel()

            if world_rank == 0:
                print_benchmark_header(bench_idx, total, size, size_bytes, warmup, iters)
                print(f"  send/recv  rank 0 <-> rank {peer}")

            # Warmup
            warmup_times = []
            for _ in range(warmup):
                rt = _ping_pong(tensor.clone(), peer, backend)
                if rt is not None:
                    warmup_times.append(rt)

            bench_times = []
            for _ in range(iters):
                rt = _ping_pong(tensor.clone(), peer, backend)
                if rt is not None:
                    bench_times.append(rt)

            if world_rank == 0 and bench_times:
                warmup_avg = statistics.mean(warmup_times) if warmup_times else 0
                bench_avg = statistics.mean(bench_times)
                bench_std = statistics.stdev(bench_times) if len(bench_times) > 1 else 0
                half_rt = bench_avg / 2
                bw = (size_bytes / half_rt / (1024**3)) if half_rt > 0 else float("inf")

                print(f"{'Send/Recv'.ljust(12)} - Warmup: {warmup_avg:.6f}s, "
                      f"Benchmark: {bench_avg:.6f}\u00b1{bench_std:.6f}s, "
                      f"One-way: {half_rt:.6f}s, BW: {bw:.2f} GB/s")

                log_csv_result(output, backend, f"send_recv_peer{peer}", world_size,
                               size, size_bytes, warmup_times, bench_times, warmup, iters)

            bench_idx += 1

        # Sync all ranks before moving to the next peer
        dist.barrier()


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
    parser = common_arg_parser("Point-to-point (send/recv) latency benchmark")
    args = parser.parse_args()
    sizes = args.tensor_sizes or default_tensor_sizes()
    run(args.backend, sizes, args.warmup, args.iters, args.output)


if __name__ == "__main__":
    main()
