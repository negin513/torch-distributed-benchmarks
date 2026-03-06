# Distributed PyTorch Benchmarks

A benchmarking suite for profiling communication collectives and distributed
training workflows on GPU clusters.  Designed for HPC environments (NCAR
Derecho, SLURM/PBS clusters) but also works locally with `torchrun`.

## Why this matters

Distributed training splits work across GPUs that must constantly exchange
data — gradients, activations, parameters.  That exchange relies on a small
set of **communication collectives** (all-reduce, all-gather, broadcast,
send/recv).  In practice these collectives are where performance falls apart:

* **All-reduce** is the workhorse of data-parallel training.  Every backward
  pass ends with an all-reduce to average gradients across ranks.  In a
  typical DDP training loop the all-reduce can account for **30–50 %** of
  total step time once the model is large enough that gradient payloads
  exceed the interconnect bandwidth.

* **All-gather** shows up in FSDP / ZeRO-3 style sharding, where each rank
  holds a shard of the parameters and must reconstruct the full layer before
  the forward pass.  If all-gather is slow the GPUs stall waiting for
  weights.

* **Broadcast** is used at initialisation (syncing model weights) and in
  pipeline-parallel schedules.  A slow broadcast at startup inflates job
  launch time; inside a pipeline it serialises stages.

* **Reduce-scatter** is the inverse of all-gather and appears in FSDP /
  ZeRO backward passes — each rank ends up with a reduced shard of the
  gradient.  It is also half of the all-reduce ring decomposition
  (reduce-scatter + all-gather).

* **Point-to-point send/recv** underpins pipeline parallelism and some
  expert-routing schemes.  Latency here directly limits how finely you can
  slice the pipeline.

When any of these operations under-perform, GPUs sit idle burning compute
hours.  Common root causes include:

| Symptom | Likely cause |
|---------|-------------|
| All-reduce 10–100x slower than expected | Wrong NCCL transport (falling back to shared memory or TCP instead of NVLink / IB) |
| Latency spike on first call only | NCCL graph initialisation; usually harmless but masks real issues if you don't separate warmup |
| Bandwidth drops at multi-node scale | PCIe bottleneck, bad IB routing, or GDR (GPU-Direct RDMA) disabled |
| send/recv hangs or times out | Driver mismatch between nodes, NIC firmware issue, or firewall blocking |
| Training throughput plateaus beyond N GPUs | Communication cost now dominates — need gradient compression, overlap, or a different parallelism strategy |

This suite lets you **isolate and measure each collective independently** so
you can tell whether a training slowdown is compute-bound or
communication-bound, and which exact operation is the bottleneck.

## Repository structure

```
benchmarks/
├── __init__.py
├── utils.py                          # shared: rank init, timing, bandwidth math, CSV output
├── run_all.py                        # run every collective benchmark in a single launch
├── collectives/
│   ├── all_reduce.py                 # all-reduce bandwidth & latency vs message size
│   ├── all_gather.py                 # all-gather bandwidth & latency vs message size
│   ├── broadcast.py                  # broadcast bandwidth & latency vs message size
│   ├── reduce_scatter.py             # reduce-scatter bandwidth & latency vs message size
│   └── point_to_point.py             # ping-pong send/recv between rank pairs
└── dataloader/
    ├── single_gpu.py                 # single-GPU dataloader throughput benchmark
    └── synthetic.py                  # synthetic dataset utilities

results/                              # benchmark output files (CSV format)
```

## Quick start

### Prerequisites

* Python 3.9+
* PyTorch with CUDA support
* NCCL (bundled with PyTorch CUDA builds)
* A multi-GPU machine **or** a multi-node allocation

# With Cray MPICH (e.g. on Derecho)
  ml conda 
  conda activate torch-
  mpiexec -np 4 --cpu-bind none python -m benchmarks.run_all --backend nccl --table

  # With OpenMPI
  mpirun -np 4 python -m benchmarks.run_all --backend nccl --table

  Run individual collectives

  torchrun --nproc_per_node=4 -m benchmarks.collectives.all_reduce --backend nccl
  --table
  torchrun --nproc_per_node=4 -m benchmarks.collectives.all_gather --backend nccl
  --table
  torchrun --nproc_per_node=4 -m benchmarks.collectives.broadcast --backend nccl
  --table
  torchrun --nproc_per_node=4 -m benchmarks.collectives.reduce_scatter --backend nccl
  --table
  torchrun --nproc_per_node=4 -m benchmarks.collectives.point_to_point --backend nccl
  --table



### Dataloader benchmarks

```bash
# Single-GPU dataloader benchmark with tabular output
python -m benchmarks.dataloader.single_gpu --table

# Customize batch sizes and worker counts, save results
python -m benchmarks.dataloader.single_gpu \
    --batch_sizes 32 64 128 \
    --num_workers 0 2 4 8 \
    --output results/dataloader_single_gpu.csv

# Adjust dataset and image parameters
python -m benchmarks.dataloader.single_gpu \
    --dataset_size 50000 \
    --image_size 256 \
    --table
```

### Local (single node)

```bash
# Run all collectives with tabular output
torchrun --nproc_per_node=4 -m benchmarks.run_all --backend nccl --table

# Run all-reduce across 4 GPUs
torchrun --nproc_per_node=4 -m benchmarks.collectives.all_reduce --backend nccl

# Specify tensor sizes and save results
torchrun --nproc_per_node=4 -m benchmarks.collectives.all_gather \
    --backend nccl \
    --tensor_sizes 1000 100000 10000000 \
    --warmup 5 --iters 50 \
    --output results/all_gather.jsonl
```

### With MPI (OpenMPI or Cray MPICH)

```bash
# Run all collectives with tabular output (Cray MPICH)
mpiexec -np 4 --cpu-bind none python -m benchmarks.run_all --backend nccl --table

# Single node (OpenMPI)
mpirun -np 4 python -m benchmarks.collectives.broadcast --backend nccl

# Multi-node (2 nodes × 4 GPUs, OpenMPI)
mpirun -np 8 --hostfile hosts.txt python -m benchmarks.collectives.point_to_point \
    --backend nccl --output results/p2p.jsonl

# Cray MPICH (mpiexec) — use --cpu-bind none to avoid binding issues with GPU workloads
mpiexec -np 8 --cpu-bind none python -m benchmarks.collectives.broadcast --backend nccl
```

### On NCAR Derecho (PBS)

```bash
qsub <<'EOF'
#!/bin/bash
#PBS -A <project_code>
#PBS -q main
#PBS -l select=2:ncpus=64:ngpus=4
#PBS -l walltime=00:30:00
#PBS -N collectives_bench

module load conda
conda activate your_env

cd /glade/work/$USER/distributed-pytorch-benchmarks

mpiexec -np 8 --cpu-bind none python -m benchmarks.collectives.all_reduce \
    --backend nccl \
    --output results/all_reduce.jsonl
EOF
```

## Common flags

### Collectives benchmarks

```
--backend {nccl,gloo,mpi}     Default: nccl
--tensor_sizes N [N ...]       1-D tensor element counts to sweep (default: 1e3 through 1e8)
--warmup W                     Warmup iterations before timing (default: 5)
--iters I                      Timed iterations (default: 20)
--output PATH                  Append results as JSONL to this file
--table                        Display results in formatted table
```

### Dataloader benchmarks

```
--batch_sizes N [N ...]        Batch sizes to sweep (default: 32, 64, 128)
--num_workers N [N ...]        Worker counts to sweep (default: 0, 2, 4)
--dataset_size N               Number of samples in synthetic dataset (default: 10000)
--image_size N                 Image dimension (default: 224)
--num_channels N               Number of image channels (default: 3)
--num_classes N                Number of classification classes (default: 1000)
--warmup W                     Warmup batches before timing (default: 5)
--iters I                      Timed batches (default: 50)
--pin_memory                   Enable pinned memory for data loading
--output PATH                  Save results to CSV file
--table                        Display results in formatted table
```

## Understanding the output

Each benchmark prints per-size timing and **algorithm bandwidth**:

```
--- all_reduce  |  elements: 10,000,000  |  38.15 MB ---
  avg: 0.001042 s  |  median: 0.001038 s  |  algobw: 54.91 GB/s
```

**Algorithm bandwidth** accounts for the data movement pattern of each
collective.  For all-reduce on a ring of N ranks the volume moved is
`2 * (N-1)/N * message_size`.  Comparing algobw against the theoretical
peak of your interconnect (e.g. ~300 GB/s for NVLink, ~25 GB/s per
InfiniBand HDR port) tells you how much headroom remains.

When `--output` is set, every measurement is appended as a JSON line:

```json
{
  "collective": "all_reduce",
  "backend": "nccl",
  "world_size": 8,
  "elements": 10000000,
  "size_bytes": 40000000,
  "avg_s": 0.001042,
  "median_s": 0.001038,
  "algobw_gbps": 54.91,
  "all_times_s": [0.00105, 0.00103, ...]
}
```

### Dataloader output

The dataloader benchmark prints a table showing how worker count affects throughput:

```
┏━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┓
┃ Batch Size ┃ Workers ┃ Prefetch ┃ Batch (ms) ┃ Data (ms) ┃ Compute (ms) ┃ Data % ┃ Samples/s ┃
┡━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━┩
│         64 │       0 │        2 │      72.28 │     65.95 │         6.33 │  91.2% │     885.4 │
│         64 │       4 │        2 │      16.65 │     10.36 │         6.29 │  62.2% │    3843.9 │
└────────────┴─────────┴──────────┴────────────┴───────────┴──────────────┴────────┴───────────┘
```

Key metrics:
- **Data %** — percentage of batch time spent loading data (lower is better)
- **Samples/s** — training throughput

When `--output` is set, results are saved as CSV with a metadata header:

```csv
# Timestamp             : 2026-02-02 23:11:06
# GPU Model             : NVIDIA A100-SXM4-40GB
# ...
backend,benchmark_type,world_size,batch_size,num_workers,prefetch_factor,...
nccl-2.21.5,single_gpu,1,64,4,2,10000,224,16.65,6.65,...
```

## Interpreting results

A useful diagnostic workflow:

1. **Run all-reduce across message sizes.**  Plot bandwidth vs size.  You
   should see bandwidth climb and plateau near interconnect peak for large
   messages.  If it doesn't plateau, something is limiting the transport.

2. **Compare single-node vs multi-node.**  Intra-node uses NVLink/NVSwitch;
   inter-node uses InfiniBand or Ethernet.  A large drop at the node
   boundary points to network configuration issues.

3. **Run point-to-point between specific rank pairs.**  This isolates
   whether the problem is one bad link or systemic.  Useful for catching
   driver mismatches or firmware issues on specific NICs.

4. **Compare backends.**  Running the same test with `--backend gloo` gives
   a CPU-only baseline.  If NCCL is *slower* than Gloo, NCCL is
   misconfigured (wrong transport, missing GDR, etc.).

## Rank detection

The benchmarks automatically detect ranks from whichever launcher you use:

| Launcher | Environment variables used |
|----------|--------------------------|
| `torchrun` / `torch.distributed.launch` | `LOCAL_RANK`, `RANK`, `WORLD_SIZE` |
| OpenMPI `mpirun` | `OMPI_COMM_WORLD_LOCAL_RANK`, etc. |
| Cray MPICH (`mpiexec`) | `PMI_LOCAL_RANK`, `PMI_RANK`, `PMI_SIZE` |
| `mpi4py` available | Uses `MPI.COMM_WORLD` directly |

No code changes are needed when switching launchers.

algbw (Algorithm Bandwidth)                                                         
   
  - Simple formula: algbw = S / t (data size / time)                                  
  - Measures the effective throughput of the entire operation
  - Useful for estimating how long a given operation will take
  - Problem: For collectives, this number decreases as you add more ranks, even if the
   hardware is fully utilized. This makes it misleading for evaluating hardware
  efficiency.

  busbw (Bus Bandwidth)

  - Applies a correction factor to algbw to account for the fact that collectives
  inherently move more data as ranks increase
  - Reflects how well the hardware (NVLink, PCIe, network) is being utilized
  - Independent of the number of ranks, so you can directly compare it to the
  theoretical peak bandwidth of your interconnect

  Correction Factors

  The formula is busbw = algbw * factor, where the factor depends on the collective
  and n = number of ranks:

```
  ┌───────────────┬───────────┐
  │  Collective   │  Factor   │
  ├───────────────┼───────────┤
  │ AllReduce     │ 2*(n-1)/n │
  ├───────────────┼───────────┤
  │ ReduceScatter │ (n-1)/n   │
  ├───────────────┼───────────┤
  │ AllGather     │ (n-1)/n   │
  ├───────────────┼───────────┤
  │ Broadcast     │ 1         │
  ├───────────────┼───────────┤
  │ Reduce        │ 1         │
  ├───────────────┼───────────┤
  │ AlltoAll      │ (n-1)/n   │
  └───────────────┴───────────┘

```

  Practical Takeaway

  - Use algbw to predict wall-clock time for a given message size: time = size / algbw
  - Use busbw to evaluate whether your interconnect is performing at its peak — e.g.,
  compare it against your NVLink or InfiniBand spec bandwidth

  For example, with 8 GPUs doing AllReduce, busbw = algbw * 2*7/8 = algbw * 1.75. If
  your NVLink bandwidth is 900 GB/s and you see busbw close to that, your hardware is
  being used optimally.

```bash
  mpiexec -np 4 --cpu-bind none python -m benchmarks.dataloader.synthetic \
      --zarr /glade/derecho/scratch/$USER/era5_bench.zarr \
      --backend nccl --table \
      --warmup 2 --iters 5 \
      --batch_sizes 1 \
      --num_workers 0 2 4 8 \
      --prefetch_factor 2 \
      --output results/zarr_sweep.csv

```
