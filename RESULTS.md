# RESULTS

## Hardware Specifications

### CPU

- Model: Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz
- Physical Cores: 6 (per socket)
- Total Logical Processors (Threads): 12
- NUMA Nodes: 1 (Node0: CPUs 0-11)

### Memory (RAM)

- Total Capacity: 7.7 GiB
- Used: 1.9 GiB
- Available/Free: 5.8 GiB (Available) / 4.3 GiB (Free)

### GPU

- Model: NVIDIA GeForce GTX 1650
- Driver Version: 595.97
- Total VRAM: 4096 MiB (4 GiB)

## SOFTWARE & COMPILER

- Cuda compilation tools, release 13.0, V13.0.88
- C++ Standard: C++17

## PERFORMANCES

| Implementation | Execution Time (ms) | Speedup (vs. CPU) | Efficiency Notes |
| --- | --- | --- | --- |
| 01 Sequential (CPU) | **31,925.71** | 1.0x | Baseline (Single-thread) |
| 02 Naive (GPU) | 94.15 | 339.1x | Brute force parallelization |
| 03 Shared (GPU) | 80.43 | 396.9x | Reduced Global Memory traffic |
| 04 Beyond (GPU) | 75.40 | 423.4x | Optimized tiling/streams |
| 05 Cutoff (GPU) | 14.36 | **2,223.8x** | Spatial binning & O(k) complexity |
