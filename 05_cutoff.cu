#define EPM_IMPLEMENTATION
#define PPROG_IMPLEMENTATION
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/sort.h>

#include <cstdio>
#include <timer.cuh>

#include "external/epm.hpp"

#define TILE_SIZE 16  // Balanced for 2D grids

// -----------------------------------------------------------------------------
// PREPROCESSING KERNELS
// -----------------------------------------------------------------------------

// Assign each particle to a 1D Bin ID based on its 3D position
__global__ void compute_bin_ids(const Particle* ps, int* bin_ids, int* indices, int n, float bin_size, int3 grid_dims) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int bx = max(0, min((int)(ps[i].x / bin_size), grid_dims.x - 1));
    int by = max(0, min((int)(ps[i].y / bin_size), grid_dims.y - 1));
    int bz = max(0, min((int)(ps[i].z / bin_size), grid_dims.z - 1));

    bin_ids[i] = bx + by * grid_dims.x + bz * grid_dims.x * grid_dims.y;
    indices[i] = i;  // Save original index to reorder later
}

// Find where each bin starts and ends in the sorted particle list
__global__ void find_bin_boundaries(const int* sorted_bin_ids, int* starts, int* ends, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int current_bin = sorted_bin_ids[i];
    if (i == 0 || current_bin != sorted_bin_ids[i - 1]) starts[current_bin] = i;
    if (i == n - 1 || current_bin != sorted_bin_ids[i + 1]) ends[current_bin] = i + 1;
}

// Simple kernel to reorder particles into their new sorted slots
__global__ void reorder_particles_kernel(const Particle* ps, const int* indices, Particle* sorted_ps, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    sorted_ps[i] = ps[indices[i]];
}

// -----------------------------------------------------------------------------
// CALCULATION KERNELS
// -----------------------------------------------------------------------------

// CUTOFF KERNEL (Using Spatial Binning)
__global__ void calculate_z_slice(
    float* pmap, int width, int height, int z_coord,
    const Particle* sorted_ps, const int* cell_starts, const int* cell_ends,
    int3 grid_dims, float bin_size, float cutoff_sq) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row >= height) return;

    // Find which bin this grid point sits in
    int bin_x = (int)(col / bin_size);
    int bin_y = (int)(row / bin_size);
    int bin_z = (int)(z_coord / bin_size);

    float result = 0.0f;

    // Look at 27 neighboring bins (3x3x3 cube)
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int nx = bin_x + dx;
                int ny = bin_y + dy;
                int nz = bin_z + dz;

                if (nx >= 0 && nx < grid_dims.x && ny >= 0 && ny < grid_dims.y && nz >= 0 && nz < grid_dims.z) {
                    int bin_idx = nx + ny * grid_dims.x + nz * grid_dims.x * grid_dims.y;
                    int start = cell_starts[bin_idx];
                    int end = cell_ends[bin_idx];

                    if (start == -1) continue;  // Bin is empty

                    for (int i = start; i < end; i++) {
                        float dx_p = (float)col - sorted_ps[i].x;
                        float dy_p = (float)row - sorted_ps[i].y;
                        float dz_p = (float)z_coord - sorted_ps[i].z;
                        float dist_sq = dx_p * dx_p + dy_p * dy_p + dz_p * dz_p;

                        if (dist_sq < cutoff_sq) {
                            result += sorted_ps[i].q * __frcp_rn(dist_sq + 1e-9f);
                        }
                    }
                }
            }
        }
    }
    pmap[row * width + col] = result;
}

int main() {
    const int N = 50000;
    const int W = 256, H = 256, D = 256;
    const int Z = 32;
    const float MAX_DIST = 16.0f;

    auto timer_gpu = timerGPU{};
    timer_gpu.start();

    // ALLOCATION
    size_t pmap_bytes = W * H * sizeof(float);
    float *d_pmapA, *d_pmapB;
    cudaMalloc(&d_pmapA, pmap_bytes);
    cudaMalloc(&d_pmapB, pmap_bytes);

    Particle *d_particles, *d_sorted_ps;
    cudaMalloc(&d_particles, N * sizeof(Particle));
    cudaMalloc(&d_sorted_ps, N * sizeof(Particle));

    const Particle* h_particles = epm_create_particles(N, W, H, D);
    cudaMemcpy(d_particles, h_particles, N * sizeof(Particle), cudaMemcpyHostToDevice);

    // SPATIAL BINNING PREPROCESSING
    int3 grid_dims = {(int)(W / MAX_DIST) + 1, (int)(H / MAX_DIST) + 1, (int)(D / MAX_DIST) + 1};
    int num_bins = grid_dims.x * grid_dims.y * grid_dims.z;

    int *d_bin_ids, *d_indices, *d_cell_starts, *d_cell_ends;
    cudaMalloc(&d_bin_ids, N * sizeof(int));
    cudaMalloc(&d_indices, N * sizeof(int));
    cudaMalloc(&d_cell_starts, num_bins * sizeof(int));
    cudaMalloc(&d_cell_ends, num_bins * sizeof(int));
    cudaMemset(d_cell_starts, -1, num_bins * sizeof(int));

    int blocks_N = (N + 255) / 256;
    compute_bin_ids<<<blocks_N, 256>>>(d_particles, d_bin_ids, d_indices, N, MAX_DIST, grid_dims);

    // Use Thrust to sort particles by their Bin ID
    thrust::device_ptr<int> th_bin_ids(d_bin_ids);
    thrust::device_ptr<int> th_indices(d_indices);
    thrust::sort_by_key(th_bin_ids, th_bin_ids + N, th_indices);

    // Reorder particles based on the sorted indices
    reorder_particles_kernel<<<blocks_N, 256>>>(d_particles, d_indices, d_sorted_ps, N);
    find_bin_boundaries<<<blocks_N, 256>>>(d_bin_ids, d_cell_starts, d_cell_ends, N);

    // EXECUTION
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((W + TILE_SIZE - 1) / TILE_SIZE, (H + TILE_SIZE - 1) / TILE_SIZE);

    calculate_z_slice<<<blocks, threads>>>(d_pmapA, W, H, Z, d_sorted_ps, d_cell_starts, d_cell_ends, grid_dims, MAX_DIST, MAX_DIST * MAX_DIST);
    calculate_z_slice<<<blocks, threads>>>(d_pmapB, W, H, Z, d_sorted_ps, d_cell_starts, d_cell_ends, grid_dims, MAX_DIST, MAX_DIST * MAX_DIST);

    // VERIFICATION
    float* h_pmapA = (float*)malloc(pmap_bytes);
    float* h_pmapB = (float*)malloc(pmap_bytes);
    cudaMemcpy(h_pmapA, d_pmapA, pmap_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pmapB, d_pmapB, pmap_bytes, cudaMemcpyDeviceToHost);

    if (epm_check_pmap_slices(h_pmapA, h_pmapB, W, H))
        printf("Potential maps are approximately equal.\n");
    else
        printf("Potential maps are NOT approximately equal.\n");

    timer_gpu.stop();
    printf("EPM GPU CUTOFF: %f ms\n\n", timer_gpu.elapsed_ms());

    // Clean up
    cudaFree(d_particles);
    cudaFree(d_sorted_ps);
    cudaFree(d_pmapA);
    cudaFree(d_pmapB);
    cudaFree(d_bin_ids);
    cudaFree(d_indices);
    cudaFree(d_cell_starts);
    cudaFree(d_cell_ends);
    return 0;
}