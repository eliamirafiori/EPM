
#define EPM_IMPLEMENTATION
#define PPROG_IMPLEMENTATION
#include <cstdio>
#include <timer.cuh>

#include "external/epm.hpp"

// CONFIGURATION
#define TILE_SIZE 32

__global__ void calculate_z_slice(
    float* pmap,         // The potential map slice (width x height)
    const int width,     // Width of the slice
    const int height,    // Height of the slice
    const int z,         // What z slice we are calculating
    const Particle* ps,  // Particles of the system
    const int n          // Number of particles
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Row in the output matrix
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Column in the output matrix

    float result = 0.0f;

    // Iterate over all particles
    for (int i = 0; i < n; i++) {
        float dx = (float)col - ps[i].x;
        float dy = (float)row - ps[i].y;
        float dz = (float)z - ps[i].z;

        float d = (dx * dx + dy * dy + dz * dz);
        result += ps[i].q / d;
    }

    pmap[width * row + col] = result;
}

int main() {
    const int N = 50000;
    const int W = 256;
    const int H = 256;
    const int D = 256;

    auto timer_gpu = timerGPU{};
    timer_gpu.start();

    // Define dimensions
    int size = W * H;
    size_t bytes = size * sizeof(float);

    // Setup blocks and threads count
    // Threads per block (The "Tile")
    // 16x16 = 256 threads. This is a multiple of 32 (a Warp)
    dim3 threads = dim3(TILE_SIZE, TILE_SIZE);

    // Blocks per grid (The "Cover")
    // We need enough tiles to cover the width and height
    dim3 blocks = dim3(
        ((W) + TILE_SIZE - 1) / TILE_SIZE,
        ((H) + TILE_SIZE - 1) / TILE_SIZE);  // (64/16, 64/16, 1) -> (4, 4, 1)

    // Create 10.000 particles in a 64x64x64 space
    const Particle* h_particles = epm_create_particles(N, W, H, D);
    Particle* d_particles;
    cudaMalloc((void**)&d_particles, N * sizeof(Particle));
    cudaMemcpy(d_particles, h_particles, N * sizeof(Particle), cudaMemcpyHostToDevice);

    // Create two potential maps slices of size 64x64 initialized to zero
    float* h_pmapA = epm_create_pmap_zeroed(W, H);
    float* h_pmapB = epm_create_pmap_zeroed(W, H);

    // Allocate matrices in GPU memory
    float* d_pmapA;
    cudaError_t err = cudaMalloc((void**)&d_pmapA, bytes);
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    cudaMemcpy(d_pmapA, h_pmapA, bytes, cudaMemcpyHostToDevice);

    float* d_pmapB;
    err = cudaMalloc((void**)&d_pmapB, bytes);
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    cudaMemcpy(d_pmapB, h_pmapB, bytes, cudaMemcpyHostToDevice);

    // Calculate the potential map slice at z=32
    const int Z = 32;
    calculate_z_slice<<<blocks, threads>>>(d_pmapA, W, H, Z, d_particles, N);
    calculate_z_slice<<<blocks, threads>>>(d_pmapB, W, H, Z, d_particles, N);

    // Move result matrices to CPU memory
    cudaMemcpy(h_pmapA, d_pmapA, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pmapB, d_pmapB, bytes, cudaMemcpyDeviceToHost);

    timer_gpu.stop();

    // Check if the potential maps are approximately equal
    if (epm_check_pmap_slices(h_pmapA, h_pmapB, W, H)) {
        printf("Potential maps are approximately equal.\n");
    } else {
        printf("Potential maps are NOT approximately equal.\n");
        for (int i = 0; i < W * H; i++) {
            if (h_pmapA[i] != h_pmapB[i]) {
                printf("Difference at index %d: h_pmapA=%f, h_pmapB=%f\n", i, h_pmapA[i], h_pmapB[i]);
            }
        }
    }

    printf("EPM GPU NAIVE: %f ms\n\n", timer_gpu.elapsed_ms());

    // Free memory
    delete[] h_particles;
    delete[] h_pmapA;
    delete[] h_pmapB;

    // Free cuda memory
    cudaFree(d_particles);
    cudaFree(d_pmapA);
    cudaFree(d_pmapB);

    return 0;
}