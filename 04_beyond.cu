
#define EPM_IMPLEMENTATION
#define PPROG_IMPLEMENTATION
#include "external/epm.hpp"

#include <timer.cuh>

#include <cstdio>

// CONFIGURATION
#define TILE_SIZE 32

// Set to 1 to use CUTOFF
#define USE_CUTOFF 1

__global__ void calculate_z_slice(
    float* pmap,        // The potential map slice (width x height) 
    const int width,    // Width of the slice
    const int height,   // Height of the slice
    const int z,        // What z slice we are calculating
    const Particle* ps, // Particles of the system
    const int n         // Number of particles
) {
    // Declare shared memory for a tile of particles
    __shared__ Particle s_particles[TILE_SIZE * TILE_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0.0f;

    // Loop over particles in chunks (tiles)
    int num_tiles = (n + (TILE_SIZE * TILE_SIZE) - 1) / (TILE_SIZE * TILE_SIZE);
    
    for (int t = 0; t < num_tiles; t++) {
        // Collaborative Load: Each thread loads ONE particle into shared memory
        int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
        int particle_idx = t * (blockDim.x * blockDim.y) + thread_id;

        if (particle_idx < n) {
            s_particles[thread_id] = ps[particle_idx];
        }
        
        // SYNC: Wait until the whole block has finished loading the tile
        __syncthreads();

        // Compute using shared memory instead of global memory
        int particles_in_this_tile = min((int)(blockDim.x * blockDim.y), n - t * (int)(blockDim.x * blockDim.y));
        
        for (int i = 0; i < particles_in_this_tile; i++) {
            float dx = (float)col - s_particles[i].x;
            float dy = (float)row - s_particles[i].y;
            float dz = (float)z - s_particles[i].z;

            // Adding a small epsilon to avoid division by zero if particle is exactly at (col, row, z)
            float d = (dx*dx + dy*dy + dz*dz) + 1e-9f;
            result += s_particles[i].q * __frcp_rn(d);
        }

        // SYNC: Wait for everyone to finish computing before loading the next tile
        __syncthreads();
    }

    // Write result back to global memory
    if (row < height && col < width) {
        pmap[width * row + col] = result;
    }
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
        ((H) + TILE_SIZE - 1) / TILE_SIZE); // (64/16, 64/16, 1) -> (4, 4, 1)

    // Create CUDA streams
    cudaStream_t streamP, streamA, streamB;
    cudaStreamCreate(&streamP);
    cudaStreamCreate(&streamA);
    cudaStreamCreate(&streamB);

    // Create 10.000 particles in a 64x64x64 space
    const Particle* h_particles = epm_create_particles(N, W, H, D);
    Particle* d_particles;
    cudaMalloc((void**)&d_particles, N * sizeof(Particle));

    // Create two potential maps slices of size 64x64 initialized to zero
    float* h_pmapA = epm_create_pmap_zeroed(W, H);
    float* h_pmapB = epm_create_pmap_zeroed(W, H);

    // Allocate matrices in GPU memory
    float* d_pmapA;
    cudaError_t err = cudaMalloc((void**)&d_pmapA, bytes);
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    float* d_pmapB;
    err = cudaMalloc((void**)&d_pmapB, bytes);
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // Grouping all uploads
    cudaMemcpyAsync(d_particles, h_particles, N * sizeof(Particle), cudaMemcpyHostToDevice, streamP);
    cudaMemcpyAsync(d_pmapA, h_pmapA, bytes, cudaMemcpyHostToDevice, streamA);
    cudaMemcpyAsync(d_pmapB, h_pmapB, bytes, cudaMemcpyHostToDevice, streamB);

    // Calculate the potential map slice at z=32
    const int Z = 32;
    calculate_z_slice<<<blocks, threads, 0, streamA>>>(d_pmapA, W, H, Z, d_particles, N);
    calculate_z_slice<<<blocks, threads, 0, streamB>>>(d_pmapB, W, H, Z, d_particles, N);

    // Move result matrices to CPU memory
    // Grouping all downloads
    cudaMemcpyAsync(h_pmapA, d_pmapA, bytes, cudaMemcpyDeviceToHost, streamA);
    cudaMemcpyAsync(h_pmapB, d_pmapB, bytes, cudaMemcpyDeviceToHost, streamB);

    // Gouping all synchronizations
    cudaStreamSynchronize(streamP);
    cudaStreamSynchronize(streamA);
    cudaStreamSynchronize(streamB);
    
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

    printf("EPM GPU BEYOND: %f ms\n\n", timer_gpu.elapsed_ms());

    // Clean up
    cudaStreamDestroy(streamP);
    cudaStreamDestroy(streamA);
    cudaStreamDestroy(streamB);
    
    cudaFreeHost(h_pmapA);
    cudaFreeHost(h_pmapB);

    cudaFree(d_particles);
    cudaFree(d_pmapA);
    cudaFree(d_pmapB);

    return 0;
}