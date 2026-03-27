
#define EPM_IMPLEMENTATION
#include "external/epm.hpp"

#include <cstdio>

void calculate_z_slice(
    float* pmap,        // The potential map slice (width x height) 
    const int width,    // Width of the slice
    const int height,   // Height of the slice
    const int z,        // What z slice we are calculating
    const Particle* ps, // Particles of the system
    const int n         // Number of particles
) {
    // Iterate over all grid points in the 2D slice
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            
            float result = 0.0f;

            // Iterate over all particles
            for (int i = 0; i < n; i++) {

                float dx = (float)x - ps[i].x;
                float dy = (float)y - ps[i].y;
                float dz = (float)z - ps[i].z;

                float d = (dx*dx + dy*dy + dz*dz);
                result += ps[i].q / d;
            }

            pmap[width * y + x] = result;
        }
    }
} 

int main() {
    const int N = 1000;
    const int W = 64;
    const int H = 64;
    const int D = 64;

    // Create 10.000 particles in a 64x64x64 space
    const Particle* particles = epm_create_particles(N, W, H, D);
    
    // Create two potential maps slices of size 64x64 initialized to zero
    float* pmapA = epm_create_pmap_zeroed(W, H);
    float* pmapB = epm_create_pmap_zeroed(W, H);
    
    // Calculate the potential map slice at z=32
    const int Z = 32;
    calculate_z_slice(pmapA, W, H, Z, particles, N);

    // Check if the potential maps are approximately equal
    if (epm_check_pmap_slices(pmapA, pmapB, W, H)) {
        printf("Potential maps are approximately equal.\n");
    } else {
        printf("Potential maps are NOT approximately equal.\n");
        for (int i = 0; i < W * H; i++) {
            if (pmapA[i] != pmapB[i]) {
                printf("Difference at index %d: pmapA=%f, pmapB=%f\n", i, pmapA[i], pmapB[i]);
            }
        }
    }

    delete[] particles;
    delete[] pmapA;
    delete[] pmapB;

    return 0;
}