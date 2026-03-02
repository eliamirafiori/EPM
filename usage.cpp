
#define EPM_IMPLEMENTATION
#include "external/epm.hpp"

#include <cstdio>

int main() {
    
    // Create 10.000 particles in a 64x64x64 space
    const Particle* particles = epm_create_particles(10000, 64, 64, 64);
    
    // Create two potential maps slices of size 64x64 initialized to zero
    float* pmapA = epm_create_pmap_zeroed(64, 64);
    float* pmapB = epm_create_pmap_zeroed(64, 64);
    pmapB[100] = 0.0004f;

    // Check if the potential maps are approximately equal
    if (epm_check_pmap_slices(pmapA, pmapB, 64, 64)) {
        printf("Potential maps are approximately equal.     \n");
    } else {
        printf("Potential maps are NOT approximately equal. \n");
    }

    delete[] particles;
    delete[] pmapA;
    delete[] pmapB;

    return 0;
}