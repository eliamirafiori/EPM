#pragma once

// Particle structure representing a charged particle in 3D space.
struct Particle {
    float x, y, z; // Position
    float q;       // Charge
};

// Function to create an array of particles with random positions and charges.
//
// N: Number of particles
// W, H, D: Dimensions of the space in which particles are placed
//
// Returns a pointer to an array of particles.
//
const Particle* epm_create_particles(int N, int W, int H, int D);

// Function to create a potential map initialized to zero.
//
// W, H: Dimensions of the potential map
//
// Returns a pointer to the potential map array.
//
float* epm_create_pmap_zeroed(int W, int H);

// Check if two potential maps are approximately equal within a certain tolerance.
//
// pmap_a, pmap_b: Pointers to the potential maps to compare
// W, H: Dimensions of the potential maps
//
// Returns true if the potential maps are approximately equal, false otherwise.
//
bool epm_check_pmap_slices(const float* pmap_a, const float* pmap_b, int W, int H);

// Implemenetation of the EPM library. 
// Define EPM_IMPLEMENTATION before including this file in one of your source files
// to create the implementation.
#ifdef EPM_IMPLEMENTATION

#include <cstdlib>
#include <algorithm>
#include <cmath>

const Particle* epm_create_particles(int N, int W, int H, int D) {
    
    Particle* particles = new Particle[N];
    for (int i = 0; i < N; ++i) {
        particles[i].x = static_cast<float>(rand()) / RAND_MAX * W;
        particles[i].y = static_cast<float>(rand()) / RAND_MAX * H;
        particles[i].z = static_cast<float>(rand()) / RAND_MAX * D;

        // Charge between -1 and 1
        particles[i].q = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    }
    return particles;
}

float* epm_create_pmap_zeroed(int W, int H) {
    float* pmap = new float[W * H];
    std::fill(pmap, pmap + W * H, 0.0f);
    return pmap;
}

bool epm_check_pmap_slices(const float* pmap_a, const float* pmap_b, int W, int H) {
    const float abs_tol = 1e-3f;
    const float rel_tol = 1e-4f;
    for (int i = 0; i < W * H; ++i) {
        float diff = std::abs(pmap_a[i] - pmap_b[i]);
        float mag  = std::max(std::abs(pmap_a[i]), std::abs(pmap_b[i]));
        if (diff > abs_tol + rel_tol * mag) {
            return false;
        }
    }
    return true;
}

#endif