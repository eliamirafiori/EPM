# Project: Parallel Electrostatic Potential Map (Direct Coulomb Summation)

![cover](doc/cover.png)

## Overview

The goal is to implement and progressively optimize the Direct Coulomb Summation (DCS) method for computing an electrostatic potential grid, showing how parallelization decisions affect performance.

### What is an Electrostatic Potential Map?

Every atom in a molecule carries a partial electric charge. These charges attract or repel other charged particles. An electrostatic potential map is a way of quantifying this influence: for every point in a 3D grid surrounding a molecule, we compute a single number that describes the net electric potential contributed by all atoms in the system. 

This map has direct physical meaning. Regions of high positive potential tend to attract negatively charged ions or electron-rich molecules. Regions of negative potential attract positively charged particless. By computing the map over a fine spatial grid, scientists get a detailed picture of where ions and solvent molecules are likely to sit, how proteins interact with drugs, and how charges redistribute during chemical reactions.

In practice, electrostatic potential maps are used heavily in molecular dynamics (MD) simulation.

### The Direct Coulomb Summation Method

There are several algorithms for computing electrostatic potential maps, ranging from approximate grid-based methods to highly accurate particle-mesh techniques. This project focuses on Direct Coulomb Summation (DCS), which is conceptually the most straightforward and physically the most accurate of the common approaches.

The mathematical foundation is simply Coulomb's law. The electrostatic potential $φ_i$ at a point $\vec{r}$ ($x$, $y$, $z$) due to a point charge $q_i$ located at position $\vec{r_i}$ is: 
$$
φ_i(\vec{r}) = \frac{q_i}{|\vec{r} - \vec{r_i}|}
$$
When there are N atoms, the total poptential at a grid point $r$ is just the sum of individual contributions:
$$
φ(\vec{r}) = \sum_{i = 0}^{N}{φ_i(\vec{r})} = \sum_{i = 0}^{N}{\frac{q_i}{|\vec{r} - \vec{r_i}|}}
$$

This must be evaluated at every grid point in a 3D lattice, giving a total operation count proportional to N × M, where M is the number of grid points. For a realistic biological system, a medium-sized protein with 50,000 atoms on a 256×256×256 grid, on a single CPU core this can take hours. This is why DCS is a natural candidate for GPU acceleration.

### Realistic scale

A 64×64×64 grid (~260K points) with 1,000–10,000 atoms is large enough to show meaningful GPU speedup but small enough to run quickly during development. For the cutoff extension, bumping to 128×128×128 with 50,000+ atoms makes the scalability advantage of binning clearly visible.

## Algorithm

We will focus only on a single z-slice of the 3D lattice (fixed z). The complete algorithm would iterate over all z slices, but in this project we focus only on a single slice to keep the scope manageable without losing any of the interesting parallel structure.

```c
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
```
This repository provides a utility library for generating particles and comparing two potential slices for correctness. Include it in exactly one .c file as follows:
```c
#define EPM_IMPLEMENTATION
#include <epm.h>
```
Look inside to see what is provided.

## What You Must Implement

You are required to implement at least three versions of the computation:

1. Sequential baseline. The C implementation shown above, used as ground truth and performance reference.
2. Naive parallel kernel. A CUDA kernel that assigns one thread per grid point and parallelizes the 2D slice in the most straightforward way. This is your first parallel version. Measure its speedup over the baseline and note any limitations.
3. Optimized parallel kernel. At least one further optimization over Version 2. You have freedom in what you choose to optimize. Explain what you changed, why you expected it to help, and what speedup you achieved.

Further versions beyond (3) are encouraged and will be evaluated positively.

## Bonus: Cutoff Summation with Spatial Binning (Not Required)

In the standard DCS algorithm, every particle contributes to every grid point regardless of distance. In practice, electrostatic influence falls off quickly with distance, and contributions from particles beyond some cutoff radius MAX_DIST are negligibly small. 

The cutoff approximation discards these distant contributions: a particle i contributes to a grid point j only if $|\vec{r_i} - \vec{r_j}|$ < MAX_DIST.

This reduces the work per grid point from O(N) to O(k), where k is the average number of particles within the cutoff sphere. The challenge is implementing this efficiently: naively checking all particles to find which ones are within range defeats the purpose.

The standard solution is spatial binning. Divide the simulation volume into a regular 3D grid of bins, each with side length MAX_DIST. For a given grid point, only particles in the immediately neighboring bins (at most 3×3×3 = 27 bins) can possibly be within the cutoff. By precomputing which particles fall in each bin, you can skip the vast majority of distance checks entirely.

For this bonus, implement a cutoff kernel using spatial binning and compare its performance against your best non-cutoff version.

## Deliverables

Your submission must include:

- Source code for all versions, clearly organized and commented
- A README.md with build instructions and hardware specifications of the machine used for benchmarking
- A results section reporting, for each version: execution time, speedup over baseline.
- A brief written explanation (a few sentences per version) of what optimization was applied and why it was expected to improve performance

## Evaluation Criteria
Your project will be evaluated on the following:

- Correctness: all parallel versions must pass the similarity threshold against the sequential baseline
- Coverage: at least three versions implemented and clearly differentiated
- Performance analysis: speedups are measured and reported accurately.
- Explanation quality: you demonstrate understanding of why each optimization works, not just that it does
- Code quality: kernels are readable, organized, and appropriately commented

The bonus is evaluated on correctness of the binning implementation, accuracy of the reported tradeoff, and quality of the performance analysis.