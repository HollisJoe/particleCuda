
#ifndef __PARTICLESYSTEM_CUDA_DEFINES_H__
#define __PARTICLESYSTEM_CUDA_DEFINES_H__

#include <vector_types.h>
typedef unsigned int uint;
// simulation parameters
typedef struct
{
    // phyx
    float3      gravity;
    float       globalDamping;


    // configure
    uint3       gridSize;
    uint        numCells;
    float3      worldOrigin;
    float3      cellSize;
    uint        numParticles;

    // particle attribs
    float       particleRadius;
    float       particleMass;
    float       spring; // particle to particle
    float       damping;
    float       shear;
    float       attraction;
    float       boundaryDamping; // particle to plane(of other object)

} SimParam;

#endif // !__PARTICLESYSTEM_CUDA_DEFINES_H__