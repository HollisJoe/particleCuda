/*
Created by Jane/Santaizi 3/19/2016
*/

#ifndef __PARTICLESYS_KERNEL_H__
#define __PARTICLESYS_KERNEL_H__

#include <helper_math.h>
#include "particleSystem_cuda_defines.h"




/*-------- constant variable resides on device ---------*/
__constant__ SimParam d_cParam; // d_c stands for device_constant_
/*-------- properties ---------*/
static const int THREADS_MAX = 256;

struct integrate_functor
{
    float deltaTime;

    __host__ __device__ integrate_functor(float delta_time) : deltaTime(delta_time) {}

    template <typename Tuple>
    __device__ void operator()(Tuple t)
    {
        volatile float4 posData = thrust::get<0>(t);
        volatile float4 velData = thrust::get<1>(t); 
        float3 pos = make_float3(posData.x, posData.y, posData.z);
        float3 vel = make_float3(velData.x, velData.y, velData.z);

        // apply gravity
        vel += d_cParam.gravity * deltaTime;
        // apply global damping
        vel *= d_cParam.globalDamping;

        // new position = old position + velocity * deltaTime
        pos += vel * deltaTime;

        // set this to zero to disable collisions with cube sides
#if 0

        if (pos.x > 1.0f - params.particleRadius)
        {
            pos.x = 1.0f - params.particleRadius;
            vel.x *= params.boundaryDamping;
        }

        if (pos.x < -1.0f + params.particleRadius)
        {
            pos.x = -1.0f + params.particleRadius;
            vel.x *= params.boundaryDamping;
        }

        if (pos.y > 1.0f - params.particleRadius)
        {
            pos.y = 1.0f - params.particleRadius;
            vel.y *= params.boundaryDamping;
        }

        if (pos.z > 1.0f - params.particleRadius)
        {
            pos.z = 1.0f - params.particleRadius;
            vel.z *= params.boundaryDamping;
        }

        if (pos.z < -1.0f + params.particleRadius)
        {
            pos.z = -1.0f + params.particleRadius;
            vel.z *= params.boundaryDamping;
        }

#endif
        // hit the ground
        if (pos.y < -1.0f + d_cParam.particleRadius)
        {
            pos.y = -1.0f + d_cParam.particleRadius;
            vel.y *= d_cParam.boundaryDamping;
        }

        // store new position and velocity
        thrust::get<0>(t) = make_float4(pos, posData.w);
        thrust::get<1>(t) = make_float4(vel, velData.w);
    }
};

__device__ int3 calcGridPos(float3 p)
{
    int3 gridPos;
    gridPos.x = (int)floorf((p.x - d_cParam.worldOrigin.x) / d_cParam.cellSize.x);
    gridPos.y = (int)floorf((p.y - d_cParam.worldOrigin.y) / d_cParam.cellSize.y);
    gridPos.z = (int)floorf((p.z - d_cParam.worldOrigin.z) / d_cParam.cellSize.z);
    return gridPos;
}

// calculate position hash based on Z-order curve hash algorithm
// link: https://en.wikipedia.org/wiki/Z-order_curve
__device__ uint calcGridHash(int3 gridPos)
{
    // wrap grid, assumes size is power of 2
    gridPos.x &= d_cParam.gridSize.x - 1;
    gridPos.y &= d_cParam.gridSize.y - 1;
    gridPos.z &= d_cParam.gridSize.z - 1;
    return gridPos.z*gridPos.y*gridPos.x + gridPos.y*gridPos.x + gridPos.x;
}


/*
@param output gridParticleHash
@param output gridParticleIndex
@param input  pos
@param input  numParticles
*/
__global__ void calcHash_kernel(uint *gridParticleHash, uint *gridParticleIndex, float4 *pos, uint numParticles)
{
    uint index = blockIdx.x * blockIdx.y + threadIdx.x;
    if (index > numParticles) return;

    volatile float4 p = pos[index]; // comes 4 in a roll
    int3 gridPos = calcGridPos(float3{ p.x, p.y, p.z });
    uint gridHash = calcGridHash(gridPos);
    gridParticleHash[index] = gridHash;
    gridParticleIndex[index] = index;
}


#endif //!__PARTICLESYS_KERNEL_H__