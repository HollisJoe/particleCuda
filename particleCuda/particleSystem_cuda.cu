/*
Created by Jane/Santaizi 3/19/2016
*/
#include <GL/freeglut.h>
#include <cstdlib>
#include <cstdio>
#include <string.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#include <helper_functions.h>
#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"

#include <device_launch_parameters.h>
#include "particleSystem_cuda_defines.h"
#include "particleSys_kernel.cuh"

#define DEFAULT_THREADS_PER_BLOCK     256
extern "C"
{


    /*-------- here goes the functions ----------*/
    void cudaGLInit_c(int argc, char **argv)
    {
        // use command-line specified CUDA device, otherwise use device with highest Gflops/s
        findCudaGLDevice(argc, (const char **)argv);
    }

    // set best performance device's property
    void cudaInit(int devID)
    {
        cudaDeviceProp dProp;
        cudaGetDeviceProperties(&dProp, devID);
        int threadsPerBlock =
            (dProp.major >= 2 ?
            2 * DEFAULT_THREADS_PER_BLOCK : DEFAULT_THREADS_PER_BLOCK);

        // here is the trick YOU SHOULD NEVER USE
        int *ptr = (int *)(&THREADS_MAX);
        *ptr = threadsPerBlock;
    }

    void cudaGraphicsGLRegisterBuffer_c(cudaGraphicsResource_t *resource, GLuint buffer, unsigned int flags)
    {
        checkCudaErrors(cudaGraphicsGLRegisterBuffer(resource, buffer, flags));
    }

    void cudaMalloc_c(void **devPtr, size_t size)
    {
        checkCudaErrors(cudaMalloc(devPtr, size));
    }

    void cudaFree_c(void *devPtr)
    {
        checkCudaErrors(cudaFree(devPtr));
    }

    void cudaSetParams(SimParam *hostParams)
    {
        /*
        I'm not sure if the prototype appearing in the CUDA Runtime API document should better read as `
        template<class T>
        cudaError_t cudaGetSymbolAddress (void **devPtr, const T symbol)
        */
        checkCudaErrors(cudaMemcpyToSymbol(d_cParam, hostParams, sizeof(SimParam)));
    }

    void cudaGraphicsUnregisterResource_c(cudaGraphicsResource_t cuda_vbo_resource)
    {
        checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource));
    }

    void cudaGraphicsMapResources_c(int count, cudaGraphicsResource_t *cuda_vbo_resource)
    {
        checkCudaErrors(cudaGraphicsMapResources(count, cuda_vbo_resource, 0));
    }

    void cudaGraphicsUnmapResources_c(int count, cudaGraphicsResource_t *cuda_vbo_resource)
    {
        checkCudaErrors(cudaGraphicsUnmapResources(count, cuda_vbo_resource, 0));
    }

    void cudaGraphicsResourceGetMappedPointer_c(void **devPtr, size_t *size, cudaGraphicsResource_t resource)
    {
        checkCudaErrors(cudaGraphicsResourceGetMappedPointer(devPtr, size,
            resource));
    }

    void cudaMemcpy_c(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
    {
        checkCudaErrors(cudaMemcpy(dst, src, count, kind));
    }

    
    /* which only apply gravity to integrate system */
    void integrateSystem(float *pos, float *vel, float deltaTime, uint numParticles)
    {
        thrust::device_ptr<float4> d_pos4((float4 *)pos);
        thrust::device_ptr<float4> d_vel4((float4 *)vel);

        thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(d_pos4, d_vel4)),
            thrust::make_zip_iterator(thrust::make_tuple(d_pos4 + numParticles, d_vel4 + numParticles)),
            integrate_functor(deltaTime));
    }

    /*
    @param return: uint2.x blocks
    */
    inline uint2 calcGridSize(uint numThreads, uint blockSize)
    {
        uint2 size;
        size.y = (uint)THREADS_MAX < blockSize ? (uint)THREADS_MAX : blockSize;
        size.x = (uint)ceilf((float)numThreads / size.y);
        return size;
    }

    void calcHash(uint *gridParticleHash, uint *gridParticleIndex, float *pos, uint numParticles)
    {
        uint2 gridSize = calcGridSize(numParticles, THREADS_MAX);
        calcHash_kernel<<<gridSize.x,gridSize.y>>>(gridParticleHash, gridParticleIndex, reinterpret_cast<float4 *>(pos), numParticles);
    }




}