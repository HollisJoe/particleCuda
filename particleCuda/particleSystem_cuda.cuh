/*
Created by Jane/Santaizi 3/19/2016
*/

// pre-declaration to export functions implemented in particleSystem_cuda.cu

extern "C"
{
    void            cudaGLInit_c(int argc, char **argv);
    void            cudaMalloc_c(void **devPtr, size_t size);
    void            cudaFree_c(void *devPtr);
    void            cudaMemcpy_c(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
    void            cudaGraphicsGLRegisterBuffer_c(cudaGraphicsResource_t *resource, GLuint buffer, unsigned int flags);
    void            cudaGraphicsUnregisterResource_c(cudaGraphicsResource_t cuda_vbo_resource);
    void            cudaGraphicsMapResources_c(int count, cudaGraphicsResource_t *cuda_vbo_resource);
    void            cudaGraphicsUnmapResources_c(int count, cudaGraphicsResource_t *cuda_vbo_resource);
    void            cudaGraphicsResourceGetMappedPointer_c(void **devPtr, size_t *size, cudaGraphicsResource_t resource);

    /* customized cuda_methods , above is on general purpose */
    void            cudaInit(int devID);
    void            cudaSetParams(SimParam *hostParams);
    
    /* system cuda methods , apart from cuda_direct functions above */
    void            integrateSystem(float *pos, float *vel, float deltaTime, uint numParticles);
    void            calcHash(uint *gridParticleHash, uint *gridParticleIndex, float *pos, uint numParticles);

}
