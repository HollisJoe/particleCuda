/*
Created by Jane/Santaizi 3/19/2016
*/
#include <assert.h>
#include <GL/glew.h>
#include "geomath.h"
#include "particleSystem.h"
#include "particleSystem_cuda.cuh"
using namespace particleSys;

static uint wrapToPower2(uint num)
{
    uint x = num - 1;
    int i = 0;
    while (x != 0)
    {
        x = x >> 1;
        ++i;
    }
    return 1 << i;
}

// gridsize should be power of 2, otherwise it will be wrap into ceil($(power of 2))
// any 0 in gridsize is undefined
ParticleSystem::ParticleSystem(uint numParticles, uint3 gridsize, int devID, bool bUseVBO) :
m_numParticles(numParticles),
m_gridSize(gridsize),
m_bUseVBO(bUseVBO),
m_bInitialized(false)
{
    m_gridSize.x = wrapToPower2(m_gridSize.x);
    m_gridSize.y = wrapToPower2(m_gridSize.y);
    m_gridSize.z = wrapToPower2(m_gridSize.z);

    m_numGridCells = m_gridSize.x*m_gridSize.y*m_gridSize.z;

    m_param = SimParam();
    // make default param
    {
        m_param.gravity = float3{ 0.0f, -10.f, 0.0f };
        m_param.globalDamping = 1.f;

        m_param.particleRadius = 1.0f / 64.0f;
        m_param.particleMass = 0.00003f;
        m_param.spring = 0.5f;
        m_param.damping = 0.02f;
        m_param.shear = 0.1f;
        m_param.attraction = 0.0f;
        m_param.boundaryDamping = -0.5f;

        m_param.gridSize = m_gridSize;
        m_param.numCells = m_numGridCells;
        m_param.worldOrigin = float3{ -1.f, -1.f, -1.f };
        float cellsize = m_param.particleRadius * 2;
        m_param.cellSize = float3{ cellsize, cellsize, cellsize };
        m_param.numParticles = m_numParticles;
    }
    init();
    //cudaInit(devID);
}
/* -temporarily deprecated
ParticleSystem::ParticleSystem(SimParam *param, int devID, bool bUseVBO) :
m_bInitialized(false),
m_bUseVBO(bUseVBO)
{
    m_param = *param;
    m_numParticles = m_param.numParticles;
    m_gridSize = m_param.gridSize;
    m_numGridCells = m_param.numCells;
}
*/


ParticleSystem::~ParticleSystem()
{
    deinit();
}

void ParticleSystem::init()
{
    assert(m_bUseVBO);
    uint memSize = sizeof(float) * 4 * m_numParticles;

    m_posVbo = createVBO(memSize);
    cudaGraphicsGLRegisterBuffer_c(&m_cudaPosVboResource, m_posVbo, cudaGraphicsMapFlagsNone);
    
    m_hPos = new float[m_numParticles * 4];
    m_hVel = new float[m_numParticles * 4];
    memset(m_hPos, 0, memSize);
    memset(m_hVel, 0, memSize);

    cudaMalloc_c((void **)&m_dVel, memSize);
    cudaMalloc_c((void **)&m_dSortedPos, memSize);
    cudaMalloc_c((void **)&m_dSortedVel, memSize);
    cudaMalloc_c((void **)&m_dCellStart, m_numGridCells*sizeof(uint));
    cudaMalloc_c((void **)&m_dCellEnd, m_numGridCells*sizeof(uint));
    cudaMalloc_c((void **)&m_dGridParticleHash, m_numParticles*sizeof(uint));
    cudaMalloc_c((void **)&m_dGridParticleIndex, m_numParticles*sizeof(uint));

#ifdef _DEBUG
    m_hCellStart = new uint[m_numGridCells];
    m_hCellEnd = new uint[m_numGridCells];
    memset(m_hCellStart, 0, m_numGridCells*sizeof(uint));
    memset(m_hCellEnd, 0, m_numGridCells*sizeof(uint));
#endif //!DEBUG

    /* Take some other rendering work - will be removed soon */
#ifdef PARTICLE_COLOR
    m_colorVbo = createVBO(memSize);
    cudaGraphicsGLRegisterBuffer_c(&m_cudaColorVboResource, m_colorVbo, cudaGraphicsMapFlagsNone);

    // fill color buffer
    glBindBufferARB(GL_ARRAY_BUFFER_ARB, m_colorVbo);
    GLfloat *data = (GLfloat *)glMapBufferARB(GL_ARRAY_BUFFER_ARB, GL_WRITE_ONLY_ARB);
    float *ptr = data;
    for (uint i = 0; i < m_numParticles; ++i)
    {
        float t = i / (float)m_numParticles;
        geomath::colorRamp(t, ptr);
        ptr += 3;
        *ptr++ = 1.f;
    }
    glUnmapBufferARB(GL_ARRAY_BUFFER_ARB);
    glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
#endif //!PARTICLE_COLOR
    
    cudaSetParams(&m_param);
    m_bInitialized = true;
}

void ParticleSystem::deinit()
{
    if (m_bUseVBO)
    {
        cudaGraphicsUnregisterResource_c(m_cudaPosVboResource);
        glDeleteBuffersARB(1, &m_posVbo);
    }
    delete[] m_hPos;
    delete[] m_hVel;
    cudaFree_c((void *)m_dVel);
    cudaFree_c((void *)m_dSortedPos);
    cudaFree_c((void *)m_dSortedVel);
    cudaFree_c((void *)m_dCellStart);
    cudaFree_c((void *)m_dCellEnd);
    cudaFree_c((void *)m_dGridParticleHash);
    cudaFree_c((void *)m_dGridParticleIndex);

#ifdef _DEBUG
    delete[] m_hCellStart;
    delete[] m_hCellEnd;
#endif //!DEBUG

    /* Take some other rendering work - will be removed soon */
#ifdef PARTICLE_COLOR
    if (m_bUseVBO)
    {
        cudaGraphicsUnregisterResource_c(m_cudaColorVboResource);
        glDeleteBuffersARB(1, &m_colorVbo);
    }
#endif //!PARTICLE_COLOR
}

GLuint ParticleSystem::createVBO(uint size)
{
    GLuint vbo;
    glGenBuffersARB(1, &vbo);
    glBindBufferARB(GL_ARRAY_BUFFER_ARB, vbo);
    glBufferDataARB(GL_ARRAY_BUFFER_ARB, size, NULL, GL_DYNAMIC_DRAW_ARB);
    glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
    return vbo;
}

void ParticleSystem::update(float deltaTime)
{
    if (!m_bInitialized)
        return;
    
    // set simulation parameters every time, it dose not cost much time
    cudaSetParams(&m_param);

    float *dPos;
    cudaGraphicsMapResources_c(1, &m_cudaPosVboResource);
    size_t size;
    cudaGraphicsResourceGetMappedPointer_c((void **)&dPos, &size, m_cudaPosVboResource);

    /* 
        now we get dPos and size
        its time to integrate position-diff from velocity within timeStep
    */
    // itegrate system get new position and velocity
    integrateSystem(dPos, m_dVel, deltaTime, m_numParticles);
    
    ///* use grid method to sort particles boosting collide phrase */
    //// calculate grid hash
    calcHash(m_dGridParticleHash, m_dGridParticleIndex, dPos, m_numParticles);

    //// sort particles by hash, since the data is space-time coherence, its first time costs
    //void sortParticles;

    //// find start and end of each cell
    //void findCellStartNEnd;

    //// particles self system collide
    //void systemCollide;

    //// particles collide with rigid body system
    //void crossSysCollide;

    //// update data into buffer

    cudaGraphicsUnmapResources_c(1, &m_cudaPosVboResource);
}


/* extensions */
void ParticleSystem::setArray(ParticleArray array, const float *data, int start, int count)
{
    assert(m_bInitialized);

    switch (array)
    {
    default:
    case POSITION:
    {
        if (m_bUseVBO)
        {
            cudaGraphicsUnregisterResource_c(m_cudaPosVboResource);
            glBindBufferARB(GL_ARRAY_BUFFER_ARB, m_posVbo);
            glBufferSubDataARB(GL_ARRAY_BUFFER_ARB, start * 4 * sizeof(float), count * 4 * sizeof(float), data);
            glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
            cudaGraphicsGLRegisterBuffer_c(&m_cudaPosVboResource, m_posVbo, cudaGraphicsMapFlagsNone);
        }
    }
    break;

    case VELOCITY:
        cudaMemcpy_c(m_dVel + start * 4 * sizeof(float), data, count * 4 * sizeof(float), cudaMemcpyHostToDevice);
        break;
    }
}

void ParticleSystem::reset(ParticleConfig config)
{
    switch (config)
    {
    default:
    case CONFIG_RANDOM:
    {
        int p = 0, v = 0;

        for (uint i = 0; i < m_numParticles; i++)
        {
            float point[3];
            point[0] = geomath::frand();
            point[1] = geomath::frand();
            point[2] = geomath::frand();
            m_hPos[p++] = 2 * (point[0] - 0.5f);
            m_hPos[p++] = 2 * (point[1] - 0.5f);
            m_hPos[p++] = 2 * (point[2] - 0.5f);
            m_hPos[p++] = 1.0f; // radius
            m_hVel[v++] = 0.0f;
            m_hVel[v++] = 0.0f;
            m_hVel[v++] = 0.0f;
            m_hVel[v++] = 0.0f;
        }
    }
    break;

    case CONFIG_GRID:
    {
        float jitter = m_param.particleRadius*0.01f;
        uint s = (int)ceilf(powf((float)m_numParticles, 1.0f / 3.0f));
        uint gridSize[3];
        gridSize[0] = gridSize[1] = gridSize[2] = s;
        initGrid(gridSize, m_param.particleRadius*2.0f, jitter, m_numParticles);
    }
    break;
    }

    setArray(POSITION, m_hPos, 0, m_numParticles);
    setArray(VELOCITY, m_hVel, 0, m_numParticles);
}

void ParticleSystem::initGrid(uint *size, float spacing, float jitter, uint numParticles)
{
    srand(1973);

    for (uint z = 0; z<size[2]; z++)
    {
        for (uint y = 0; y<size[1]; y++)
        {
            for (uint x = 0; x<size[0]; x++)
            {
                uint i = (z*size[1] * size[0]) + (y*size[0]) + x;

                if (i < numParticles)
                {
                    m_hPos[i * 4] = (spacing * x) + m_param.particleRadius - 1.0f + (geomath::frand()*2.0f - 1.0f)*jitter;
                    m_hPos[i * 4 + 1] = (spacing * y) + m_param.particleRadius - 1.0f + (geomath::frand()*2.0f - 1.0f)*jitter;
                    m_hPos[i * 4 + 2] = (spacing * z) + m_param.particleRadius - 1.0f + (geomath::frand()*2.0f - 1.0f)*jitter;
                    m_hPos[i * 4 + 3] = 1.0f;

                    m_hVel[i * 4] = 0.0f;
                    m_hVel[i * 4 + 1] = 0.0f;
                    m_hVel[i * 4 + 2] = 0.0f;
                    m_hVel[i * 4 + 3] = 0.0f;
                }
            }
        }
    }
}

