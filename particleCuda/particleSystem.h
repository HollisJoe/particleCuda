/*
Created by Jane/Santaizi 3/19/2016
*/


#ifndef __PARTICLESYSTEM_H__
#define __PARTICLESYSTEM_H__

typedef unsigned int        uint; //declared as gobal

#include <GL/freeglut.h>
#include <vector_types.h>
#include "particleSystem_cuda_defines.h"

namespace particleSys
{


class ParticleSystem
{
    public:
        ParticleSystem(uint numParticles, uint3 gridsize, int devID, bool bUseVBO = true);
        // ParticleSystem(SimParam *param, int devID, bool bUseVBO = true);
        ~ParticleSystem();

    protected:
        void            init();
        void            deinit();
        void            update(float deltaTime);
        GLuint          createVBO(uint size);

    protected:
        bool            m_bInitialized;
        uint            m_numParticles;
        uint3           m_gridSize;
        uint            m_bUseVBO;
        uint            m_numGridCells;

        //              System params
        SimParam        m_param;

        //              openGL vbo
        GLuint                   m_posVbo;
        cudaGraphicsResource_t   m_cudaPosVboResource;

        /* Take some other rendering work - will be removed soon */
#define PARTICLE_COLOR      1  // You can set Zero to unuse it
#ifdef PARTICLE_COLOR
        GLuint                   m_colorVbo;
        cudaGraphicsResource_t   m_cudaColorVboResource;
#endif //!PARTICLE_COLOR

        //              Alloc Memorys
        float*          m_hPos;
        float*          m_hVel;
        float*          m_dVel;
        float*          m_dSortedPos;
        float*          m_dSortedVel;
        uint*           m_dCellStart;
        uint*           m_dCellEnd;
        uint*           m_dGridParticleHash;
        uint*           m_dGridParticleIndex;
        //              Just for debug dump
        uint*           m_hCellStart;
        uint*           m_hCellEnd;

#define PARTICLE_SYSTEM_EXTENSION 1
#ifdef PARTICLE_SYSTEM_EXTENSION // extensions which completes in $(filename)_ext.cpp
    public:
        enum ParticleConfig
        {
            CONFIG_RANDOM,
            CONFIG_GRID,
            _NUM_CONFIGS
        };
        enum ParticleArray
        {
            POSITION,
            VELOCITY,
        };

        void            reset(ParticleConfig config);
    protected:
        void            setArray(ParticleArray array, const float *data, int start, int count);
        void            initGrid(uint *size, float spacing, float jitter, uint numParticles);

#endif //!PARTICLE_SYSTEM_EXTENSION
};

}
#endif //!__PARTICLESYSTEM_H__
