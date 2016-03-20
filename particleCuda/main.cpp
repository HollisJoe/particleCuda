/*
Created by Jane/Santaizi 3/19/2016
*/

#include <GL/glew.h>
#include <GL/wglew.h>
#include <GL/freeglut.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "helper_cuda.h"
#include "helper_cuda_gl.h"

// particle system
#include "particleSystem.h"

using namespace particleSys;

uint width = 640;
uint height = 480;


// initialize OpenGL
void initGL(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutCreateWindow("ParticleSystem_CUDA Copyright. Jane/Santaizi 2016");

    glewInit();

    if (!glewIsSupported("GL_VERSION_2_0 GL_VERSION_1_5 GL_ARB_multitexture GL_ARB_vertex_buffer_object"))
    {
        fprintf(stderr, "Required OpenGL extensions missing.");
        exit(EXIT_FAILURE);
    }

#if defined (WIN32)

    if (wglewIsSupported("WGL_EXT_swap_control"))
    {
        // disable vertical sync
        wglSwapIntervalEXT(0);
    }

#endif

    glEnable(GL_DEPTH_TEST);
    glClearColor(0.25, 0.25, 0.25, 1.0);

    glutReportErrors();
}

int cudaGLInit(int argc, char **argv)
{
    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    return findCudaGLDevice(argc, (const char **)argv);
}


int main(int argc, char **argv)
{
    initGL(&argc, argv);
    int devID = cudaGLInit(argc, argv);
    ParticleSystem *pSys = new ParticleSystem(18000, uint3{ 64, 64, 64 }, devID);
    pSys->reset(ParticleSystem::CONFIG_GRID);
    
    delete pSys;
}
