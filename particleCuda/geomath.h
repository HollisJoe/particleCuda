/*
Created by Jane/Santaizi 3/19/2016
*/
#ifndef __GEOMATH_H__
#define __GEOMATH_H__

#include <stdlib.h>
#include <math.h>

namespace geomath
{

inline float frand()
{
    return rand() / (float)RAND_MAX;
}

inline float lerp(float a, float b, float t)
{
    return a + t*(b - a);
}

// create a color ramp
static void colorRamp(float t, float *r)
{
    const int ncolors = 7;
    float c[ncolors][3] =
    {
        { 1.0, 0.0, 0.0, },
        { 1.0, 0.5, 0.0, },
        { 1.0, 1.0, 0.0, },
        { 0.0, 1.0, 0.0, },
        { 0.0, 1.0, 1.0, },
        { 0.0, 0.0, 1.0, },
        { 1.0, 0.0, 1.0, },
    };
    t = t * (ncolors - 1);
    int i = (int)t;
    float u = t - floor(t);
    r[0] = lerp(c[i][0], c[i + 1][0], u);
    r[1] = lerp(c[i][1], c[i + 1][1], u);
    r[2] = lerp(c[i][2], c[i + 1][2], u);
}

}

#endif //!__GEOMATH_H__