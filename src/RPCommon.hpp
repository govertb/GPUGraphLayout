/*
 ==============================================================================

 RPCommon.hpp
 Copyright © 2016, 2017, 2018  G. Brinkmann

 This file is part of graph_viewer.

 graph_viewer is free software: you can redistribute it and/or modify
 it under the terms of version 3 of the GNU Affero General Public License as
 published by the Free Software Foundation.

 graph_viewer is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU Affero General Public License for more details.

 You should have received a copy of the GNU Affero General Public License
 along with graph_viewer.  If not, see <https://www.gnu.org/licenses/>.

 ==============================================================================
*/

#ifndef RPCommonUtils_hpp
#define RPCommonUtils_hpp

#ifdef __NVCC__
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

#define cudaCatchError(ans) { assert_d((ans), __FILE__, __LINE__); }
inline void assert_d(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"error: (GPUassert) %s (error %d). %s:%d\n", cudaGetErrorString(code), code, file, line);
        if (abort) exit(code);
    }
}
#endif
bool is_file_exists (const char *filename);

namespace RPGraph
{
    float get_random(float lowerbound, float upperbound);

    class Real2DVector
    {
    public:
        Real2DVector(float x, float y);
        float x, y;
        float magnitude();
        float distance(Real2DVector to); // to some other Real2DVector `to'

        // Varous operators on Real2DVector
        Real2DVector operator*(float b);
        Real2DVector operator/(float b);
        Real2DVector operator+(Real2DVector b);
        Real2DVector operator-(Real2DVector b);
        void operator+=(Real2DVector b);

        Real2DVector getNormalized();
        Real2DVector normalize();
    };

    class Coordinate
    {
    public:
        float x, y;
        Coordinate(float x, float y);

        // Various operators on Coordinate
        Coordinate operator+(float b);
        Coordinate operator*(float b);
        Coordinate operator/(float b);
        Coordinate operator+(Real2DVector b);
        Coordinate operator-(Coordinate b);
        bool operator==(Coordinate b);
        void operator/=(float b);
        void operator+=(Coordinate b);
        void operator+=(RPGraph::Real2DVector b);

        int quadrant(); // Of `this' wrt. (0,0).
        float distance(Coordinate to);
        float distance2(Coordinate to);

    };

    float distance(Coordinate from, Coordinate to);
    float distance2(Coordinate from, Coordinate to);

    Real2DVector normalizedDirection(Coordinate from, Coordinate to);
    Real2DVector direction(Coordinate from, Coordinate to);

}

#endif /* RPCommonUtils_hpp */
