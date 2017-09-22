/*
 ==============================================================================

 RPCommon.cpp
 Copyright (C) 2016, 2017  G. Brinkmann

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

#include "RPCommon.hpp"
#include <math.h>
#include <stdlib.h>
#include <fstream>

// by http://stackoverflow.com/a/19841704
bool is_file_exists (const char *filename)
{
    std::ifstream infile(filename);
    return infile.good();
}

namespace RPGraph
{
    float get_random(float lowerbound, float upperbound)
    {
        return lowerbound + (upperbound-lowerbound) * static_cast <float> (random()) / static_cast <float> (RAND_MAX);
    }


    /* Definitions for Real2DVector */
    Real2DVector::Real2DVector(float x, float y): x(x), y(y) {};

    float Real2DVector::magnitude()
    {
        return sqrtf(x*x + y*y);
    }

    float Real2DVector::distance(RPGraph::Real2DVector to)
    {
        const float dx = (x - to.x)*(x - to.x);
        const float dy = (y - to.y)*(y - to.y);
        return sqrtf(dx*dx + dy*dy);
    }

    // Various operators on Real2DVector
    Real2DVector Real2DVector::operator*(float b)
    {
        return Real2DVector(this->x * b, this->y * b);
    }

    Real2DVector Real2DVector::operator/(float b)
    {
        return Real2DVector(this->x / b, this->y / b);
    }


    Real2DVector Real2DVector::operator+(Real2DVector b)
    {
        return Real2DVector(this->x + b.x, this->y + b.y);
    }


    Real2DVector Real2DVector::operator-(Real2DVector b)
    {
        return Real2DVector(this->x - b.x, this->y - b.y);
    }

    void Real2DVector::operator+=(Real2DVector b)
    {
        this->x += b.x;
        this->y += b.y;
    }

    Real2DVector Real2DVector::getNormalized()
    {
        return Real2DVector(this->x / magnitude(), this->y / magnitude());
    }

    Real2DVector Real2DVector::normalize()
    {
        const float m = magnitude();
        this->x /= m;
        this->y /= m;
        return *this;
    }

    /* Definitions for Coordinate */
    Coordinate::Coordinate(float x, float y) : x(x), y(y) {};

    // Various operators on Coordinate
    Coordinate Coordinate::operator+(float b)
    {
        return Coordinate(x + b, y + b);
    }

    Coordinate Coordinate::operator*(float b)
    {
        return Coordinate(this->x*b, this->y*b);
    }

    Coordinate Coordinate::operator/(float b)
    {
        return Coordinate(this->x/b, this->y/b);
    }

    Coordinate Coordinate::operator+(Real2DVector b)
    {
        return Coordinate(this->x + b.x, this->y + b.y);
    }

    Coordinate Coordinate::operator-(Coordinate b)
    {
        return Coordinate(this->x - b.x, this->y - b.y);
    }

    bool Coordinate::operator==(Coordinate b)
    {
        return (this->x == b.x && this->y == b.y);
    }

    float Coordinate::distance(RPGraph::Coordinate to)
    {
        return sqrtf((x - to.x)*(x - to.x) + (y - to.y)*(y - to.y));
    }

    float Coordinate::distance2(RPGraph::Coordinate to)
    {
        return (x - to.x)*(x - to.x) + (y - to.y)*(y - to.y);
    }

    void Coordinate::operator/=(float b)
    {
        this->x /= b;
        this->y /= b;
    }

    void Coordinate::operator+=(RPGraph::Coordinate b)
    {
        this->x += b.x;
        this->y += b.y;
    }

    void Coordinate::operator+=(RPGraph::Real2DVector b)
    {
        this->x += b.x;
        this->y += b.y;
    }

    int Coordinate::quadrant()
    {
        if (x <= 0)
        {
            if (y >= 0) return 0;
            else        return 3;

        }
        else
        {
            if (y >= 0) return 1;
            else        return 2;
        }
    }

    float distance(Coordinate from, Coordinate to)
    {
        const float dx = from.x - to.x;
        const float dy = from.y - to.y;
        return sqrtf(dx*dx + dy*dy);
    }

    float distance2(Coordinate from, Coordinate to)
    {
        const float dx = from.x - to.x;
        const float dy = from.y - to.y;
        return dx*dx + dy*dy;
    }

    Real2DVector normalizedDirection(Coordinate from, Coordinate to)
    {
        const float dx = from.x - to.x;
        const float dy = from.y - to.y;
        const float len = sqrtf(dx*dx + dy*dy);
        return Real2DVector(dx/len, dy/len);
    }

    Real2DVector direction(Coordinate from, Coordinate to)
    {
        const float dx = from.x - to.x;
        const float dy = from.y - to.y;
        return Real2DVector(dx, dy);
    }
}
