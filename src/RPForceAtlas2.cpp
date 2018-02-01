/*
 ==============================================================================

 RPForceAtlas2.cpp
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

#include "RPForceAtlas2.hpp"

namespace RPGraph
{
    ForceAtlas2::ForceAtlas2(GraphLayout &layout)
    : LayoutAlgorithm(layout)
    {
        iteration = 0;

        k_g = 1.0; // Gravitational force
        k_r = 1.0; // Repulsive force

        global_speed = 1.0;
        speed_efficiency = 1.0;
        jitter_tolerance = 1.0;

        k_s = 0.1;
        k_s_max = 10.0;
        theta = 1.0;

        delta = 0.0;

        prevent_overlap = false;
        strong_gravity = false;
        use_barneshut = true;
        use_linlog = false;

        layout.randomizePositions();
    }

    ForceAtlas2::~ForceAtlas2(){};

    void ForceAtlas2::doSteps(int n)
    {
        for (int i = 0; i < n; ++i) doStep();
    }

    void ForceAtlas2::setScale(float s)
    {
        k_r = s;
    }

    void ForceAtlas2::setGravity(float g)
    {
        k_g = g;
    }

    float ForceAtlas2::mass(nid_t n)
    {
        return layout.graph.degree(n) + 1.0;
    }
}
