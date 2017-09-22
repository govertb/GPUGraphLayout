/*
 ==============================================================================

 RPBarnesHutApproximator.hpp
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

#ifndef RPBarnesHutApproximator_hpp
#define RPBarnesHutApproximator_hpp

#include "RPGraph.hpp"
#include "RPGraphLayout.hpp"
#include "RPCommon.hpp"

namespace RPGraph
{
    class BarnesHutCell
    {
    public:
        void add_leafcell(int quadrant, float mass, Coordinate pos);
        float lb, rb, ub, bb;

        // BarnesHutCell always contain either a single particle, or subcells (at most 4).
        BarnesHutCell(Coordinate position, float length, Coordinate particle_position, float particle_mass);
        ~BarnesHutCell();

        Coordinate cell_center, mass_center;
        nid_t num_subparticles = 0;
        float total_mass;
        const float length;   // length of a cell = width = height
        BarnesHutCell *sub_cells[4] = {nullptr, nullptr, nullptr, nullptr}; // per quadrant.

        void insertParticle(Coordinate particle_position, float particle_mass);
    };

    class BarnesHutApproximator
    {
    private:
        GraphLayout &layout;
        BarnesHutCell *root_cell = nullptr;
        const float theta;

    public:
        BarnesHutApproximator(float theta, GraphLayout &layout);
        Real2DVector approximateForce(Coordinate particle_pos, float particle_mass, float theta);
        void insertParticle(Coordinate particle_position, float particle_mass);

        void rebuild();
        void setTheta(float theta);
    };
}

#endif /* RPBarnesHutApproximator_hpp */
