/*
 ==============================================================================

 RPForceAtlas2.hpp
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

#ifndef RPForceAtlas2_hpp
#define RPForceAtlas2_hpp

#include "RPGraphLayout.hpp"
#include "RPBarnesHutApproximator.hpp"


namespace RPGraph
{
    class FA2Layout : public GraphLayout
    {
    private:
        Real2DVector *forces, *prev_forces;
        int iteration;

        // Scalars for repulsive and gravitational force.
        float k_r, k_g;
        float delta; // edgeweight influence.
        float global_speed;

        // Parameters used in adaptive temperature
        float speed_efficiency, jitter_tolerance;
        float k_s, k_s_max; // magic constants related to swinging.
        float theta;        // an accuracy parameter used for BarnesHut.


        BarnesHutApproximator *BH_Approximator = nullptr;

        float runningtimes[10][6] = {{5.5,},};


        float mass(nid_t n);
        float swg(nid_t n);            // swinging ..
        float s(nid_t n);              // swinging as well ..
        float tra(nid_t n);            // traction ..

        // Substeps of one step in layout process.
        void apply_repulsion(nid_t n);
        void apply_gravity(nid_t n);
        void apply_attract(nid_t n);
        void updateSpeeds();
        void apply_displacement(nid_t n);

    public:
        bool prevent_overlap, use_barneshut, use_linlog, strong_gravity;

        FA2Layout(UGraph &graph, float width, float height);
        ~FA2Layout();
        void doStep();
        void doSteps(int n);
        void setScale(float s);
        void setGravity(float s);

        void print_benchmarks();

    };
}

#endif /* ForceAtlas2_hpp */
