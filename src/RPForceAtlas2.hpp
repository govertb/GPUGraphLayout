/*
 ==============================================================================

 RPForceAtlas2.hpp
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

#ifndef RPForceAtlas2_hpp
#define RPForceAtlas2_hpp

#include "RPLayoutAlgorithm.hpp"
#include "RPBarnesHutApproximator.hpp"

namespace RPGraph
{
    class ForceAtlas2 : public LayoutAlgorithm
    {
        public:
            ForceAtlas2(GraphLayout &layout);
            ~ForceAtlas2();

            virtual void doStep() = 0;
            void doSteps(int n);
            void setScale(float s);
            void setGravity(float s);
            float mass(nid_t n);
            bool prevent_overlap, use_barneshut, use_linlog, strong_gravity;

        protected:
            int iteration;
            float k_r, k_g; // scalars for repulsive and gravitational force.
            float delta; // edgeweight influence.
            float global_speed;

            // Parameters used in adaptive temperature
            float speed_efficiency, jitter_tolerance;
            float k_s, k_s_max; // magic constants related to swinging.
            
            // Barnes-Hut parameters
            float theta;   // Accuracy
            float epssq;   // Softening (Epsilon, squared)
            float itolsq;  // Inverse tolerance, squared
    };
}
#endif
