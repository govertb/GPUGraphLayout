/*
 ==============================================================================

 RPCPUForceAtlas2.cpp
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

#include "RPCPUForceAtlas2.hpp"
#include <stdlib.h>
#include <math.h>
#include <limits>
#include <cmath>
#include <chrono>

namespace RPGraph
{
    // CPUForceAtlas2 definitions.
    CPUForceAtlas2::CPUForceAtlas2(GraphLayout &layout)
    :  ForceAtlas2(layout), BH_Approximator{theta, layout}
    {
        forces      = (Real2DVector *)malloc(sizeof(Real2DVector) * layout.graph.num_nodes());
        prev_forces = (Real2DVector *)malloc(sizeof(Real2DVector) * layout.graph.num_nodes());
        for (nid_t n = 0; n < layout.graph.num_nodes(); ++n)
        {
            forces[n]      = Real2DVector(0.0f, 0.0f);
            prev_forces[n] = Real2DVector(0.0f, 0.0f);
        }
    }

    CPUForceAtlas2::~CPUForceAtlas2()
    {
        free(forces);
        free(prev_forces);
    }

    void CPUForceAtlas2::apply_attract(nid_t n)
    {
        Real2DVector f = Real2DVector(0.0, 0.0);
        for (nid_t t : layout.graph.neighbors_with_geq_id(n))
        {
            // Here we define the magnitude of the attractive force `f_a'
            // *divided* by the length distance between `n' and `t', i.e. `f_a_over_d'
            float f_a_over_d;
            if (use_linlog)
            {
                float dist = layout.getDistance(n, t);
                f_a_over_d = dist == 0.0 ? std::numeric_limits<float>::max() : logf(1+dist) / dist;
            }

            else
            {
                f_a_over_d = 1.0;
            }

            f += layout.getDistanceVector(n, t) * f_a_over_d;

            //TODO: this is temporary, but required due to
            //      iteration over neighbors_with_geq_id
            forces[t] += layout.getDistanceVector(n, t) * (-f_a_over_d);

    //            forces[n] += getNormalizedDistanceVector(n, t) * f_a(n, t);
        }
        forces[n] += f;
    }

    void CPUForceAtlas2::apply_repulsion(nid_t n)
    {
        if (use_barneshut)
        {
            forces[n] += (BH_Approximator.approximateForce(layout.getCoordinate(n), mass(n), theta) * k_r);
        }

        else
        {
            for (nid_t t = 0; t < layout.graph.num_nodes(); ++t)
            {
                if (n == t) continue;
                float  distance = layout.getDistance(n, t);
                float f_r = distance == 0.0 ? std::numeric_limits<float>::max() : k_r * mass(n) * mass(t) / distance / distance;
                forces[n] += layout.getDistanceVector(n, t) * f_r;
            }
        }
    }

    void CPUForceAtlas2::apply_gravity(nid_t n)
    {
        float f_g, d;

        // `d' is the distance from `n' to the center (0.0, 0.0)
        d = std::sqrt(layout.getX(n)*layout.getX(n) + layout.getY(n)*layout.getY(n));
        if(d == 0.0) return;

        // Here we define the magnitude of the gravitational force `f_g'.
        if (strong_gravity)
        {
            f_g = k_g*mass(n);
        }

        else
        {
            f_g = k_g*mass(n) / d;
        }

        forces[n] += (Real2DVector(-layout.getX(n), -layout.getY(n)) * f_g);
    }

    // Eq. (8)
    float CPUForceAtlas2::swg(nid_t n)
    {
        return (forces[n] - prev_forces[n]).magnitude();
    }

    // Eq. (9)
    float CPUForceAtlas2::s(nid_t n)
    {
        return (k_s * global_speed)/(1.0f+global_speed*std::sqrt(swg(n)));
    }

    // Eq. (12)
    float CPUForceAtlas2::tra(nid_t n)
    {
        return (forces[n] + prev_forces[n]).magnitude() / 2.0;
    }

    void CPUForceAtlas2::updateSpeeds()
    {
        // The following speed-update procedure for ForceAtlas2 follows
        // the one by Gephi:
        // https://github.com/gephi/gephi/blob/6efb108718fa67d1055160f3a18b63edb4ca7be2/modules/LayoutPlugin/src/main/java/org/gephi/layout/plugin/forceAtlas2/ForceAtlas2.java

        // `Auto adjust speeds'
        float total_swinging = 0.0;
        float total_effective_traction = 0.0;
        for (nid_t nid = 0; nid < layout.graph.num_nodes(); ++nid)
        {
            total_swinging += mass(nid) * swg(nid); // Eq. (11)
            total_effective_traction += mass(nid) * tra(nid); // Eq. (13)
        }

        // We want to find the right jitter tollerance for this graph,
        // such that totalSwinging < tolerance * totalEffectiveTraction

        float estimated_optimal_jitter_tollerance = 0.05 * std::sqrt(layout.graph.num_nodes());
        float minJT = std::sqrt(estimated_optimal_jitter_tollerance);
        float jt = jitter_tolerance * fmaxf(minJT,
                                           fminf(k_s_max,
                                                 estimated_optimal_jitter_tollerance * total_effective_traction / powf(layout.graph.num_nodes(), 2.0)
                                                 )
                                           );
        float min_speed_efficiency = 0.05;

        // `Protect against erratic behavior'
        if (total_swinging / total_effective_traction > 2.0)
        {
            if (speed_efficiency > min_speed_efficiency) speed_efficiency *= 0.5;
            jt = fmaxf(jt, jitter_tolerance);
        }

        // `Speed efficiency is how the speed really corrosponds to the swinging vs. convergence tradeoff.'
        // `We adjust it slowly and carefully'
        float targetSpeed = jt * speed_efficiency * total_effective_traction / total_swinging;

        if (total_swinging > jt * total_effective_traction)
        {
            if (speed_efficiency > min_speed_efficiency)
            {
                speed_efficiency *= 0.7;
            }
        }
        else if (global_speed < 1000)
        {
            speed_efficiency *= 1.3;
        }

        // `But the speed shouldn't rise much too quickly, ... would make convergence drop dramatically'.
        float max_rise = 0.5;
        global_speed += fminf(targetSpeed - global_speed, max_rise * global_speed);
    }

    void CPUForceAtlas2::apply_displacement(nid_t n)
    {
        if (prevent_overlap)
        {
            // Not yet implemented
            exit(EXIT_FAILURE);
        }

        else
        {

            float factor = global_speed / (1.0 + std::sqrt(global_speed * swg(n)));
            layout.moveNode(n, forces[n] * factor);
        }
    }

    void CPUForceAtlas2::doStep()
    {
        auto starttime = std::chrono::high_resolution_clock::now();
        auto endtime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> runningtime;
        std::chrono::microseconds runningtime_us;

        starttime = std::chrono::high_resolution_clock::now();
        if (use_barneshut)
            BH_Approximator.rebuild();
        endtime = std::chrono::high_resolution_clock::now();
        runningtime = (endtime - starttime);
        runningtime_us = std::chrono::duration_cast<std::chrono::microseconds>(runningtime);
        //runningtimes[iteration][2] = runningtime_us.count();

        starttime = std::chrono::high_resolution_clock::now();
            for (nid_t n = 0; n < layout.graph.num_nodes(); ++n) apply_gravity(n);
        endtime = std::chrono::high_resolution_clock::now();
        runningtime = (endtime - starttime);
        runningtime_us = std::chrono::duration_cast<std::chrono::microseconds>(runningtime);
        //runningtimes[iteration][0] = runningtime_us.count();

        starttime = std::chrono::high_resolution_clock::now();
            for (nid_t n = 0; n < layout.graph.num_nodes(); ++n)    apply_attract(n);
        endtime = std::chrono::high_resolution_clock::now();
        runningtime = (endtime - starttime);
        runningtime_us = std::chrono::duration_cast<std::chrono::microseconds>(runningtime);
        //runningtimes[iteration][1] = runningtime_us.count();

        starttime = std::chrono::high_resolution_clock::now();
            for (nid_t n = 0; n < layout.graph.num_nodes(); ++n)    apply_repulsion(n);
        endtime = std::chrono::high_resolution_clock::now();
        runningtime = (endtime - starttime);
        runningtime_us = std::chrono::duration_cast<std::chrono::microseconds>(runningtime);
        //runningtimes[iteration][3] = runningtime_us.count();


        starttime = std::chrono::high_resolution_clock::now();
            updateSpeeds();
        endtime = std::chrono::high_resolution_clock::now();
        runningtime = (endtime - starttime);
        runningtime_us = std::chrono::duration_cast<std::chrono::microseconds>(runningtime);
        //runningtimes[iteration][4] = runningtime_us.count();

        starttime = std::chrono::high_resolution_clock::now();
        for (nid_t n = 0; n < layout.graph.num_nodes(); ++n)
        {
            apply_displacement(n);
            prev_forces[n]  = forces[n];
            forces[n]       = Real2DVector(0.0f, 0.0f);
        }
        endtime = std::chrono::high_resolution_clock::now();
        runningtime = (endtime - starttime);
        runningtime_us = std::chrono::duration_cast<std::chrono::microseconds>(runningtime);
        //runningtimes[iteration][5] = runningtime_us.count();

        iteration++;
    }
    void CPUForceAtlas2::benchmark()
    {
        printf("Benchmarks:\n");
        printf("Gravity ");
        float total = 0, grandtotal = 0;
        for (int i = 0; i < 10; ++i) total += runningtimes[i][0];
        printf("%.4f", total/10.0);
        printf("\n");

        grandtotal += total;
        total = 0;
        printf("Attractive ");
        for (int i = 0; i < 10; ++i) total += runningtimes[i][1];
        printf("%.4f", total/10.0);
        printf("\n");

        grandtotal += total;
        total = 0;
        printf("Barnes-HutTreeBuild ");
        for (int i = 0; i < 10; ++i) total += runningtimes[i][2];
        printf("%.4f", total/10.0);
        printf("\n");

        grandtotal += total;
        total = 0;
        printf("Barnes-HutApproximation ");
        for (int i = 0; i < 10; ++i) total += runningtimes[i][3];
        printf("%.4f", total/10.0);
        printf("\n");

        grandtotal += total;
        total = 0;
        printf("Speed ");
        for (int i = 0; i < 10; ++i) total += runningtimes[i][4];
        printf("%.4f", total/10.0);
        printf("\n");

        grandtotal += total;
        total = 0;
        printf("Displacement ");
        for (int i = 0; i < 10; ++i) total += runningtimes[i][5];
        printf("%.4f", total/10.0);
        printf("\n\n");
        printf("Total ");
        printf("%.4f", grandtotal/10.0);
        printf("\n");
    }
    void CPUForceAtlas2::sync_layout() {}

}
