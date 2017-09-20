/*
 ==============================================================================

 RPForceAtlas2.cpp

 This code was written as part of a research project at the Leiden Institute of
 Advanced Computer Science (www.liacs.nl). For other resources related to this
 project, see https://liacs.leidenuniv.nl/~takesfw/GPUNetworkVis/.

 Copyright © 2016, 2017  G. Brinkmann

 This program is free software: you can redistribute it and/or modify
 it under the terms of version 3 of the GNU Affero General Public License as
 published by the Free Software Foundation.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU Affero General Public License for more details.

 You should have received a copy of the GNU Affero General Public License
 along with this program.  If not, see <https://www.gnu.org/licenses/>.

 ==============================================================================
*/

#include "RPForceAtlas2.hpp"
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <chrono>

namespace RPGraph
{
    FA2Layout::FA2Layout(UGraph &graph, float width, float height)
    : GraphLayout{graph, width, height}
    {
        iteration = 0;

        forces      = (Real2DVector *)malloc(sizeof(Real2DVector) * graph.num_nodes());
        prev_forces = (Real2DVector *)malloc(sizeof(Real2DVector) * graph.num_nodes());

        k_g = 1.0;
        k_r = 1.0;

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

        for (nid_t n = 0; n < graph.num_nodes(); ++n)
        {
            forces[n]      = Real2DVector(0.0f, 0.0f);
            prev_forces[n] = Real2DVector(0.0f, 0.0f);
        }

        randomizePositions();

        if (use_barneshut)
            BH_Approximator = new BarnesHutApproximator(theta, *this);
    }

    FA2Layout::~FA2Layout()
    {
        delete BH_Approximator;
        free(forces);
        free(prev_forces);
    }

    void FA2Layout::apply_attract(nid_t n)
    {
        Real2DVector f = Real2DVector(0.0, 0.0);
        for (nid_t t : graph.neighbors_with_geq_id(n))
        {
            // Here we define the magnitude of the attractive force `f_a'
            // *divided* by the length distance between `n' and `t', i.e. `f_a_over_d'
            float f_a_over_d;
            if (use_linlog)
            {
                float dist = getDistance(n, t);
                f_a_over_d = logf(1+dist) / dist;
            }

            else
            {
                f_a_over_d = 1.0;
            }

            f += getDistanceVector(n, t) * f_a_over_d;

            //TODO: this is temporary, but required due to
            //      iteration over neighbors_with_geq_id
            forces[t] += getDistanceVector(n, t) * (-f_a_over_d);

//            forces[n] += getNormalizedDistanceVector(n, t) * f_a(n, t);
        }
        forces[n] += f;
    }

    void FA2Layout::apply_repulsion(nid_t n)
    {
        if (use_barneshut)
        {
            forces[n] += (BH_Approximator->approximateForce(getCoordinate(n), mass(n), theta) * k_r);
        }

        else
        {
            for (nid_t t = 0; t < graph.num_nodes(); ++t)
            {
                if (n == t) continue;
                float f_r;
                float  distance = getDistance(n, t);
                distance == 0.0 ? f_r = FLT_MAX : k_r * mass(n) * mass(t) / distance / distance;
                forces[n] += getDistanceVector(n, t) * f_r;
            }
        }
    }

    void FA2Layout::apply_gravity(nid_t n)
    {
        float f_g, d;

        // `d' is the distance from `n' to the center (0.0, 0.0)
        d = sqrtf(getX(n)*getX(n) + getY(n)*getY(n));
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

        forces[n] += (Real2DVector(-getX(n), -getY(n)) * f_g);
    }

    // Eq. (8)
    float FA2Layout::swg(nid_t n)
    {
        return (forces[n] - prev_forces[n]).magnitude();
    }

    // Eq. (9)
    float FA2Layout::s(nid_t n)
    {
        return (k_s * global_speed)/(1.0f+global_speed*sqrtf(swg(n)));
    }

    // Eq. (12)
    float FA2Layout::tra(nid_t n)
    {
        return (forces[n] + prev_forces[n]).magnitude() / 2.0;
    }

    float FA2Layout::mass(nid_t n)
    {
        return graph.degree(n) + 1.0;
    }

    void FA2Layout::updateSpeeds()
    {
        // The following speed-update procedure for ForceAtlas2 follows
        // the one by Gephi:
        // https://github.com/gephi/gephi/blob/6efb108718fa67d1055160f3a18b63edb4ca7be2/modules/LayoutPlugin/src/main/java/org/gephi/layout/plugin/forceAtlas2/ForceAtlas2.java

        // `Auto adjust speeds'
        float total_swinging = 0.0;
        float total_effective_traction = 0.0;
        for (nid_t nid = 0; nid < graph.num_nodes(); ++nid)
        {
            total_swinging += mass(nid) * swg(nid); // Eq. (11)
            total_effective_traction += mass(nid) * tra(nid); // Eq. (13)
        }

        // We want to find the right jitter tollerance for this graph,
        // such that totalSwinging < tolerance * totalEffectiveTraction

        float estimated_optimal_jitter_tollerance = 0.05 * sqrtf(graph.num_nodes());
        float minJT = sqrtf(estimated_optimal_jitter_tollerance);
        float jt = jitter_tolerance * fmaxf(minJT,
                                           fminf(k_s_max,
                                                 estimated_optimal_jitter_tollerance * total_effective_traction / powf(graph.num_nodes(), 2.0)
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

    void FA2Layout::apply_displacement(nid_t n)
    {
        if (prevent_overlap)
        {
            // Not yet implemented
            exit(EXIT_FAILURE);
        }

        else
        {

            float factor = global_speed / (1.0 + sqrtf(global_speed * swg(n)));
            moveNode(n, forces[n] * factor);

        }
    }

    void FA2Layout::doStep()
    {
        auto starttime = std::chrono::high_resolution_clock::now();
        auto endtime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> runningtime;
        std::chrono::microseconds runningtime_us;

        starttime = std::chrono::high_resolution_clock::now();
        if (use_barneshut)
            BH_Approximator->rebuild();
        endtime = std::chrono::high_resolution_clock::now();
        runningtime = (endtime - starttime);
        runningtime_us = std::chrono::duration_cast<std::chrono::microseconds>(runningtime);
        //runningtimes[iteration][2] = runningtime_us.count();

        starttime = std::chrono::high_resolution_clock::now();
            for (nid_t n = 0; n < graph.num_nodes(); ++n) apply_gravity(n);
        endtime = std::chrono::high_resolution_clock::now();
        runningtime = (endtime - starttime);
        runningtime_us = std::chrono::duration_cast<std::chrono::microseconds>(runningtime);
        //runningtimes[iteration][0] = runningtime_us.count();

        starttime = std::chrono::high_resolution_clock::now();
            for (nid_t n = 0; n < graph.num_nodes(); ++n)    apply_attract(n);
        endtime = std::chrono::high_resolution_clock::now();
        runningtime = (endtime - starttime);
        runningtime_us = std::chrono::duration_cast<std::chrono::microseconds>(runningtime);
        //runningtimes[iteration][1] = runningtime_us.count();

        starttime = std::chrono::high_resolution_clock::now();
            for (nid_t n = 0; n < graph.num_nodes(); ++n)    apply_repulsion(n);
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
        for (nid_t n = 0; n < graph.num_nodes(); ++n)
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
    void FA2Layout::print_benchmarks()
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

    void FA2Layout::doSteps(int n)
    {
        for (int i = 0; i < n; ++i) doStep();
    }

    void FA2Layout::setScale(float s)
    {
        k_r = s;
    }

    void FA2Layout::setGravity(float g)
    {
        k_g = g;
    }
}
