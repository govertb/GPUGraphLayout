/*
 ==============================================================================

 main.cpp

 This code was written as part of a research project at the Leiden Institute of
 Advanced Computer Science (www.liacs.nl). For other resources related to this
 project, see https://liacs.leidenuniv.nl/~takesfw/GPUNetworkVis/.

 Copyright (C) 2016, 2017  G. Brinkmann

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


#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <math.h>

#include "RPCommon.hpp"
#include "RPGraph.hpp"
#include "RPGraphLayout.hpp"
#include "RPForceAtlas2.hpp"

#ifdef __CUDA__
#include <cuda_runtime_api.h>
#include "RPCUDAForceAtlas2.hpp"
#endif

int main(int argc, const char **argv)
{
    // For reproducibility.
    srandom(1234);

    // Parse commandline arguments
    if (argc != 11)
    {
        fprintf(stderr, "Usage: graph_viewer cuda|seq max_iterations num_snaps sg|wg scale gravity exact|approximate edgelist_path out_path test|image\n");
        exit(EXIT_FAILURE);
    }

    const bool cuda_requested = std::string(argv[1]) == "cuda";
    const int max_iterations = std::stoi(argv[2]);
    const int num_screenshots = std::stoi(argv[3]);
    const bool strong_gravity = std::string(argv[4]) == "sg";
    const float scale = std::stof(argv[5]);
    const float gravity = std::stof(argv[6]);
    const bool approximate = std::string(argv[7]) == "approximate";
    const char *edgelist_path = argv[8];
    const char *out_path = argv[9];
    const bool testmode = std::string(argv[10]) == "test";
    const int framesize = 10000;
    const float w = framesize;
    const float h = framesize;

    // Check in_path and out_path
    if (!is_file_exists(edgelist_path))
    {
        fprintf(stderr, "error: No edgelist at %s\n", edgelist_path);
        exit(EXIT_FAILURE);
    }
    if (!is_file_exists(out_path))
    {
        fprintf(stderr, "error: No output folder at %s\n", out_path);
        exit(EXIT_FAILURE);
    }

    printf("Loading edgelist at '%s'...", edgelist_path);
    fflush(stdout);
    RPGraph::UGraph graph = RPGraph::UGraph(edgelist_path);
    printf("done.\n");
    printf("    fetched %d nodes and %d edges.\n", graph.num_nodes(), graph.num_edges());

    if(cuda_requested)
    {
#ifdef __CUDA__
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        if (deviceCount == 0)
        {
            fprintf(stderr, "error: There is no device supporting CUDA...\n");
            exit(EXIT_FAILURE);
        }

        RPGraph::CUDAFA2Layout layout(graph, w, h);
        layout.strong_gravity = strong_gravity;
        layout.use_barneshut = approximate;
        layout.setScale(scale);
        layout.setGravity(gravity);

        if(testmode)
        {
	        layout.benchmark();
	        exit(EXIT_SUCCESS);
	    }

        printf("Started Layout algorithm...\n");
        const int snap_period = ceil((float)max_iterations/num_screenshots);
        const int print_period = ceil((float)max_iterations*0.05);

        for (int iteration = 1; iteration <= max_iterations; ++iteration)
        {
            layout.doStep();
            // If we need to, write the result to a png
            if (num_screenshots > 0 && (iteration % snap_period == 0 || iteration == max_iterations))
            {
                std::string op(out_path);
                op.append("/").append(std::to_string(iteration)).append(".png");
                printf("Starting iteration %d (%.2f%%), writing png...", iteration, 100*(float)iteration/max_iterations);
                fflush(stdout);
                layout.writeToPNG(framesize, framesize, op.c_str());
                printf("done.\n");
            }

            // Else we print (if we need to)
            else if (iteration % print_period == 0)
            {
                printf("Starting iteration %d (%.2f%%).\n", iteration, 100*(float)iteration/max_iterations);
            }
        }
#else
        fprintf(stderr, "error: CUDA was requested, but not compiled for.\n");
        exit(EXIT_FAILURE);
#endif
    }

    else if(!cuda_requested)
    {
        // Create the layout
        RPGraph::FA2Layout layout = RPGraph::FA2Layout(graph, w, h);
        layout.strong_gravity = strong_gravity;
        layout.use_barneshut = approximate;
        layout.setScale(scale);
        layout.setGravity(gravity);

        if(testmode)
        {
            layout.doSteps(100);
            layout.print_benchmarks();
            exit(EXIT_SUCCESS);
        }

        printf("Started Layout algorithm...\n");
        const int snap_period = ceil((float)max_iterations/num_screenshots);
        const int print_period = ceil((float)max_iterations*0.05);
        for (int iteration = 1; iteration <= max_iterations; ++iteration)
        {
            layout.doStep();
            // If we need to, write the result to a png
            if (num_screenshots > 0 && (iteration % snap_period == 0 || iteration == max_iterations))
            {
                std::string op(out_path);
                op.append("/").append(std::to_string(iteration)).append(".png");
                printf("At iteration %d (%.2f%%), writing png...", iteration, 100*(float)iteration/max_iterations);
                fflush(stdout);
                layout.writeToPNG(framesize, framesize, op.c_str());
                printf("done.\n");
            }

            // Else we print (if we need to)
            else if (iteration % print_period == 0)
            {
                printf("At iteration %d (%.2f%%).\n", iteration, 100*(float)iteration/max_iterations);
            }
        }
    }
    exit(EXIT_SUCCESS);
}
