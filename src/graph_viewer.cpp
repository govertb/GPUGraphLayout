/*
 ==============================================================================

 graph_viewer.cpp
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

-------------------------------------------------------------------------------

 This code was written as part of a research project at the Leiden Institute of
 Advanced Computer Science (www.liacs.nl). For other resources related to this
 project, see https://liacs.leidenuniv.nl/~takesfw/GPUNetworkVis/.


 ==============================================================================
*/


#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <math.h>

#include "RPCommon.hpp"
#include "RPGraph.hpp"
#include "RPGraphLayout.hpp"
#include "RPCPUForceAtlas2.hpp"

#ifdef __NVCC__
#include <cuda_runtime_api.h>
#include "RPGPUForceAtlas2.hpp"
#endif

int main(int argc, const char **argv)
{
    // For reproducibility.
    srandom(1234);

    // Parse commandline arguments
    if (argc < 10)
    {
        fprintf(stderr, "Usage: graph_viewer cuda|seq max_iterations num_snaps sg|wg scale gravity exact|approximate edgelist_path out_path [png|csv|bin]\n");
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
    std::string out_format;
    if (argc > 10) out_format = std::string(argv[10]);
    else out_format = "png";

    const int framesize = 10000;
    const float w = framesize;
    const float h = framesize;

    if(cuda_requested and not approximate)
    {
        fprintf(stderr, "error: The CUDA implementation (currently) requires Barnes-Hut approximation.\n");
        exit(EXIT_FAILURE);
    }

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

    // If not compiled with cuda support, check if cuda is requested.
    #ifndef __NVCC__
    if(cuda_requested)
    {
        fprintf(stderr, "error: CUDA was requested, but not compiled for.\n");
        exit(EXIT_FAILURE);
    }
    #endif

    // Load graph.
    printf("Loading edgelist at '%s'...", edgelist_path);
    fflush(stdout);
    RPGraph::UGraph graph = RPGraph::UGraph(edgelist_path);
    printf("done.\n");
    printf("    fetched %d nodes and %d edges.\n", graph.num_nodes(), graph.num_edges());

    // Create the GraphLayout and ForceAtlas2 objects.
    RPGraph::GraphLayout layout(graph, w, h);
    RPGraph::ForceAtlas2 *fa2;
    #ifdef __NVCC__
    if(cuda_requested)
        fa2 = new RPGraph::CUDAForceAtlas2(layout);
    else
    #endif
        fa2 = new RPGraph::CPUForceAtlas2(layout);
    
    fa2->strong_gravity = strong_gravity;
    fa2->use_barneshut = approximate;
    fa2->setScale(scale);
    fa2->setGravity(gravity);

    printf("Started Layout algorithm...\n");
    const int snap_period = ceil((float)max_iterations/num_screenshots);
    const int print_period = ceil((float)max_iterations*0.05);

    for (int iteration = 1; iteration <= max_iterations; ++iteration)
    {
        fa2->doStep();
        // If we need to, write the result to a png
        if (num_screenshots > 0 && (iteration % snap_period == 0 || iteration == max_iterations))
        {
            std::string op(out_path);
            op.append("/").append(std::to_string(iteration)).append(".").append(out_format);
            printf("Starting iteration %d (%.2f%%), writing %s...", iteration, 100*(float)iteration/max_iterations, out_format.c_str());
            fflush(stdout);
            fa2->sync_layout();

            if (out_format == "png")
                layout.writeToPNG(framesize, framesize, op.c_str());
            else if (out_format == "csv")
                layout.writeToCSV(op.c_str());
            else if (out_format == "bin")
                layout.writeToBin(op.c_str());
            
            printf("done.\n");
        }

        // Else we print (if we need to)
        else if (iteration % print_period == 0)
        {
            printf("Starting iteration %d (%.2f%%).\n", iteration, 100*(float)iteration/max_iterations);
        }
    }
    
    delete fa2;
    exit(EXIT_SUCCESS);
}
