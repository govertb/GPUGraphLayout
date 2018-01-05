/*
 ==============================================================================

 RPGPUForceAtlas2.cu
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

#include <stdio.h>
#include <fstream>
#include <chrono>
#include <algorithm>
#include "time.h"

#include "RPGPUForceAtlas2.hpp"
#include "RPBHFA2LaunchParameters.cuh"
#include "RPBHKernels.cuh"
#include "RPFA2Kernels.cuh"

namespace RPGraph
{
    CUDAForceAtlas2::CUDAForceAtlas2(GraphLayout &layout)
    : ForceAtlas2(layout)
    {
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        if (deviceCount == 0)
        {
            fprintf(stderr, "error: No CUDA devices found.\n");
            exit(EXIT_FAILURE);
        }

        // Host initialization and setup //
        nbodies = layout.graph.num_nodes();
        nedges  = layout.graph.num_edges();

        body_pos = (float2 *)malloc(sizeof(float2) * layout.graph.num_nodes());
        body_mass = (float *)malloc(sizeof(float) * layout.graph.num_nodes());
        sources  = (int *)  malloc(sizeof(int)   * layout.graph.num_edges());
        targets  = (int *)  malloc(sizeof(int)   * layout.graph.num_edges());
        fx       = (float *)malloc(sizeof(float) * layout.graph.num_nodes());
        fy       = (float *)malloc(sizeof(float) * layout.graph.num_nodes());
        fx_prev  = (float *)malloc(sizeof(float) * layout.graph.num_nodes());
        fy_prev  = (float *)malloc(sizeof(float) * layout.graph.num_nodes());

        for (nid_t n = 0; n < layout.graph.num_nodes(); ++n)
        {
            body_pos[n] = {layout.getX(n), layout.getY(n)};
            body_mass[n] = ForceAtlas2::mass(n);
            fx[n] = 0.0;
            fy[n] = 0.0;
            fx_prev[n] = 0.0;
            fy_prev[n] = 0.0;
        }

        int cur_sources_idx = 0;
        int cur_targets_idx = 0;

        // Initialize the sources and targets arrays with edge-data.
        for (nid_t source_id = 0; source_id < layout.graph.num_nodes(); ++source_id)
        {
            for (nid_t target_id : layout.graph.neighbors_with_geq_id(source_id))
            {
                sources[cur_sources_idx++] = source_id;
                targets[cur_targets_idx++] = target_id;
            }
        }

        // GPU initialization and setup //
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);

        if (deviceProp.warpSize != WARPSIZE)
        {
            printf("Warpsize of device is %d, but we anticipated %d\n", deviceProp.warpSize, WARPSIZE);
            exit(EXIT_FAILURE);

        }
        cudaFuncSetCacheConfig(BoundingBoxKernel, cudaFuncCachePreferShared);
        cudaFuncSetCacheConfig(TreeBuildingKernel, cudaFuncCachePreferL1);
        cudaFuncSetCacheConfig(ClearKernel1, cudaFuncCachePreferL1);
        cudaFuncSetCacheConfig(ClearKernel2, cudaFuncCachePreferL1);
        cudaFuncSetCacheConfig(SummarizationKernel, cudaFuncCachePreferShared);
        cudaFuncSetCacheConfig(SortKernel, cudaFuncCachePreferL1);
#if __CUDA_ARCH__ < 300
        cudaFuncSetCacheConfig(ForceCalculationKernel, cudaFuncCachePreferL1);
#endif
        cudaFuncSetCacheConfig(DisplacementKernel, cudaFuncCachePreferL1);

        cudaGetLastError();  // reset error value

        // Allocate space on device.
        mp_count = deviceProp.multiProcessorCount;
        max_threads_per_block = deviceProp.maxThreadsPerBlock;

        nnodes = std::max(2 * nbodies, mp_count * max_threads_per_block);

        // Round up to next multiple of WARPSIZE
        while ((nnodes & (WARPSIZE-1)) != 0) nnodes++;
        nnodes--;

        // child stores structure of the quadtree. values point to IDs.
        cudaCatchError(cudaMalloc((void **)&childl,  sizeof(int)   * (nnodes+1) * 4));

        // the following properties, for each node in the quadtree (both internal and leaf)
        cudaCatchError(cudaMalloc((void **)&body_massl,   sizeof(float) * nbodies));
        cudaCatchError(cudaMalloc((void **)&node_massl,   sizeof(float) * (nnodes+1)));
        cudaCatchError(cudaMalloc((void **)&body_posl,sizeof(float2) * nbodies));
        cudaCatchError(cudaMalloc((void **)&node_posl,    sizeof(float2) * (nnodes+1)));
        // count contains the number of nested nodes for each node in quadtree
        cudaCatchError(cudaMalloc((void **)&countl,  sizeof(int)   * (nnodes+1)));
        // start contains ...
        cudaCatchError(cudaMalloc((void **)&startl,  sizeof(int)   * (nnodes+1)));
        cudaCatchError(cudaMalloc((void **)&sortl,   sizeof(int)   * (nnodes+1)));


        cudaCatchError(cudaMalloc((void **)&sourcesl,sizeof(int)   * (nedges)));
        cudaCatchError(cudaMalloc((void **)&targetsl,sizeof(int)   * (nedges)));
        cudaCatchError(cudaMalloc((void **)&fxl,     sizeof(float) * (nbodies)));
        cudaCatchError(cudaMalloc((void **)&fyl,     sizeof(float) * (nbodies)));
        cudaCatchError(cudaMalloc((void **)&fx_prevl,sizeof(float) * (nbodies)));
        cudaCatchError(cudaMalloc((void **)&fy_prevl,sizeof(float) * (nbodies)));

        // Used for reduction in BoundingBoxKernel
        cudaCatchError(cudaMalloc((void **)&maxxl,   sizeof(float) * mp_count * FACTOR1));
        cudaCatchError(cudaMalloc((void **)&maxyl,   sizeof(float) * mp_count * FACTOR1));
        cudaCatchError(cudaMalloc((void **)&minxl,   sizeof(float) * mp_count * FACTOR1));
        cudaCatchError(cudaMalloc((void **)&minyl,   sizeof(float) * mp_count * FACTOR1));

        // Used for reduction in SpeedKernel
        cudaCatchError(cudaMalloc((void **)&swgl,    sizeof(float) * mp_count * FACTOR6));
        cudaCatchError(cudaMalloc((void **)&etral,   sizeof(float) * mp_count * FACTOR6));

        // Copy host data to device.
        cudaCatchError(cudaMemcpy(body_massl, body_mass, sizeof(float) * nbodies, cudaMemcpyHostToDevice));
        cudaCatchError(cudaMemcpy(body_posl,  body_pos,  sizeof(float2) * nbodies, cudaMemcpyHostToDevice));
        cudaCatchError(cudaMemcpy(sourcesl, sources, sizeof(int) * nedges, cudaMemcpyHostToDevice));
        cudaCatchError(cudaMemcpy(targetsl, targets, sizeof(int) * nedges, cudaMemcpyHostToDevice));

        // cpy fx, fy , fx_prevl, fy_prevl so they are all initialized to 0 in device memory.
        cudaCatchError(cudaMemcpy(fxl, fx,           sizeof(float) * nbodies, cudaMemcpyHostToDevice));
        cudaCatchError(cudaMemcpy(fyl, fy,           sizeof(float) * nbodies, cudaMemcpyHostToDevice));
        cudaCatchError(cudaMemcpy(fx_prevl, fx_prev, sizeof(float) * nbodies, cudaMemcpyHostToDevice));
        cudaCatchError(cudaMemcpy(fy_prevl, fy_prev, sizeof(float) * nbodies, cudaMemcpyHostToDevice));
    }

    void CUDAForceAtlas2::freeGPUMemory()
    {
        cudaFree(childl);

        cudaFree(body_massl);
        cudaFree(node_massl);
        cudaFree(body_posl);
        cudaFree(node_posl);
        cudaFree(sourcesl);
        cudaFree(targetsl);
        cudaFree(countl);
        cudaFree(startl);
        cudaFree(sortl);

        cudaFree(fxl);
        cudaFree(fx_prevl);
        cudaFree(fyl);
        cudaFree(fy_prevl);

        cudaFree(maxxl);
        cudaFree(maxyl);
        cudaFree(minxl);
        cudaFree(minyl);

        cudaFree(swgl);
        cudaFree(etral);
    }

    CUDAForceAtlas2::~CUDAForceAtlas2()
    {
        free(body_mass);
        free(body_pos);
        free(sources);
        free(targets);
        free(fx);
        free(fy);
        free(fx_prev);
        free(fy_prev);

        freeGPUMemory();
    }

    void CUDAForceAtlas2::benchmark()
    {
        printf("Using %d MPs\n", mp_count);
        const int num_reps = 5;

        float times[11] = {0.0, };
        float time; // to temporarily hold time of a kernel

        const char *kernel_names[11] = {"Gravity", "Attractive", "BoundingBox", "Clear1", "TreeBuilding", "Clear2", "Summarization", "Sort", "Force", "Speed", "Displacement"};

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaDeviceSynchronize();
        auto starttime = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < num_reps; ++i)
        {
            cudaEventRecord(start, 0);
            GravityKernel<<<mp_count * FACTOR6, THREADS6>>>(nbodies, k_g, strong_gravity, body_massl, body_posl, fxl, fyl);
            cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&time, start, stop);
            times[0] += time;
            cudaCatchError(cudaGetLastError());

            cudaEventRecord(start, 0);
            AttractiveForceKernel<<<mp_count * FACTOR6, THREADS6>>>(nedges, body_posl, fxl, fyl, sourcesl, targetsl);
            cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&time, start, stop);
            times[1] += time;
            cudaCatchError(cudaGetLastError());

            cudaEventRecord(start, 0);
            BoundingBoxKernel<<<mp_count * FACTOR1, THREADS1>>>(nnodes, nbodies, startl, childl, node_massl, body_posl, node_posl, maxxl, maxyl, minxl, minyl);
            cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&time, start, stop);
            times[2] += time;
            cudaCatchError(cudaGetLastError());

            cudaEventRecord(start, 0);
            ClearKernel1<<<1024, 1>>>(nnodes, nbodies, childl);
            cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&time, start, stop);
            times[3] += time;
            cudaCatchError(cudaGetLastError());

            cudaEventRecord(start, 0);
            TreeBuildingKernel<<<mp_count * FACTOR2, THREADS2>>>(nnodes, nbodies, childl, body_posl, node_posl);
            cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&time, start, stop);
            times[4] += time;
            cudaCatchError(cudaGetLastError());

            cudaEventRecord(start, 0);
            ClearKernel2<<<1024, 1>>>(nnodes, startl, node_massl);
            cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&time, start, stop);
            times[5] += time;
            cudaCatchError(cudaGetLastError());

            cudaEventRecord(start, 0);
            SummarizationKernel<<<mp_count * FACTOR3, THREADS3>>>(nnodes, nbodies, countl, childl, body_massl, node_massl, body_posl, node_posl);
            cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&time, start, stop);
            times[6] += time;
            cudaCatchError(cudaGetLastError());

            cudaEventRecord(start, 0);
            SortKernel<<<mp_count * FACTOR4, THREADS4>>>(nnodes, nbodies, sortl, countl, startl, childl);
            cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&time, start, stop);
            times[7] += time;
            cudaCatchError(cudaGetLastError());

            float epssq  = 0.05 * 0.05;            // Some sort of softening (eps, squared)
            float itolsq = 1.0f / (theta * theta); // Inverse tolerance, squared
            cudaEventRecord(start, 0);
            ForceCalculationKernel<<<mp_count * FACTOR5, THREADS5>>>(nnodes, nbodies, itolsq, epssq, sortl, childl, body_massl, node_massl, body_posl, node_posl, fxl, fyl, k_r);
            cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&time, start, stop);
            times[8] += time;
            cudaCatchError(cudaGetLastError());

            cudaEventRecord(start, 0);
            SpeedKernel<<<mp_count * FACTOR1, THREADS1>>>(nbodies, fxl, fyl, fx_prevl, fy_prevl, body_massl, swgl, etral);
            cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&time, start, stop);
            times[9] += time;
            cudaCatchError(cudaGetLastError());

            cudaEventRecord(start, 0);
            DisplacementKernel<<<mp_count * FACTOR6, THREADS6>>>(nbodies, body_posl, fxl, fyl, fx_prevl, fy_prevl);
            cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&time, start, stop);
            times[10] += time;
            cudaCatchError(cudaGetLastError());
        }

        cudaDeviceSynchronize(); // Not really neccesary given the preceding cudaEventSynchronize(stop).
        auto endtime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> runningtime = (endtime - starttime);
        std::chrono::microseconds runningtime_us = std::chrono::duration_cast<std::chrono::microseconds>(runningtime);

        printf("Benchmarking Results (averaging %d times):\n", num_reps);
        printf("\tkernel durations (us), resolution: ... us.:\n");
        for (int i = 0; i < 11; ++i)
        {
            printf("%s %.4f\n", kernel_names[i], 1000.0 * times[i] / (float)num_reps);
        }
        printf("\n");
        printf("Total %.2f\n", runningtime_us.count() / (float)num_reps);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void CUDAForceAtlas2::doStep()
    {
        int err = 0;

        GravityKernel<<<mp_count * FACTOR6, THREADS6>>>(nbodies, k_g, strong_gravity, body_massl, body_posl, fxl, fyl);

        AttractiveForceKernel<<<mp_count * FACTOR6, THREADS6>>>(nedges, body_posl, fxl, fyl, sourcesl, targetsl);

        BoundingBoxKernel<<<mp_count * FACTOR1, THREADS1>>>(nnodes, nbodies, startl, childl, node_massl, body_posl, node_posl, maxxl, maxyl, minxl, minyl);

        // Build Barnes-Hut Tree
        // 1.) Set all child pointers of internal nodes (in childl) to null (-1)
        ClearKernel1<<<mp_count, 1024>>>(nnodes, nbodies, childl);
        // 2.) Build the tree
        TreeBuildingKernel<<<mp_count * FACTOR2, THREADS2>>>(nnodes, nbodies, childl, body_posl, node_posl);

        cudaDeviceSynchronize();
        cudaCatchError(cudaMemcpyFromSymbol(&err, errd, sizeof(int), 0, cudaMemcpyDeviceToHost));
        if (err != 0)
        {
            fprintf(stderr, "error: An error occurred in TreeBuildingKernel, errd == %d\n", err);
            // exit(EXIT_FAILURE);
        }

        // 3.) Set all cell mass values to -1.0, set all startd to null (-1)
        ClearKernel2<<<mp_count, 1024>>>(nnodes, startl, node_massl);

        // Recursively compute mass for each BH. cell.
        SummarizationKernel<<<mp_count * FACTOR3, THREADS3>>>(nnodes, nbodies, countl, childl, body_massl, node_massl, body_posl, node_posl);

        SortKernel<<<mp_count * FACTOR4, THREADS4>>>(nnodes, nbodies, sortl, countl, startl, childl);

        // Compute repulsive forces between nodes using BH. tree.
        float epssq  = 0.05 * 0.05;            // Some sort of softening (eps, squared)
        float itolsq = 1.0f / (theta * theta); // Inverse tolerance, squared
        ForceCalculationKernel<<<mp_count * FACTOR5, THREADS5>>>(nnodes, nbodies, itolsq, epssq, sortl, childl, body_massl, node_massl, body_posl, node_posl, fxl, fyl, k_r);

        cudaDeviceSynchronize();
        cudaCatchError(cudaMemcpyFromSymbol(&err, errd, sizeof(int), 0, cudaMemcpyDeviceToHost));
        if (err != 0)
        {
            fprintf(stderr, "error: An error occurred in ForceCalculationKernel, errd == %d\n", err);
            exit(EXIT_FAILURE);
        }

        SpeedKernel<<<mp_count * FACTOR1, THREADS1>>>(nbodies, fxl, fyl, fx_prevl, fy_prevl, body_massl, swgl, etral);

        DisplacementKernel<<<mp_count * FACTOR6, THREADS6>>>(nbodies, body_posl, fxl, fyl, fx_prevl, fy_prevl);

        iteration++;
    }

    void CUDAForceAtlas2::retrieveLayoutFromGPU()
    {
        cudaCatchError(cudaMemcpy(body_pos, body_posl, sizeof(float2) * nbodies, cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
    }

    void CUDAForceAtlas2::sendLayoutToGPU()
    {
        cudaCatchError(cudaMemcpy(body_posl, body_pos, sizeof(float2) * nbodies, cudaMemcpyHostToDevice));
        cudaDeviceSynchronize();
    }

    void CUDAForceAtlas2::sendGraphToGPU()
    {
        cudaCatchError(cudaMemcpy(body_massl, body_mass, sizeof(float) * nbodies, cudaMemcpyHostToDevice));
        cudaCatchError(cudaMemcpy(sourcesl, sources, sizeof(int) * nedges, cudaMemcpyHostToDevice));
        cudaCatchError(cudaMemcpy(targetsl, targets, sizeof(int) * nedges, cudaMemcpyHostToDevice));
        cudaDeviceSynchronize();
    }
    
    void CUDAForceAtlas2::sync_layout() 
    {
        retrieveLayoutFromGPU();
        for(nid_t n = 0; n < layout.graph.num_nodes(); ++n)
        {
            layout.setX(n, body_pos[n].x);
            layout.setY(n, body_pos[n].y);
        }
    }
}
