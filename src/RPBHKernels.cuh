/*
 The following code is a modified version of the CUDA BarnesHut v3.1 code
 by Martin Burtscher. Modifications were made to transform the code from a
 three-dimensional Barnes-Hut implementation to a two-dimensional implementation.
 Since our application (graph layout) only needs two dimensions.

 What follows is the copyright notice associated with that
 original code, as it is provided by the copyright holder:
 Texas State University-San Macros.
*/


/*
 CUDA BarnesHut v3.1: Simulation of the gravitational forces
 in a galactic cluster using the Barnes-Hut n-body algorithm

 Copyright (c) 2013, Texas State University-San Marcos. All rights reserved.

 Redistribution and use in source and binary forms, with or without modification,
 are permitted for academic, research, experimental, or personal use provided that
 the following conditions are met:

 * Redistributions of source code must retain the above copyright notice,
 this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation
 and/or other materials provided with the distribution.
 * Neither the name of Texas State University-San Marcos nor the names of its
 contributors may be used to endorse or promote products derived from this
 software without specific prior written permission.

 For all other uses, please contact the Office for Commercialization and Industry
 Relations at Texas State University-San Marcos <http://www.txstate.edu/ocir/>.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED
 IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 OF THE POSSIBILITY OF SUCH DAMAGE.

 Author: Martin Burtscher <burtscher@txstate.edu>
 */

#ifndef RPBHKernels_cuh
#define RPBHKernels_cuh

#include "RPBHFA2LaunchParameters.cuh"

extern __device__ volatile int errd;
extern __device__ float minxdg, minydg, maxxdg, maxydg;

__global__
__launch_bounds__(THREADS1, FACTOR1)
void BoundingBoxKernel(int nnodesd, int nbodiesd, volatile int * __restrict startd,
                       volatile int   * __restrict childd, volatile float * __restrict massd,
                       volatile float2 * __restrict body_posd, volatile float2 * __restrict node_posd,
                       volatile float * __restrict maxxd,  volatile float * __restrict maxyd,
                       volatile float * __restrict minxd,  volatile float * __restrict minyd);

__global__
__launch_bounds__(1024, 1)
void ClearKernel1(int nnodesd, int nbodiesd, volatile int * __restrict childd);

__global__
__launch_bounds__(THREADS2, FACTOR2)
void TreeBuildingKernel(int nnodesd, int nbodiesd, volatile int * __restrict childd,
                        volatile float2 * __restrict body_posd, volatile float2 * __restrict node_posd);

__global__
__launch_bounds__(1024, 1)
void ClearKernel2(int nnodesd, volatile int * __restrict startd, volatile float * __restrict massd);

__global__
__launch_bounds__(THREADS3, FACTOR3)
void SummarizationKernel(const int nnodesd, const int nbodiesd, volatile int * __restrict countd, const int * __restrict childd,
                         volatile float * __restrict massd, volatile float2 * __restrict body_posd, volatile float2 * __restrict node_posd);

__global__
__launch_bounds__(THREADS4, FACTOR4)
void SortKernel(int nnodesd, int nbodiesd, int * __restrict sortd, int * __restrict countd, volatile int * __restrict startd, int * __restrict childd);

__global__
__launch_bounds__(THREADS5, FACTOR5)
void ForceCalculationKernel(int nnodesd, int nbodiesd, float itolsqd, float epssqd,
                            volatile int * __restrict sortd, volatile int * __restrict childd, volatile float * __restrict massd,
                            volatile float2 * __restrict body_posd, volatile float2 * __restrict node_posd,
                            volatile float * __restrict fxd, volatile float * __restrict fyd, const float k_rd);

#endif
