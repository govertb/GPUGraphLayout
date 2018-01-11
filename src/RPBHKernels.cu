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

#include <stdio.h>
#include "RPBHKernels.cuh"

// Variables marked extern in header.
__device__ volatile int errd = 0;
__device__ float minxdg, minydg, maxxdg, maxydg;


// Variables for use in this file only.
static __device__ volatile int stepd = -1;
static __device__ volatile int maxdepthd = 1;
static __device__ volatile int bottomd; // initialized by BoundingBoxKernel
static __device__ unsigned int blkcntd = 0;
static __device__ volatile float radiusd;


/*** The Kernel Definitions ***/
/******************************************************************************/
/*** compute center and radius ************************************************/
/******************************************************************************/

__global__
__launch_bounds__(THREADS1, FACTOR1)
void BoundingBoxKernel(int nnodesd, int nbodiesd, volatile int * __restrict startd,
                       volatile int   * __restrict childd, volatile float * __restrict node_massd,
                       volatile float2 * __restrict body_posd, volatile float2 * __restrict node_posd,
                       volatile float * __restrict maxxd,  volatile float * __restrict maxyd,
                       volatile float * __restrict minxd,  volatile float * __restrict minyd)
{
    register int i, j, k, inc;
    register float val, minx, maxx, miny, maxy;
    __shared__ volatile float sminx[THREADS1], smaxx[THREADS1], sminy[THREADS1], smaxy[THREADS1];

    // initialize with valid data (in case #bodies < #threads)
    minx = maxx = body_posd[0].x;
    miny = maxy = body_posd[0].y;

    // scan all bodies
    i = threadIdx.x;
    inc = THREADS1 * gridDim.x;
    for (j = i + blockIdx.x * THREADS1; j < nbodiesd; j += inc)
    {
        val = body_posd[j].x;
        minx = fminf(minx, val);
        maxx = fmaxf(maxx, val);
        val = body_posd[j].y;
        miny = fminf(miny, val);
        maxy = fmaxf(maxy, val);
    }

    // reduction in shared memory
    sminx[i] = minx;
    smaxx[i] = maxx;
    sminy[i] = miny;
    smaxy[i] = maxy;

    for (j = THREADS1 / 2; j > 0; j /= 2)
    {
        __syncthreads();
        if (i < j)
        {
            k = i + j;
            sminx[i] = minx = fminf(minx, sminx[k]);
            smaxx[i] = maxx = fmaxf(maxx, smaxx[k]);
            sminy[i] = miny = fminf(miny, sminy[k]);
            smaxy[i] = maxy = fmaxf(maxy, smaxy[k]);
        }
    }

    // write block result to global memory
    if (i == 0)
    {
        k = blockIdx.x;
        minxd[k] = minx;
        maxxd[k] = maxx;
        minyd[k] = miny;
        maxyd[k] = maxy;
        __threadfence();

        inc = gridDim.x - 1;
        if (inc == atomicInc(&blkcntd, inc))
        {
            // I'm the last block, so combine all block results
            for (j = 0; j <= inc; j++)
            {
                minx = fminf(minx, minxd[j]);
                maxx = fmaxf(maxx, maxxd[j]);
                miny = fminf(miny, minyd[j]);
                maxy = fmaxf(maxy, maxyd[j]);
            }
            // compute 'radius'
            radiusd = fmaxf(maxx - minx, maxy - miny) * 0.5f;

            // insert the root node into the BH tree.
            k = nnodesd;
            bottomd = k;

            node_massd[k] = -1.0f;
            node_posd[k].x = (minx + maxx) * 0.5f;
            node_posd[k].y = (miny + maxy) * 0.5f;
            startd[k] = 0;

            k *= 4; // skip over the children of all nodes
            for (i = 0; i < 4; i++) childd[k + i] = -1;

            stepd++;
        }
    }
}

/******************************************************************************/
/*** build tree ***************************************************************/
/******************************************************************************/

// Sets all child pointers of internal nodes in BH tree to null (-1) in childd
__global__
__launch_bounds__(1024, 1)void ClearKernel1(int nnodesd, int nbodiesd, volatile int * __restrict childd)
{
    register int k, inc, top, bottom;

    top = 4 * nnodesd; // children of root node initialized before.
    bottom = 4 * nbodiesd;
    inc = blockDim.x * gridDim.x;
    k = (bottom & (-WARPSIZE)) + threadIdx.x + blockIdx.x * blockDim.x;
    if (k < bottom) k += inc;

    // iterate over all cells assigned to thread
    while (k < top)
    {
        childd[k] = -1;
        k += inc;
    }
}


__global__
__launch_bounds__(THREADS2, FACTOR2)
void TreeBuildingKernel(int nnodesd, int nbodiesd, volatile int * __restrict childd,
                        volatile float2 * __restrict body_posd, volatile float2 * __restrict node_posd)
{
    register int i, j, depth, localmaxdepth, skip, inc;
    register float x, y, r;
    register float px, py;
    register float dx, dy;
    register int ch, n, cell, locked, patch;
    register float rootr, rootx, rooty;

    // cache root data
    rootx = node_posd[nnodesd].x;
    rooty = node_posd[nnodesd].y;
    rootr = radiusd;

    localmaxdepth = 1;
    skip = 1;
    inc = blockDim.x * gridDim.x;
    i = threadIdx.x + blockIdx.x * blockDim.x;

    // iterate over all bodies assigned to thread
    while (i < nbodiesd)
    {
        if (skip != 0)
        {
            // new body, so start traversing at root
            skip = 0;
            px = body_posd[i].x;
            py = body_posd[i].y;
            n = nnodesd;
            depth = 1;
            r = rootr * 0.5f;
            dx = dy = -r;
            j = 0;
            // determine which child to follow,
            if (rootx < px) {j  = 1; dx = r;}
            if (rooty < py) {j |= 2; dy = r;}
            x = rootx + dx;
            y = rooty + dy;
        }

        // follow path to leaf cell
        ch = childd[n*4+j];

        while (ch >= nbodiesd)
        {
            n = ch;
            depth++;
            r *= 0.5f;
            dx = dy = -r;
            j = 0;
            // determine which child to follow
            if (x < px) {j  = 1; dx = r;}
            if (y < py) {j |= 2; dy = r;}
            x += dx;
            y += dy;
            ch = childd[n*4+j];
        }

        // here ch is either leaf (< nbodiesd), null (-1), locked (-2)

        if (ch != -2)
        {
        // here we insert body into either empty cell, or split leafcell.
            // skip if child pointer is locked and try again later
            locked = n*4+j;
            if (ch == -1)
            {
                if (-1 == atomicCAS((int *)&childd[locked], -1, i))
                {  // if null, just insert the new body
                    localmaxdepth = max(depth, localmaxdepth);
                    i += inc;  // move on to next body
                    skip = 1;
                }
                // else: failed to claim cell, re-traverse next iteration.
            }
            else
            {  // there already is a body in this position
                if (ch == atomicCAS((int *)&childd[locked], ch, -2))
                {
                    // lock is now aquired on childd[locked].
                    // ch is old BH node id living at childd[locked]

                    // if bodies have same position, offset the body to insert
                    // and redo traversal
                    if (body_posd[ch].x == px && body_posd[ch].y == py)
                    {
                        body_posd[i].x *= .99;
                        body_posd[i].y *= .99;
                        skip = 0; // start all over
                        childd[locked] = ch; // release lock
                        break;
                    }

                    patch = -1;
                    // create new cell(s) and insert the new and old body
                    do
                    {
                        // 1.) Create new cell
                        cell = atomicSub((int *)&bottomd, 1) - 1;
                        if (cell <= nbodiesd)
                        {
                            errd = 1;
                            printf("Error in TreekBuildingKernel: cell <= nbodiesd\n");
                            bottomd = nnodesd;
                        }

                        if (patch != -1) childd[n*4+j] = cell;
                        patch = max(patch, cell);

                        // 2.) Make newly created cell current
                        depth++;
                        n = cell;
                        r *= 0.5f;

                        // 3.) Insert old body into correct quadrant
                        j = 0;
                        if (x < body_posd[ch].x) j  = 1;
                        if (y < body_posd[ch].y) j |= 2;
                        childd[cell*4+j] = ch;

                        // 4.) Determine center + quadrant for cell of new body
                        j = 0;
                        dx = dy = -r;
                        if (x < px) {j  = 1; dx = r;}
                        if (y < py) {j |= 2; dy = r;}
                        x += dx;
                        y += dy;

                        // 5.) Visit this cell/check if in use (possibly by old body)
                        ch = childd[n*4+j];
                        // repeat until the two bodies are different children
                    } while (ch >= 0);
                    childd[n*4+j] = i; // insert new body

                    localmaxdepth = max(depth, localmaxdepth);
                    i += inc;  // move on to next body
                    skip = 2;
                }
                // else: failed to aquire lock, re-traverse next iteration.
            }
        }
        __syncthreads();  // __threadfence();

        if (skip == 2) childd[locked] = patch; // unlock
    }
    // record maximum tree depth
    atomicMax((int *)&maxdepthd, localmaxdepth);
}

// Sets mass of cells to -1.0, and all startd entries to null (-1).
__global__
__launch_bounds__(1024, 1)
void ClearKernel2(int nnodesd, volatile int * __restrict startd, volatile float * __restrict node_massd)
{
    register int k, inc, bottom;

    bottom = bottomd;
    inc = blockDim.x * gridDim.x;
    k = (bottom & (-WARPSIZE)) + threadIdx.x + blockIdx.x * blockDim.x;
    if (k < bottom) k += inc;

    // iterate over all cells assigned to thread, skip root cell.
    while (k < nnodesd)
    {
        node_massd[k] = -1.0f;
        startd[k] = -1;
        k += inc;
    }
}


/******************************************************************************/
/*** compute center of mass ***************************************************/
/******************************************************************************/

__global__
__launch_bounds__(THREADS3, FACTOR3)
void SummarizationKernel(const int nnodesd, const int nbodiesd, volatile int * __restrict countd, const int * __restrict childd,
                         volatile float * __restrict body_massd, volatile float * __restrict node_massd, volatile float2 * __restrict body_posd, volatile float2 * __restrict node_posd)
{
    register int i, j, k, ch, inc, cnt, bottom, flag;
    register float m, cm, px, py;
    __shared__ int  child[THREADS3 * 4];
    __shared__ float mass[THREADS3 * 4];

    bottom = bottomd;
    inc = blockDim.x * gridDim.x;
    k = (bottom & (-WARPSIZE)) + threadIdx.x + blockIdx.x * blockDim.x;
    if (k < bottom) k += inc;

    register int restart = k;
    for (j = 0; j < 5; j++)
    {  // wait-free pre-passes
        // iterate over all cells assigned to thread
        while (k <= nnodesd)
        {
            if (node_massd[k] < 0.0f)
            {
                for (i = 0; i < 4; i++)
                {
                    ch = childd[k*4+i];
                    child[i*THREADS3+threadIdx.x] = ch;  // cache children
                    if ((ch >= nbodiesd) && ((mass[i*THREADS3+threadIdx.x] = node_massd[ch]) < 0.0f)) break;
                }
                if (i == 4)
                {
                    // all children are ready
                    cm = 0.0f;
                    px = 0.0f;
                    py = 0.0f;
                    cnt = 0;
                    for (i = 0; i < 4; i++)
                    {
                        ch = child[i*THREADS3+threadIdx.x];
                        if (ch >= 0)
                        {
                            if (ch >= nbodiesd)
                            {  // count bodies (needed later)
                                m = mass[i*THREADS3+threadIdx.x];
                                cnt += countd[ch];
                                px += node_posd[ch].x * m;
                                py += node_posd[ch].y * m;
                            }
                            else
                            {
                                m = body_massd[ch];
                                cnt++;
                                px += body_posd[ch].x * m;
                                py += body_posd[ch].y * m;
                            }
                            // add child's contribution
                            cm += m;
                        }
                    }
                    countd[k] = cnt;
                    m = 1.0f / cm;
                    node_posd[k].x = px * m;
                    node_posd[k].y = py * m;
                    __threadfence();  // make sure data are visible before setting mass
                    node_massd[k] = cm;
                }
            }
            k += inc;  // move on to next cell
        }
        k = restart;
    }

    flag = 0;
    j = 0;
    // iterate over all cells assigned to thread
    while (k <= nnodesd)
    {
        if (k < nbodiesd and body_massd[k] >= 0.0f)
            k += inc;
        else if(k >= nbodiesd and node_massd[k] >= 0.0f)
            k += inc;
        
        else
        {
            if (j == 0)
            {
                j = 4;
                for (i = 0; i < 4; i++)
                {
                    ch = childd[k*4+i];
                    child[i*THREADS3+threadIdx.x] = ch;  // cache children
                    if ((ch < nbodiesd) || ((mass[i*THREADS3+threadIdx.x] = node_massd[ch]) >= 0.0f)) j--;
                }
            }
            else
            {
                j = 4;
                for (i = 0; i < 4; i++)
                {
                    ch = child[i*THREADS3+threadIdx.x];
                    if ((ch < nbodiesd) || (mass[i*THREADS3+threadIdx.x] >= 0.0f) || ((mass[i*THREADS3+threadIdx.x] = node_massd[ch]) >= 0.0f)) j--;
                }
            }

            if (j == 0)
            {
                // all children are ready
                cm = 0.0f;
                px = 0.0f;
                py = 0.0f;
                cnt = 0;
                for (i = 0; i < 4; i++)
                {
                    ch = child[i*THREADS3+threadIdx.x];
                    if (ch >= 0)
                    {
                        if (ch >= nbodiesd)
                        {  // count bodies (needed later)
                            m = mass[i*THREADS3+threadIdx.x];
                            cnt += countd[ch];
                            px += node_posd[ch].x * m;
                            py += node_posd[ch].y * m;
                        }
                        else
                        {
                            m = body_massd[ch];
                            cnt++;
                            px += body_posd[ch].x * m;
                            py += body_posd[ch].y * m;
                        }
                        // add child's contribution
                        cm += m;
                    }
                }
                countd[k] = cnt;
                m = 1.0f / cm;
                node_posd[k].x = px * m;
                node_posd[k].y = py * m;
                flag = 1;
            }
        }
        __syncthreads();  // __threadfence();
        if (flag != 0)
        {
            k < nbodiesd ? body_massd[k] = cm : node_massd[k] = cm;
            k += inc;
            flag = 0;
        }
    }
}


/******************************************************************************/
/*** sort bodies **************************************************************/
/******************************************************************************/

__global__
__launch_bounds__(THREADS4, FACTOR4)
void SortKernel(int nnodesd, int nbodiesd, int * __restrict sortd, int * __restrict countd, volatile int * __restrict startd, int * __restrict childd)
{
    register int i, j, k, ch, dec, start, bottom;

    bottom = bottomd;
    dec = blockDim.x * gridDim.x;
    k = nnodesd + 1 - dec + threadIdx.x + blockIdx.x * blockDim.x;

    // iterate over all cells assigned to thread
    while (k >= bottom)
    {
        start = startd[k];
        if (start >= 0)
        {
            j = 0;
            for (i = 0; i < 4; i++)
            {
                ch = childd[k*4+i];
                if (ch >= 0)
                {
                    if (i != j)
                    {
                        // move children to front (needed later for speed)
                        childd[k*4+i] = -1;
                        childd[k*4+j] = ch;
                    }
                    j++;
                    if (ch >= nbodiesd)
                    {
                        // child is a cell
                        startd[ch] = start;  // set start ID of child
                        start += countd[ch];  // add #bodies in subtree
                    }
                    else
                    {
                        // child is a body
                        sortd[start] = ch;  // record body in 'sorted' array
                        start++;
                    }
                }
            }
            k -= dec;  // move on to next cell
        }
    }
}


/******************************************************************************/
/*** compute force ************************************************************/
/******************************************************************************/

__global__
__launch_bounds__(THREADS5, FACTOR5)
void ForceCalculationKernel(int nnodesd, int nbodiesd, float itolsqd, float epssqd,
                            volatile int * __restrict sortd, volatile int * __restrict childd, 
                            volatile float * __restrict body_massd, volatile float * __restrict node_massd,
                            volatile float2 * __restrict body_posd, volatile float2 * __restrict node_posd,
                            volatile float * __restrict fxd, volatile float * __restrict fyd, const float k_rd)
{
    register int i, j, k, n, depth, base, sbase, diff, pd, nd;
    register float px, py, ax, ay, dx, dy, tmp;
    __shared__ volatile int pos[MAXDEPTH * THREADS5/WARPSIZE], node[MAXDEPTH * THREADS5/WARPSIZE];
    __shared__ float dq[MAXDEPTH * THREADS5/WARPSIZE];

    if (0 == threadIdx.x)
    {
        tmp = radiusd * 2;
        // precompute values that depend only on tree level
        dq[0] = tmp * tmp * itolsqd;
        for (i = 1; i < maxdepthd; i++)
        {
            dq[i] = dq[i - 1] * 0.25f;
            dq[i - 1] += epssqd;
        }
        dq[i - 1] += epssqd;

        if (maxdepthd > MAXDEPTH) errd = maxdepthd;
    }
    __syncthreads();

    if (maxdepthd <= MAXDEPTH)
    {
        // figure out first thread in each warp (lane 0)
        base = threadIdx.x / WARPSIZE;
        sbase = base * WARPSIZE;
        j = base * MAXDEPTH;

        diff = threadIdx.x - sbase;
        // make multiple copies to avoid index calculations later
        if (diff < MAXDEPTH) dq[diff+j] = dq[diff];

        __syncthreads();
        __threadfence_block();

        // iterate over all bodies assigned to thread
        for (k = threadIdx.x + blockIdx.x * blockDim.x; k < nbodiesd; k += blockDim.x * gridDim.x)
        {
            i = sortd[k];  // get permuted/sorted
            // cache position info
            px = body_posd[i].x;
            py = body_posd[i].y;

            ax = 0.0f;
            ay = 0.0f;

            // initialize iteration stack, i.e., push root node onto stack
            depth = j;
            if (sbase == threadIdx.x)
            {
                pos[j] = 0;
                node[j] = nnodesd * 4;
            }

            do
            {
                // stack is not empty
                pd = pos[depth];
                nd = node[depth];
                while (pd < 4)
                {
                    // node on top of stack has more children to process
                    n = childd[nd + pd];  // load child pointer
                    pd++;

                    if (n >= 0)
                    {
                        if(n < nbodiesd)
                        {
                            dx = px - body_posd[n].x;
                            dy = py - body_posd[n].y;
                        }
                        else
                        {
                            dx = px - node_posd[n].x;
                            dy = py - node_posd[n].y;
                        }
                        tmp = dx*dx + dy*dy + epssqd;  // compute distance squared (plus softening)

                        // check body-body interaction
                        if (n < nbodiesd)
                        {
                            ax += k_rd * dx * body_massd[i] * body_massd[n] / tmp;
                            ay += k_rd * dy * body_massd[i] * body_massd[n] / tmp;
                        }
                        
                        // or, if n is cell, ensure all threads agree that cell is far enough away
                        else if(__all(tmp >= dq[depth]))
                        {
                            ax += k_rd * dx * body_massd[i] * node_massd[n] / tmp;
                            ay += k_rd * dy * body_massd[i] * node_massd[n] / tmp;
                        }
                        else
                        {
                            // push cell onto stack
                            if (sbase == threadIdx.x)
                            {  // maybe don't push and inc if last child
                                pos[depth] = pd;
                                node[depth] = nd;
                            }
                            depth++;
                            pd = 0;
                            nd = n * 4;
                        }
                    }
                    else
                    {
                        pd = 4;  // early out because all remaining children are also zero
                    }
                }
                depth--;  // done with this level
            } while (depth >= j);


            // save computed acceleration
            fxd[i] += ax;
            fyd[i] += ay;
        }
    }
}
