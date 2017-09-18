/// Per kernel launch configuration parameters
// 1: BoundingBoxKernel, SpeedKernel, ComputeLayoutDimensions (reductions of size |V|)
// 2: TreeBuild
// 3: Summurization
// 4: SortKernel
// 5: ForceKernel
// 6: DisplacementKernel, GravityKernel, AttractiveForce (all 'streaming' kernels)
// InitializationKernel, ClearKernel1, ClearKernel2 don't use macros for launch configuration.

#if __CUDA_ARCH__ >= 500 // Maxwell (5.x) or Pascal (6.x)

#define THREADS1 1024  /* must be a power of 2 */
#define THREADS2 1024
#define THREADS3 768
#define THREADS4 128
#define THREADS5 1024
#define THREADS6 1024

#define FACTOR1 2
#define FACTOR2 2
#define FACTOR3 1  /* must all be resident at the same time */
#define FACTOR4 4  /* must all be resident at the same time */
#define FACTOR5 2
#define FACTOR6 2


#elif __CUDA_ARCH__ >= 300 // Kepler (3.x)

#define THREADS1 1024  /* must be a power of 2 */
#define THREADS2 1024
#define THREADS3 768
#define THREADS4 128
#define THREADS5 1024
#define THREADS6 1024

#define FACTOR1 2
#define FACTOR2 2
#define FACTOR3 1  /* must all be resident at the same time */
#define FACTOR4 4  /* must all be resident at the same time */
#define FACTOR5 2
#define FACTOR6 2

#elif __CUDA_ARCH__ < 300 // Fermi (2.x) or Tesla (1.x)

#define THREADS1 512  /* must be a power of 2 */
#define THREADS2 512
#define THREADS3 128
#define THREADS4 64
#define THREADS5 256
#define THREADS6 1024

#define FACTOR1 3
#define FACTOR2 3
#define FACTOR3 6  /* must all be resident at the same time */
#define FACTOR4 6  /* must all be resident at the same time */
#define FACTOR5 5
#define FACTOR6 1

#endif

#define WARPSIZE 32
#define MAXDEPTH 32
