/**
 * jacobi2D.cl: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

typedef float DATA_TYPE;

#define M 16384  // global size
#define N 256    // local size
#define T 256    // # merged iterations


__kernel __attribute__ ((reqd_work_group_size(1,1,1)))
void runJacobi2D_kernel1(__global DATA_TYPE* A, __global DATA_TYPE* B, int n)
{
        __local DATA_TYPE A_local[N+T*2];
        __local DATA_TYPE B_local[N+T*2];
	int i = get_global_id(1);
	int j = get_global_id(0);

	if ((i >= 1) && (i < (n-1)) && (j >= 1) && (j < (n-1)))
	{
		B[i*n + j] = 0.2f * (A[i*n + j] + A[i*n + (j-1)] + A[i*n + (1 + j)] + A[(1 + i)*n + j] 
				+ A[(i-1)*n + j]);	
	}
        barrier(CLK_LOCAL_MEM_FENCE);

	if ((i >= 1) && (i < (n-1)) && (j >= 1) && (j < (n-1)))
	{
		A[i*n + j] = B[i*n + j];
	}
}
