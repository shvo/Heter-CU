/**
 * jacobi1D.cl: This file is part of the PolyBench/GPU 1.0 test suite.
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

__kernel __attribute__ ((reqd_work_group_size(1024,1,1)))
void runJacobi1D_kernel1(__global DATA_TYPE* A, __global DATA_TYPE* B)
{
        __local DATA_TYPE A_local[1026];
        __local DATA_TYPE B_local[1024];
        int g_id = get_group_id(0);
        if (g_id == 0) {
          async_work_group_copy(A_local, A, 1025, 0);
        }
        else if ( g_id == 1 ) {
          async_work_group_copy(A_local, &A[1023], 1026, 0);
        }
        else if ( g_id == 2 ) {
          async_work_group_copy(A_local, &A[2047], 1026, 0);
        }
        else if ( g_id == 3 ) {
          async_work_group_copy(A_local, &A[3071], 1025, 0);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

	int i = get_local_id(0);
        if (g_id == 0) {
	  if (i >= 1)
	  {
	    B_local[i] = 0.33333f * (A_local[i-1] + A_local[i] + A_local[i + 1]);

	  }
          else {
            B_local[i] = A_local[i]; 
          }
        }
        else if (g_id == 1 | g_id == 2) { 
	    B_local[i] = 0.33333f * (A_local[i] + A_local[i+1] + A_local[i + 2]);
        }
        else if (g_id == 3) {
	  if (i < 1023)
	  {
	    B_local[i] = 0.33333f * (A_local[i] + A_local[i+1] + A_local[i + 2]);

	  }
          else {
            B_local[i] = A_local[i]; 
          }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        async_work_group_copy(&A[1024*g_id], B_local, 1024, 0);
}
