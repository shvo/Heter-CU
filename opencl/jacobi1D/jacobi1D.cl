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

__kernel void runJacobi1D_kernel1(__global DATA_TYPE* A, __global DATA_TYPE* B)
{
        //__local DATA_TYPE a0, a1, a2;
        int i = get_global_id(0);
        
        if ( i > 0 & i < 16383) {
	    B[i] = 0.33333 * (A[i-1] + A[i] + A[i + 1]);
            /*
            a0 = A[i-1];
            a1 = A[i];
            a2 = A[i+1];
            B[i] = 0.33333 * (a0 + a1 + a2);
            */
        }
        barrier(CLK_LOCAL_MEM_FENCE);
		
	A[i] = B[i];
        //barrier(CLK_LOCAL_MEM_FENCE);
}
