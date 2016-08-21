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

#define M 16384  // global size
#define N 256    // local size
#define T 256    // inner-iteration number

__kernel __attribute__ ((reqd_work_group_size(1,1,1)))
void runJacobi1D_kernel1(__global DATA_TYPE* A, __global DATA_TYPE* B)
{
        __local DATA_TYPE A_local[N+T*2];
        __local DATA_TYPE B_local[N+T*2];
        int g_id = get_group_id(0);
        if ( g_id == 0 ) {
          async_work_group_copy(A_local, A, N+T, 0);
        } 
        else if ( g_id > 0 & g_id < M/N - 1 ) {
          async_work_group_copy(A_local, &A[N*g_id-T], N+T*2, 0);
        }
        else if ( g_id == M/N - 1 ) {
          async_work_group_copy(A_local, &A[N*g_id-T], N+T, 0);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        int i = 0, t = 0;
        for (t = 1; t <= T; ++t) {
            if (g_id == 0) {
                for (i = 0; i < N+T - t; ++i) {
                    if (i == 0) {
                        B_local[i] = A_local[i]; 
	            }
                    else {
                        B_local[i] = 0.33333f * (A_local[i-1] + A_local[i] + A_local[i + 1]);
                    }
                }
                for (i = 0; i < N+T - t; ++i) {
                    A_local[i] = B_local[i];
                }
            }
            else if (g_id > 0 & g_id < M/N - 1) { 
                for (i = 0; i < N+T*2 - t*2; ++i) {
                    B_local[i] = 0.33333f * (A_local[i + t - 1] + A_local[i + t] + A_local[i + t + 1]);
                }
                for (i = 0; i < N+T*2 - t*2; ++i) {
                    A_local[t+i] = B_local[i];
                }
            }
            else {
                for (i = 0; i < N+T - t; ++i) {
                    if (i == N+T - t - 1) {
                        B_local[i] = A_local[i]; 
	            }
                    else {
                        B_local[i] = 0.33333f * (A_local[i + t - 1] + A_local[i + t] + A_local[i + t + 1]);
                    }
                }
                for (i = 0; i < N+T - t; ++i) {
                       A_local[i + t] = B_local[i];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        async_work_group_copy(&A[N*g_id], B_local, N, 0);
}
