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
#define N 512    // local size
#define T 512    // inner-iteration number

__kernel __attribute__ ((reqd_work_group_size(1,1,1)))
void runJacobi1D_kernel1(__global DATA_TYPE* A, __global DATA_TYPE* B)
{
        __local DATA_TYPE A_local[N+T*2] __attribute__((xcl_array_partition(cyclic,4,1)));;
        __local DATA_TYPE B_local[N+T*2] __attribute__((xcl_array_partition(cyclic,4,1)));;
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

        int i, t;
        for (t = 1; t <= T; ++t) {
            if (g_id == 0) {
                __attribute__((xcl_pipeline_loop))
                for (i = 0; i < N+T - t; ++i) {
                    if (i == 0) {
                        B_local[i] = A_local[i]; 
	            }
                    else {
                        B_local[i] = 0.33333f * (A_local[i - 1] + A_local[i] + A_local[i + 1]);
                    }
                }
                __attribute__((xcl_pipeline_loop))
                for (i = 0; i < N+T - t; ++i) {
                    A_local[i] = B_local[i];
                }
            }
            else if (g_id > 0 & g_id < M/N - 1) { 
                __attribute__((xcl_pipeline_loop))
                for (i = 0; i < N+T*2 - t*2; ++i) {
                    B_local[i] = 0.33333f * (A_local[i + t - 1] + A_local[i + t] + A_local[i + t + 1]);
                }
                __attribute__((xcl_pipeline_loop))
                for (i = 0; i < N+T*2 - t*2; ++i) {
                    A_local[t+i] = B_local[i];
                }
            }
            else {
                __attribute__((xcl_pipeline_loop))
                for (i = t; i < N+T; ++i) {
                    if (i == N+T - 1) {
                        B_local[i] = A_local[i]; 
	            }
                    else {
                        B_local[i] = 0.33333f * (A_local[i - 1] + A_local[i] + A_local[i + 1]);
                    }
                }
                __attribute__((xcl_pipeline_loop))
                for (i = t; i < N+T; ++i) {
                       A_local[i] = B_local[i];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (g_id == M/N - 1 ) {
            async_work_group_copy(&A[M-N], &B_local[T], N, 0);
        }
        else {
            async_work_group_copy(&A[N*g_id], B_local, N, 0);
        }
}
