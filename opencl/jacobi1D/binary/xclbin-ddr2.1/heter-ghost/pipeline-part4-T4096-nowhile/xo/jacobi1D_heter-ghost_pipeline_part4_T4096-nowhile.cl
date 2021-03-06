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

#define M 16384   // global size
#define N 8192    // local size
#define T 4096    // inner-iteration number

pipe float p1 __attribute__((xcl_reqd_pipe_depth(16)));
pipe float p2 __attribute__((xcl_reqd_pipe_depth(16)));
pipe float p3 __attribute__((xcl_reqd_pipe_depth(16)));
pipe float p4 __attribute__((xcl_reqd_pipe_depth(16)));

__kernel __attribute__ ((reqd_work_group_size(1,1,1)))
void runJacobi1D_kernel1(__global DATA_TYPE* A, __global DATA_TYPE* B)
{
        __local DATA_TYPE A_local[N] __attribute__((xcl_array_partition(cyclic,8,1)));
        __local DATA_TYPE B_local[N] __attribute__((xcl_array_partition(cyclic,8,1)));

        async_work_group_copy(A_local, A, N, 0);
        barrier(CLK_LOCAL_MEM_FENCE);

        int i, t;
            __attribute__((xcl_pipeline_loop))
            for (i = 0; i <= N-2; ++i) {
                /*
                if (i == 0) {
                    B_local[i] = A_local[i]; 
	        }
                else {
                    B_local[i] = 0.33333f * (A_local[i - 1] + A_local[i] + A_local[i + 1]);
                }
                */
                B_local[i] = 0.33333f * (A_local[i - 1] + A_local[i] + A_local[i + 1]);
            }


        barrier(CLK_LOCAL_MEM_FENCE);
        async_work_group_copy(A, B_local, N, 0);
}


/*
__kernel __attribute__ ((reqd_work_group_size(1,1,1)))
void runJacobi1D_kernel1(__global DATA_TYPE* A, __global DATA_TYPE* B)
{
        __local DATA_TYPE A_local[N] __attribute__((xcl_array_partition(cyclic,8,1)));
        __local DATA_TYPE B_local[N] __attribute__((xcl_array_partition(cyclic,8,1)));

        async_work_group_copy(A_local, A, N, 0);
        barrier(CLK_LOCAL_MEM_FENCE);

        write_pipe_block(p1, &A_local[N-2]);
        write_pipe_block(p1, &A_local[N-1]);

        int i, t;
        for (t = 1; t <= T; ++t) {
            __attribute__((xcl_pipeline_loop))
            for (i = 0; i <= N-2; ++i) {
                if (i == 0) {
                    B_local[i] = A_local[i]; 
	        }
                else {
                    B_local[i] = 0.33333f * (A_local[i - 1] + A_local[i] + A_local[i + 1]);
                }
            }
            read_pipe_block(p2, &B_local[N-1]);

            if (t != T) {
                __attribute__((xcl_pipeline_loop))
                for (i = 0; i <= N-1; ++i) {
                    A_local[i] = B_local[i];
                }
                write_pipe_block(p1, &A_local[N-2]);
                write_pipe_block(p1, &A_local[N-1]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        async_work_group_copy(A, B_local, N, 0);
}
*/

__kernel __attribute__ ((reqd_work_group_size(1,1,1)))
void runJacobi1D_kernel2(__global DATA_TYPE* A, __global DATA_TYPE* B)
{
        __local DATA_TYPE A_local[N] __attribute__((xcl_array_partition(cyclic,4,1)));
        __local DATA_TYPE B_local[N] __attribute__((xcl_array_partition(cyclic,4,1)));

        async_work_group_copy(A_local, A, N, 0);
        barrier(CLK_LOCAL_MEM_FENCE);

        write_pipe_block(p3, &A_local[0]);
        write_pipe_block(p3, &A_local[1]);

        int i, t;
        for (t = 1; t <= T; ++t) {
            __attribute__((xcl_pipeline_loop))
            for (i = 1; i <= N-1; ++i) {
                if (i == N-1) {
                    B_local[i] = A_local[i]; 
	        }
                else {
                    B_local[i] = 0.33333f * (A_local[i - 1] + A_local[i] + A_local[i + 1]);
                }
            }
            read_pipe_block(p4, &B_local[0]);

            if (t != T) {
                __attribute__((xcl_pipeline_loop))
                for (i = 0; i <= N-1; ++i) {
                    A_local[i] = B_local[i];
                }
                write_pipe_block(p3, &A_local[0]);
                write_pipe_block(p3, &A_local[1]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        async_work_group_copy(A, B_local, N, 0);
}

__kernel __attribute__ ((reqd_work_group_size(1,1,1)))
void runJacobi1D_connect_1_2(int foo)
{
    float tmp1, tmp2, tmp3, tmp4;
    //while(1) {
        read_pipe_block(p1, &tmp1);
        read_pipe_block(p1, &tmp2);
        read_pipe_block(p3, &tmp3);
        read_pipe_block(p3, &tmp4);

        tmp2 = 0.33333f * (tmp1 + tmp2 + tmp3);
        tmp3 = 0.33333f * (tmp2 + tmp3 + tmp4);
        write_pipe_block(p2, &tmp2);
        write_pipe_block(p4, &tmp3);
    //}
}
