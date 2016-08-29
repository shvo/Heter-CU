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
#define N 2730    // local size
//#define T 2048    // inner-iteration number

pipe float p1 __attribute__((xcl_reqd_pipe_depth(16)));
pipe float p2 __attribute__((xcl_reqd_pipe_depth(16)));
pipe float p3 __attribute__((xcl_reqd_pipe_depth(16)));
pipe float p4 __attribute__((xcl_reqd_pipe_depth(16)));
pipe float p5 __attribute__((xcl_reqd_pipe_depth(16)));
pipe float p6 __attribute__((xcl_reqd_pipe_depth(16)));
pipe float p7 __attribute__((xcl_reqd_pipe_depth(16)));
pipe float p8 __attribute__((xcl_reqd_pipe_depth(16)));
pipe float p9 __attribute__((xcl_reqd_pipe_depth(16)));
pipe float p10 __attribute__((xcl_reqd_pipe_depth(16)));
pipe float p11 __attribute__((xcl_reqd_pipe_depth(16)));
pipe float p12 __attribute__((xcl_reqd_pipe_depth(16)));


__kernel __attribute__ ((reqd_work_group_size(1,1,1)))
void runJacobi1D_kernel1(__global DATA_TYPE* A, __global DATA_TYPE* B)
{
        __local DATA_TYPE A_local[N+2] __attribute__((xcl_array_partition(cyclic,4,1)));
        __local DATA_TYPE B_local[N+2] __attribute__((xcl_array_partition(cyclic,4,1)));

        async_work_group_copy(A_local, A, N+2, 0);
        barrier(CLK_LOCAL_MEM_FENCE);

        write_pipe_block(p1, &A_local[N]);
        write_pipe_block(p1, &A_local[N+1]);

        int i, t;
        for (t = 1; t <= T; ++t) {
            __attribute__((xcl_pipeline_loop))
            for (i = 0; i <= N; ++i) {
                if (i == 0) {
                    B_local[i] = A_local[i]; 
	        }
                else {
                    B_local[i] = 0.33333f * (A_local[i - 1] + A_local[i] + A_local[i + 1]);
                }
            }
            read_pipe_block(p2, &B_local[N+1]);

            if (t != T) {
                __attribute__((xcl_pipeline_loop))
                for (i = 0; i <= N+1; ++i) {
                    A_local[i] = B_local[i];
                }
                write_pipe_block(p1, &A_local[N]);
                write_pipe_block(p1, &A_local[N+1]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        async_work_group_copy(A, B_local, N+2, 0);
}


__kernel __attribute__ ((reqd_work_group_size(1,1,1)))
void runJacobi1D_kernel2(__global DATA_TYPE* A, __global DATA_TYPE* B)
{
        __local DATA_TYPE A_local[N] __attribute__((xcl_array_partition(cyclic,4,1)));
        __local DATA_TYPE B_local[N] __attribute__((xcl_array_partition(cyclic,4,1)));

        async_work_group_copy(A_local, A, N, 0);
        barrier(CLK_LOCAL_MEM_FENCE);

        write_pipe_block(p3, &A_local[0]);
        write_pipe_block(p3, &A_local[1]);
        write_pipe_block(p3, &A_local[N-2]);
        write_pipe_block(p3, &A_local[N-1]);

        int i, t;
        for (t = 1; t <= T; ++t) {
            __attribute__((xcl_pipeline_loop))
            for (i = 1; i <= N-2; ++i) {
                B_local[i] = 0.33333f * (A_local[i - 1] + A_local[i] + A_local[i + 1]);
            }
            read_pipe_block(p4, &B_local[0]);
            read_pipe_block(p4, &B_local[N-1]);

            if (t != T) {
                __attribute__((xcl_pipeline_loop))
                for (i = 0; i <= N-1; ++i) {
                    A_local[i] = B_local[i];
                }
                write_pipe_block(p3, &A_local[0]);
                write_pipe_block(p3, &A_local[1]);
                write_pipe_block(p3, &A_local[N-2]);
                write_pipe_block(p3, &A_local[N-1]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        async_work_group_copy(A, B_local, N, 0);
}

__kernel __attribute__ ((reqd_work_group_size(1,1,1)))
void runJacobi1D_kernel3(__global DATA_TYPE* A, __global DATA_TYPE* B)
{
        __local DATA_TYPE A_local[N] __attribute__((xcl_array_partition(cyclic,4,1)));
        __local DATA_TYPE B_local[N] __attribute__((xcl_array_partition(cyclic,4,1)));

        async_work_group_copy(A_local, A, N, 0);
        barrier(CLK_LOCAL_MEM_FENCE);

        write_pipe_block(p5, &A_local[0]);
        write_pipe_block(p5, &A_local[1]);
        write_pipe_block(p5, &A_local[N-2]);
        write_pipe_block(p5, &A_local[N-1]);

        int i, t;
        for (t = 1; t <= T; ++t) {
            __attribute__((xcl_pipeline_loop))
            for (i = 1; i <= N-2; ++i) {
                B_local[i] = 0.33333f * (A_local[i - 1] + A_local[i] + A_local[i + 1]);
            }
            read_pipe_block(p6, &B_local[0]);
            read_pipe_block(p6, &B_local[N-1]);

            if (t != T) {
                __attribute__((xcl_pipeline_loop))
                for (i = 0; i <= N-1; ++i) {
                    A_local[i] = B_local[i];
                }
                write_pipe_block(p5, &A_local[0]);
                write_pipe_block(p5, &A_local[1]);
                write_pipe_block(p5, &A_local[N-2]);
                write_pipe_block(p5, &A_local[N-1]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        async_work_group_copy(A, B_local, N, 0);
}

__kernel __attribute__ ((reqd_work_group_size(1,1,1)))
void runJacobi1D_kernel4(__global DATA_TYPE* A, __global DATA_TYPE* B)
{
        __local DATA_TYPE A_local[N] __attribute__((xcl_array_partition(cyclic,4,1)));
        __local DATA_TYPE B_local[N] __attribute__((xcl_array_partition(cyclic,4,1)));

        async_work_group_copy(A_local, A, N, 0);
        barrier(CLK_LOCAL_MEM_FENCE);

        write_pipe_block(p7, &A_local[0]);
        write_pipe_block(p7, &A_local[1]);
        write_pipe_block(p7, &A_local[N-2]);
        write_pipe_block(p7, &A_local[N-1]);

        int i, t;
        for (t = 1; t <= T; ++t) {
            __attribute__((xcl_pipeline_loop))
            for (i = 1; i <= N-2; ++i) {
                B_local[i] = 0.33333f * (A_local[i - 1] + A_local[i] + A_local[i + 1]);
            }
            read_pipe_block(p8, &B_local[0]);
            read_pipe_block(p8, &B_local[N-1]);

            if (t != T) {
                __attribute__((xcl_pipeline_loop))
                for (i = 0; i <= N-1; ++i) {
                    A_local[i] = B_local[i];
                }
                write_pipe_block(p7, &A_local[0]);
                write_pipe_block(p7, &A_local[1]);
                write_pipe_block(p7, &A_local[N-2]);
                write_pipe_block(p7, &A_local[N-1]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        async_work_group_copy(A, B_local, N, 0);
}

__kernel __attribute__ ((reqd_work_group_size(1,1,1)))
void runJacobi1D_kernel5(__global DATA_TYPE* A, __global DATA_TYPE* B)
{
        __local DATA_TYPE A_local[N] __attribute__((xcl_array_partition(cyclic,4,1)));
        __local DATA_TYPE B_local[N] __attribute__((xcl_array_partition(cyclic,4,1)));

        async_work_group_copy(A_local, A, N, 0);
        barrier(CLK_LOCAL_MEM_FENCE);

        write_pipe_block(p9, &A_local[0]);
        write_pipe_block(p9, &A_local[1]);
        write_pipe_block(p9, &A_local[N-2]);
        write_pipe_block(p9, &A_local[N-1]);

        int i, t;
        for (t = 1; t <= T; ++t) {
            __attribute__((xcl_pipeline_loop))
            for (i = 1; i <= N-2; ++i) {
                B_local[i] = 0.33333f * (A_local[i - 1] + A_local[i] + A_local[i + 1]);
            }
            read_pipe_block(p10, &B_local[0]);
            read_pipe_block(p10, &B_local[N-1]);

            if (t != T) {
                __attribute__((xcl_pipeline_loop))
                for (i = 0; i <= N-1; ++i) {
                    A_local[i] = B_local[i];
                }
                write_pipe_block(p9, &A_local[0]);
                write_pipe_block(p9, &A_local[1]);
                write_pipe_block(p9, &A_local[N-2]);
                write_pipe_block(p9, &A_local[N-1]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        async_work_group_copy(A, B_local, N, 0);
}

__kernel __attribute__ ((reqd_work_group_size(1,1,1)))
void runJacobi1D_kernel6(__global DATA_TYPE* A, __global DATA_TYPE* B)
{
        __local DATA_TYPE A_local[N+2] __attribute__((xcl_array_partition(cyclic,4,1)));
        __local DATA_TYPE B_local[N+2] __attribute__((xcl_array_partition(cyclic,4,1)));

        async_work_group_copy(A_local, A, N+2, 0);
        barrier(CLK_LOCAL_MEM_FENCE);

        write_pipe_block(p11, &A_local[0]);
        write_pipe_block(p11, &A_local[1]);

        int i, t;
        for (t = 1; t <= T; ++t) {
            __attribute__((xcl_pipeline_loop))
            for (i = 1; i <= N+1; ++i) {
                if (i == N+1) {
                    B_local[i] = A_local[i]; 
	        }
                else {
                    B_local[i] = 0.33333f * (A_local[i - 1] + A_local[i] + A_local[i + 1]);
                }
            }
            read_pipe_block(p12, &B_local[0]);

            if (t != T) {
                __attribute__((xcl_pipeline_loop))
                for (i = 0; i <= N+1; ++i) {
                    A_local[i] = B_local[i];
                }
                write_pipe_block(p11, &A_local[0]);
                write_pipe_block(p11, &A_local[1]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        async_work_group_copy(A, B_local, N+2, 0);
}

__kernel __attribute__ ((reqd_work_group_size(1,1,1)))
void runJacobi1D_connect_1_6(int foo)
{
    float tmp1, tmp2, tmp3, tmp4,  tmp5,  tmp6, tmp7;
    float tmp8, tmp9, tmp10, tmp11, tmp12, tmp13, tmp14;
    float tmp15, tmp16, tmp17, tmp18, tmp19, tmp20;
    //while(1) {
        // read from kernel1
        read_pipe_block(p1, &tmp1);
        read_pipe_block(p1, &tmp2);

        // read from kernel2
        read_pipe_block(p3, &tmp3);
        read_pipe_block(p3, &tmp4);
        read_pipe_block(p3, &tmp5);
        read_pipe_block(p3, &tmp6);

        // read from kernel3
        read_pipe_block(p5, &tmp7);
        read_pipe_block(p5, &tmp8);
        read_pipe_block(p5, &tmp9);
        read_pipe_block(p5, &tmp10);

        // read from kernel4
        read_pipe_block(p7, &tmp11);
        read_pipe_block(p7, &tmp12);
        read_pipe_block(p7, &tmp13);
        read_pipe_block(p7, &tmp14);

        // read from kernel5
        read_pipe_block(p9, &tmp15);
        read_pipe_block(p9, &tmp16);
        read_pipe_block(p9, &tmp17);
        read_pipe_block(p9, &tmp18);

        // read from kernel6
        read_pipe_block(p11, &tmp19);
        read_pipe_block(p11, &tmp20);


        // calculate
        tmp2  = 0.33333f * (tmp1  + tmp2  + tmp3);
        tmp3  = 0.33333f * (tmp2  + tmp3  + tmp4);
        tmp6  = 0.33333f * (tmp5  + tmp6  + tmp7);
        tmp7  = 0.33333f * (tmp6  + tmp7  + tmp8);
        tmp10 = 0.33333f * (tmp9  + tmp10 + tmp11);
        tmp11 = 0.33333f * (tmp10 + tmp11 + tmp12);
        tmp14 = 0.33333f * (tmp13 + tmp14 + tmp15);
        tmp15 = 0.33333f * (tmp14 + tmp15 + tmp16);
        tmp18 = 0.33333f * (tmp17 + tmp18 + tmp19);
        tmp19 = 0.33333f * (tmp18 + tmp19 + tmp20);

        // write to kernel1
        write_pipe_block(p2, &tmp2);

        // write to kernel2
        write_pipe_block(p4, &tmp3);
        write_pipe_block(p4, &tmp6);

        // write to kernel3
        write_pipe_block(p6, &tmp7);
        write_pipe_block(p6, &tmp10);

        // write to kernel4
        write_pipe_block(p8, &tmp11);
        write_pipe_block(p8, &tmp14);

        // write to kernel5
        write_pipe_block(p10, &tmp15);
        write_pipe_block(p10, &tmp18);

        // write to kernel6
        write_pipe_block(p12, &tmp19);
    //}
}
