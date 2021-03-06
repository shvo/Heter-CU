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

#define M 1024  // global size
#define X 16   // local 0-dimension size
#define Y 16   // local 1-dimension size
//#define T 16   // # merged iterations

pipe float p1 __attribute__((xcl_reqd_pipe_depth(64)));
pipe float p2 __attribute__((xcl_reqd_pipe_depth(64)));
pipe float p3 __attribute__((xcl_reqd_pipe_depth(64)));
pipe float p4 __attribute__((xcl_reqd_pipe_depth(64)));
pipe float p5 __attribute__((xcl_reqd_pipe_depth(64)));
pipe float p6 __attribute__((xcl_reqd_pipe_depth(64)));
pipe float p7 __attribute__((xcl_reqd_pipe_depth(64)));
pipe float p8 __attribute__((xcl_reqd_pipe_depth(64)));


__kernel __attribute__ ((reqd_work_group_size(1,1,1)))
void runJacobi2D_kernel1(__global DATA_TYPE* A, __global DATA_TYPE* B, int n)
{
    __local DATA_TYPE A_local[(X+T)*(Y+T)];
    __local DATA_TYPE B_local[(X+T)*(Y+T)];
    int gid_x = get_group_id(0);
    int gid_y = get_group_id(1);

    int i,j;
    if (gid_y == 0) {
        if (gid_x == 0) {
            for (j = 0; j <= Y-1; ++j) {
                async_work_group_copy(&A_local[X*j], &A[(gid_y*2*Y+j)*M + gid_x*2*X], X, 0);
            }
        }
        else {
            for (j = 0; j <= Y-1; ++j) {
                async_work_group_copy(&A_local[(X+T)*j], &A[(gid_y*2*Y+j)*M + gid_x*2*X-T], X+T, 0);
            }
        }
    }
    else {
        if (gid_x == 0) {
            for (j = 0; j <= Y+T-1; ++j) {
                async_work_group_copy(&A_local[X*j], &A[(gid_y*2*Y+j-T)*M + gid_x*2*X], X, 0);
            }
        }
        else {
            for (j = 0; j <= Y+T-1; ++j) {
                async_work_group_copy(&A_local[(X+T)*j], &A[(gid_y*2*Y+j-T)*M + gid_x*2*X-T], X+T, 0);
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int t;
    for (t = 1; t <= T; ++t) { 
        if (gid_y == 0) {
            if (gid_x == 0) { 
                // compute
                for (j = 1; j <= Y-2; ++j) {
                    __attribute__((xcl_pipeline_loop))
                    for (i = 1; i <= X-2; ++i) {
                        B_local[j*X + i] = 0.2f * (A_local[j*X + i] + A_local[j*X + (i - 1)] + A_local[j*X + (i + 1)] + A_local[(j+1)*X + i] + A_local[(j-1)*X + i]);
                    }
                }
                // read from pipe
                for (j = 1; j <= Y-1; ++j) {
                    read_pipe_block(p2, &A_local[j*X + X-1]);
                }
                for (i = 1; i <= X-1; ++i) {
                    read_pipe_block(p2, &A_local[(Y-1)*X + i]);
                }
                // swap
                for (j = 1; j <= Y-2; ++j) {
                    __attribute__((xcl_pipeline_loop))
                    for (i = 1; i <= X-2; ++i) {
                        A_local[j*X + i] = B_local[j*X + i];
                    }
                }
                // write to pipe
                for (j = 1; j <= Y-1; ++j) {
                    write_pipe_block(p1, &A_local[j*X + X-2]);
                }
                for (i = 1; i <= X-1; ++i) {
                    write_pipe_block(p1, &A_local[(Y-2)*X + i]);
                }
            }
            else {
                // compute
                for (j = 1; j <= Y-2; ++j) {
                    __attribute__((xcl_pipeline_loop))
                    for (i = t; i <= X+T-2; ++i) {
                        B_local[j*(X+T) + i] = 0.2f * (A_local[j*(X+T) + i] + A_local[j*(X+T) + (i - 1)] + A_local[j*(X+T) + (i + 1)] + A_local[(j+1)*(X+T) + i] + A_local[(j-1)*(X+T) + i]);
                    }
                }
                // read from pipe
                for (j = 1; j <= Y-1; ++j) {
                    read_pipe_block(p2, &A_local[j*(X+T) + X+T-1]);
                }
                for (i = t; i <= X+T-1; ++i) {
                    read_pipe_block(p2, &A_local[(Y-1)*(X+T) + i]);
                }
                // swap
                for (j = 1; j <= Y-2; ++j) {
                    __attribute__((xcl_pipeline_loop))
                    for (i = t; i <= X+T-1; ++i) {
                        A_local[j*(X+T) + i] = B_local[j*(X+T) + i];
                    }
                }
                // write to pipe
                for (j = 1; j <= Y-1; ++j) {
                    write_pipe_block(p1, &A_local[j*(X+T) + X+T-2]);
                }
                for (i = t; i <= X+T-1; ++i) {
                    write_pipe_block(p1, &A_local[(Y-2)*(X+T) + i]);
                }
            }
        }
        else {
            if (gid_x == 0) {
                // compute
                for (j = t; j <= Y+T-2; ++j) {
                    __attribute__((xcl_pipeline_loop))
                    for (i = 1; i <= X-2; ++i) {
                        B_local[j*X + i] = 0.2f * (A_local[j*X + i] + A_local[j*X + (i - 1)] + A_local[j*X + (i + 1)] + A_local[(j+1)*X + i] + A_local[(j-1)*X + i]);
                    }
                }
                // read from pipe
                for (j = t; j <= Y+T-1; ++j) {
                    read_pipe_block(p2, &A_local[j*X + X-1]);
                }
                for (i = 1; i <= X-1; ++i) {
                    read_pipe_block(p2, &A_local[(Y-1)*X + i]);
                }
                // swap
                for (j = t; j <= Y+T-2; ++j) {
                    __attribute__((xcl_pipeline_loop))
                    for (i = 1; i <= X-2; ++i) {
                        A_local[j*X + i] = B_local[j*X + i];
                    }
                }
                // write to pipe
                for (j = t; j <= Y+T-1; ++j) {
                    write_pipe_block(p1, &A_local[j*X + X-2]);
                }
                for (i = 1; i <= X-1; ++i) {
                    write_pipe_block(p1, &A_local[(Y-2)*X + i]);
                }
            }
            else {
                // compute
                for (j = t; j <= Y+T-2; ++j) {
                    __attribute__((xcl_pipeline_loop))
                    for (i = t; i <= X+T-2; ++i) {
                        B_local[j*(X+T) + i] = 0.2f * (A_local[j*(X+T) + i] + A_local[j*(X+T) + (i - 1)] + A_local[j*(X+T) + (i + 1)] + A_local[(j+1)*(X+T) + i] + A_local[(j-1)*(X+T) + i]);
                    }
                }
                // read from pipe
                for (j = t; j <= Y+T-1; ++j) {
                    read_pipe_block(p2, &A_local[j*(X+T) + X+T-1]);
                }
                for (i = t; i <= X+T-1; ++i) {
                    read_pipe_block(p2, &A_local[(Y-1)*(X+T) + i]);
                }
                // swap
                for (j = t; j <= Y+T-2; ++j) {
                    __attribute__((xcl_pipeline_loop))
                    for (i = t; i <= X+T-2; ++i) {
                        A_local[j*(X+T) + i] = B_local[j*(X+T) + i];
                    }
                }
                // write to pipe
                for (j = t; j <= Y+T-1; ++j) {
                    write_pipe_block(p1, &A_local[j*(X+T) + X+T-2]);
                }
                for (i = t; i <= X+T-1; ++i) {
                    write_pipe_block(p1, &A_local[(Y-2)*(X+T) + i]);
                }
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gid_y == 0) {
        if (gid_x == 0) {
            for (j = 0; j <= Y-1; ++j) {
                async_work_group_copy(&A[(gid_y*2*Y+j)*M + gid_x*2*X], &A_local[X*j], X, 0);
            }
        }
        else {
            for (j = 0; j <= Y-1; ++j) {
                async_work_group_copy(&A[(gid_y*2*Y+j)*M + gid_x*2*X], &A_local[(X+T)*j+T], X, 0);
            }
        }
    }
    else {
        if (gid_x == 0) {
            for (j = 0; j <= Y-1; ++j) {
                async_work_group_copy(&A[(gid_y*2*Y+j)*M + gid_x*2*X], &A_local[X*(j+T)], X, 0);
            }
        }
        else {
            for (j = 0; j <= Y-1; ++j) {
                async_work_group_copy(&A[(gid_y*2*Y+j)*M + gid_x*2*X], &A_local[(X+T)*(j+T)+T], X, 0);
            }
        }
    }
}


__kernel __attribute__ ((reqd_work_group_size(1,1,1)))
void runJacobi2D_kernel2(__global DATA_TYPE* A, __global DATA_TYPE* B, int n)
{
    __local DATA_TYPE A_local[(X+T)*(Y+T)];
    __local DATA_TYPE B_local[(X+T)*(Y+T)];
    int gid_x = get_group_id(0);
    int gid_y = get_group_id(1);

    int i,j;
    if (gid_y == 0) {
        if (gid_x < M/(2*X)-1) {
            for (j = 0; j <= Y-1; ++j) {
                async_work_group_copy(&A_local[(X+T)*j], &A[(gid_y*2*Y+j)*M + gid_x*2*X+X], X+T, 0);
            }
        }
        else {
            for (j = 0; j <= Y-1; ++j) {
                async_work_group_copy(&A_local[X*j], &A[(gid_y*2*Y+j)*M + gid_x*2*X+X], X, 0);
            }
        }
    }
    else {
        if (gid_x < M/(2*X)-1) {
            for (j = 0; i <= Y+T-1; ++j) {
                async_work_group_copy(&A_local[(X+T)*j], &A[(gid_y*2*Y+j-T)*M + gid_x*2*X+X], X+T, 0);
            }
        }
        else {
            for (j = 0; j < Y+T-1; ++j) {
                async_work_group_copy(&A_local[X*j], &A[(gid_y*2*Y+j-T)*M + gid_x*2*X+X], X, 0);
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int t;
    for (t = 1; t <= T; ++t) { 
        if (gid_y == 0) {
            if (gid_x < M/(2*X)-1) {
                // compute
                for (j = 1; j <= Y-2; ++j) {
                    __attribute__((xcl_pipeline_loop))
                    for (i = 1; i <= X+T-1-t; ++i) {
                        B_local[j*(X+T) + i] = 0.2f * (A_local[j*(X+T) + i] + A_local[j*(X+T) + (i - 1)] + A_local[j*(X+T) + (i + 1)] + A_local[(j+1)*(X+T) + i] + A_local[(j-1)*(X+T) + i]);
                    }
                }
                // read from pipe
                for (j = 1; j <= Y-1; ++j) {
                    read_pipe_block(p4, &A_local[j*(X+T)]);
                }
                for (i = 0; i <= X+T-1-t; ++i) {
                    read_pipe_block(p4, &A_local[(Y-1)*(X+T) + i]);
                }
                // swap
                for (j = 1; j <= Y-2; ++j) {
                    __attribute__((xcl_pipeline_loop))
                    for (i = 1; i <= X+T-1-t; ++i) {
                        A_local[j*(X+T) + i] = B_local[j*(X+T) + i];
                    }
                }
                // write to pipe
                for (j = 1; j <= Y-1; ++j) {
                    write_pipe_block(p3, &A_local[j*(X+T) + 1]);
                }
                for (i = 0; i <= X+T-1-t; ++i) {
                    write_pipe_block(p3, &A_local[(Y-2)*(X+T) + i]);
                }
            }
            else {
                // compute
                for (j = 1; j <= Y-2; ++j) {
                    __attribute__((xcl_pipeline_loop))
                    for (i = 1; i <= X-2; ++i) {
                        B_local[j*X + i] = 0.2f * (A_local[j*X + i] + A_local[j*X + (i - 1)] + A_local[j*X + (i + 1)] + A_local[(j+1)*X + i] + A_local[(j-1)*X + i]);
                    }
                }
                // read from pipe
                for (j = 1; j <= Y-1; ++j) {
                    read_pipe_block(p4, &A_local[j*X]);
                }
                for (i = 0; i <= X-2; ++i) {
                    read_pipe_block(p4, &A_local[(Y-1)*X + i]);
                }
                // swap
                for (j = 1; j <= Y-2; ++j) {
                    __attribute__((xcl_pipeline_loop))
                    for (i = 1; i <= X-2; ++i) {
                         A_local[j*X + i] = B_local[j*X + i];
                    }
                }
                // write to pipe
                for (j = 1; j <= Y-1; ++j) {
                    write_pipe_block(p3, &A_local[j*X + 1]);
                }
                for (i = 0; i <= X-2; ++i) {
                    write_pipe_block(p3, &A_local[(Y-2)*X + i]);
                }
            }
        }
        else {
            if (gid_x < M/(2*X)-1) {
                // compute
                for (j = t; j <= Y+T-2; ++j) {
                    __attribute__((xcl_pipeline_loop))
                    for (i = 1; i <= X+T-2-t; ++i) {
                        B_local[j*(X+T) + i] = 0.2f * (A_local[j*(X+T) + i] + A_local[j*(X+T) + (i - 1)] + A_local[j*(X+T) + (i + 1)] + A_local[(j+1)*(X+T) + i] + A_local[(j-1)*(X+T) + i]);
                    }
                }
                // read from pipe
                for (j = t; j <= Y+T-1; ++j) {
                    read_pipe_block(p4, &A_local[j*(X+T)]);
                }
                for (i = 0; i <= X+T-1-t; ++i) {
                    read_pipe_block(p4, &A_local[(Y-1)*(X+T) + i]);
                }
                // swap
                for (j = t; j <= Y+T-2; ++j) {
                    __attribute__((xcl_pipeline_loop))
                    for (i = 1; i <= X+T-1-t; ++i) {
                        A_local[j*(X+T) + i] = B_local[j*(X+T) + i];
                    }
                }
                // write to pipe
                for (j = t; j <= Y+T-1; ++j) {
                    write_pipe_block(p3, &A_local[j*(X+T) + 1]);
                }
                for (i = 0; i <= X+T-1-t; ++i) {
                    write_pipe_block(p3, &A_local[(Y-2)*(X+T) + i]);
                }
            }
            else {
                // compute
                for (j = t; j <= Y+T-2; ++j) {
                    __attribute__((xcl_pipeline_loop))
                    for (i = 1; i <= X-2; ++i) {
                        B_local[j*X + i] = 0.2f * (A_local[j*X + i] + A_local[j*X + (i - 1)] + A_local[j*X + (i + 1)] + A_local[(j+1)*X + i] + A_local[(j-1)*X + i]);
                    }
                }
                // read from pipe
                for (j = t; j <= Y+T-1; ++j) {
                    read_pipe_block(p4, &A_local[j*X]);
                }
                for (i = 0; i <= X-2; ++i) {
                    read_pipe_block(p4, &A_local[(Y-1)*X + i]);
                }
                // swap
                for (j = t; j <= Y+T-2; ++j) {
                    __attribute__((xcl_pipeline_loop))
                    for (i = 1; i <= X-2; ++i) {
                        A_local[j*X + i] = B_local[j*X + i];
                    }
                }
                // write to pipe
                for (j = t; j <= Y+T-1; ++j) {
                    write_pipe_block(p3, &A_local[j*X + 1]);
                }
                for (i = 0; i <= X-2; ++i) {
                    write_pipe_block(p3, &A_local[(Y-2)*X + i]);
                }
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gid_y == 0) {
        if (gid_x < M/(2*X)-1) {
            for (j = 0; j <= Y-1; ++j) {
                async_work_group_copy(&A[(gid_y*2*Y+j)*M + gid_x*2*X+X], &A_local[(X+T)*j], X, 0);
            }
        }
        else {
            for (j = 0; j <= Y-1; ++j) {
                async_work_group_copy(&A[(gid_y*2*Y+j)*M + gid_x*2*X], &A_local[X*j], X, 0);
            }
        }
    }
    else {
        if (gid_x < M/(2*X)-1) {
            for (j = 0; j <= Y-1; ++j) {
                async_work_group_copy(&A[(gid_y*2*Y+j)*M + gid_x*2*X+X], &A_local[(X+T)*(j+T)], X, 0);
            }
        }
        else {
            for (j = 0; j <= Y-1; ++j) {
                async_work_group_copy(&A[(gid_y*2*Y+j)*M + gid_x*2*X+X], &A_local[X*(j+T)], X, 0);
            }
        }
    }
}


__kernel __attribute__ ((reqd_work_group_size(1,1,1)))
void runJacobi2D_kernel3(__global DATA_TYPE* A, __global DATA_TYPE* B, int n)
{
    __local DATA_TYPE A_local[(X+T)*(Y+T)];
    __local DATA_TYPE B_local[(X+T)*(Y+T)];
    int gid_x = get_group_id(0);
    int gid_y = get_group_id(1);

    int i,j;
    if (gid_y < M/(2*Y)-1) {
        if (gid_x == 0) {
            for (j = 0; j <= Y+T-1; ++j) {
                async_work_group_copy(&A_local[X*j], &A[(gid_y*2*Y+j+Y)*M + gid_x*2*X], X, 0);
            }
        }
        else {
            for (j = 0; j <= Y+T-1; ++j) {
                async_work_group_copy(&A_local[(X+T)*j], &A[(gid_y*2*Y+j+Y)*M + gid_x*2*X-T], X+T, 0);
            }
        }
    }
    else {
        if (gid_x == 0) {
            for (j = 0; j <= Y-1; ++j) {
                async_work_group_copy(&A_local[X*j], &A[(gid_y*2*Y+j+Y)*M + gid_x*2*X], X, 0);
            }
        }
        else {
            for (j = 0; j <= Y-1; ++j) {
                async_work_group_copy(&A_local[(X+T)*j], &A[(gid_y*2*Y+j+Y)*M + gid_x*2*X-T], X+T, 0);
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int t;
    for (t = 1; t <= T; ++t) { 
        if (gid_y < M/(2*Y)-1) {
            if (gid_x == 0) { 
                // compute
                for (j = 1; j <= Y+T-1-t; ++j) {
                    __attribute__((xcl_pipeline_loop))
                    for (i = 1; i <= X-2; ++i) {
                        B_local[j*X + i] = 0.2f * (A_local[j*X + i] + A_local[j*X + (i - 1)] + A_local[j*X + (i + 1)] + A_local[(j+1)*X + i] + A_local[(j-1)*X + i]);
                    }
                }
                // read from pipe
                for (j = 0; j <= Y+T-1-t; ++j) {
                    read_pipe_block(p6, &A_local[j*X + X-1]);
                }
                for (i = 1; i <= X-1; ++i) {
                    read_pipe_block(p6, &A_local[i]);
                }
                // swap
                for (j = 1; j <= Y+T-1-t; ++j) {
                    __attribute__((xcl_pipeline_loop))
                    for (i = 1; i <= X-2; ++i) {
                        A_local[j*X + i] = B_local[j*X + i];
                    }
                }
                // write to pipe
                for (j = 0; j <= Y+T-1-t; ++j) {
                    write_pipe_block(p5, &A_local[j*X + X-2]);
                }
                for (i = 1; i <= X-1; ++i) {
                    write_pipe_block(p5, &A_local[X + i]);
                }
            }
            else {
                // compute
                for (j = 1; j <= Y+T-1-t; ++j) {
                    __attribute__((xcl_pipeline_loop))
                    for (i = t; i <= X+T-2; ++i) {
                        B_local[j*(X+T) + i] = 0.2f * (A_local[j*(X+T) + i] + A_local[j*(X+T) + (i - 1)] + A_local[j*(X+T) + (i + 1)] + A_local[(j+1)*(X+T) + i] + A_local[(j-1)*(X+T) + i]);
                    }
                }
                // read from pipe
                for (j = 0; j <= Y+T-1-t; ++j) {
                    read_pipe_block(p6, &A_local[j*(X+T) + X+T-1]);
                }
                for (i = t; i <= X+T-1; ++i) {
                    read_pipe_block(p6, &A_local[i]);
                }
                // swap
                for (j = 1; j <= Y+T-1-t; ++j) {
                    __attribute__((xcl_pipeline_loop))
                    for (i = t; i <= X+T-1; ++i) {
                        A_local[j*(X+T) + i] = B_local[j*(X+T) + i];
                    }
                }
                // write to pipe
                for (j = 0; j <= Y+T-1-t; ++j) {
                    write_pipe_block(p5, &A_local[j*(X+T) + X+T-2]);
                }
                for (i = t; i <= X+T-1; ++i) {
                    write_pipe_block(p5, &A_local[X+T + i]);
                }
            }
        }
        else {
            if (gid_x == 0) {
                // compute
                for (j = 1; j <= Y-2; ++j) {
                    __attribute__((xcl_pipeline_loop))
                    for (i = 1; i <= X-2; ++i) {
                        B_local[j*X + i] = 0.2f * (A_local[j*X + i] + A_local[j*X + (i - 1)] + A_local[j*X + (i + 1)] + A_local[(j+1)*X + i] + A_local[(j-1)*X + i]);
                    }
                }
                // read from pipe
                for (j = 0; j <= Y-2; ++j) {
                    read_pipe_block(p6, &A_local[j*X + X-1]);
                }
                for (i = 1; i <= X-1; ++i) {
                    read_pipe_block(p6, &A_local[i]);
                }
                // swap
                for (j = 1; j <= Y-2; ++j) {
                    __attribute__((xcl_pipeline_loop))
                    for (i = 1; i <= X-2; ++i) {
                        A_local[j*X + i] = B_local[j*X + i];
                    }
                }
                // write to pipe
                for (j = 0; j <= Y-2; ++j) {
                    write_pipe_block(p5, &A_local[j*X + X-2]);
                }
                for (i = 1; i <= X-1; ++i) {
                    write_pipe_block(p5, &A_local[X + i]);
                }
            }
            else {
                // compute
                for (j = 1; j <= Y-2; ++j) {
                    __attribute__((xcl_pipeline_loop))
                    for (i = t; i <= X+T-2; ++i) {
                        B_local[j*(X+T) + i] = 0.2f * (A_local[j*(X+T) + i] + A_local[j*(X+T) + (i - 1)] + A_local[j*(X+T) + (i + 1)] + A_local[(j+1)*(X+T) + i] + A_local[(j-1)*(X+T) + i]);
                    }
                }
                // read from pipe
                for (j = 0; j <= Y-2; ++j) {
                    read_pipe_block(p6, &A_local[j*(X+T) + X+T-1]);
                }
                for (i = t; i <= X+T-1; ++i) {
                    read_pipe_block(p6, &A_local[i]);
                }
                // swap
                for (j = 1; j <= Y-2; ++j) {
                    __attribute__((xcl_pipeline_loop))
                    for (i = t; i <= X+T-2; ++i) {
                        A_local[j*(X+T) + i] = B_local[j*(X+T) + i];
                    }
                }
                // write to pipe
                for (j = 0; j <= Y-2; ++j) {
                    write_pipe_block(p5, &A_local[j*(X+T) + X+T-2]);
                }
                for (i = t; i <= X+T-1; ++i) {
                    write_pipe_block(p5, &A_local[X+T + i]);
                }
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gid_x == 0) {
        for (j = 0; j <= Y-1; ++j) {
            async_work_group_copy(&A[(gid_y*2*Y+j+Y)*M + gid_x*2*X], &A_local[X*j], X, 0);
        }
    }
    else {
        for (j = 0; j <= Y-1; ++j) {
            async_work_group_copy(&A[(gid_y*2*Y+j+Y)*M + gid_x*2*X], &A_local[X*j+T], X, 0);
        }
    }
}

__kernel __attribute__ ((reqd_work_group_size(1,1,1)))
void runJacobi2D_kernel4(__global DATA_TYPE* A, __global DATA_TYPE* B, int n)
{
    __local DATA_TYPE A_local[(X+T)*(Y+T)];
    __local DATA_TYPE B_local[(X+T)*(Y+T)];
    int gid_x = get_group_id(0);
    int gid_y = get_group_id(1);

    int i,j;
    if (gid_y < M/(2*Y)-1) {
        if (gid_x < M/(2*X)-1) {
            for (j = 0; j <= Y+T-1; ++j) {
                async_work_group_copy(&A_local[(X+T)*j], &A[(gid_y*2*Y+j+Y)*M + gid_x*2*X+X], X+T, 0);
            }
        }
        else {
            for (j = 0; j <= Y+T-1; ++j) {
                async_work_group_copy(&A_local[X*j], &A[(gid_y*2*Y+j+Y)*M + gid_x*2*X+X], X, 0);
            }
        }
    }
    else {
        if (gid_x < M/(2*X)-1) {
            for (j = 0; i <= Y-1; ++j) {
                async_work_group_copy(&A_local[(X+T)*j], &A[(gid_y*2*Y+j+Y)*M + gid_x*2*X+X], X+T, 0);
            }
        }
        else {
            for (j = 0; j < Y-1; ++j) {
                async_work_group_copy(&A_local[X*j], &A[(gid_y*2*Y+j+Y)*M + gid_x*2*X+X], X, 0);
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int t;
    for (t = 1; t <= T; ++t) { 
        if (gid_y < M/(2*Y)-1) {
            if (gid_x < M/(2*X)-1) {
                // compute
                for (j = 1; j <= Y+T-1-t; ++j) {
                    __attribute__((xcl_pipeline_loop))
                    for (i = 1; i <= X+T-1-t; ++i) {
                        B_local[j*(X+T) + i] = 0.2f * (A_local[j*(X+T) + i] + A_local[j*(X+T) + (i - 1)] + A_local[j*(X+T) + (i + 1)] + A_local[(j+1)*(X+T) + i] + A_local[(j-1)*(X+T) + i]);
                    }
                }
                // read from pipe
                for (j = 0; j <= Y+T-1-t; ++j) {
                    read_pipe_block(p8, &A_local[j*(X+T)]);
                }
                for (i = 0; i <= X+T-1-t; ++i) {
                    read_pipe_block(p8, &A_local[i]);
                }
                // swap
                for (j = 1; j <= Y+T-1-t; ++j) {
                    __attribute__((xcl_pipeline_loop))
                    for (i = 1; i <= X+T-1-t; ++i) {
                        A_local[j*(X+T) + i] = B_local[j*(X+T) + i];
                    }
                }
                // write to pipe
                for (j = 0; j <= Y+T-1-t; ++j) {
                    write_pipe_block(p7, &A_local[j*(X+T) + 1]);
                }
                for (i = 0; i <= X+T-1-t; ++i) {
                    write_pipe_block(p7, &A_local[X+T + i]);
                }
            }
            else {
                // compute
                for (j = 1; j <= Y+T-1-t; ++j) {
                    __attribute__((xcl_pipeline_loop))
                    for (i = 1; i <= X-2; ++i) {
                        B_local[j*X + i] = 0.2f * (A_local[j*X + i] + A_local[j*X + (i - 1)] + A_local[j*X + (i + 1)] + A_local[(j+1)*X + i] + A_local[(j-1)*X + i]);
                    }
                }
                // read from pipe
                for (j = 0; j <= Y+T-1-t; ++j) {
                    read_pipe_block(p8, &A_local[j*X]);
                }
                for (i = 0; i <= X-2; ++i) {
                    read_pipe_block(p8, &A_local[i]);
                }
                // swap
                for (j = 1; j <= Y+T-1-t; ++j) {
                    __attribute__((xcl_pipeline_loop))
                    for (i = 1; i <= X-2; ++i) {
                         A_local[j*X + i] = B_local[j*X + i];
                    }
                }
                // write to pipe
                for (j = 0; j <= Y+T-1-t; ++j) {
                    write_pipe_block(p7, &A_local[j*X + 1]);
                }
                for (i = 0; i <= X-2; ++i) {
                    write_pipe_block(p7, &A_local[X + i]);
                }
            }
        }
        else {
            if (gid_x < M/(2*X)-1) {
                // compute
                for (j = 1; j <= Y-2; ++j) {
                    __attribute__((xcl_pipeline_loop))
                    for (i = 1; i <= X+T-2-t; ++i) {
                        B_local[j*(X+T) + i] = 0.2f * (A_local[j*(X+T) + i] + A_local[j*(X+T) + (i - 1)] + A_local[j*(X+T) + (i + 1)] + A_local[(j+1)*(X+T) + i] + A_local[(j-1)*(X+T) + i]);
                    }
                }
                // read from pipe
                for (j = 0; j <= Y-2; ++j) {
                    read_pipe_block(p8, &A_local[j*(X+T)]);
                }
                for (i = 0; i <= X+T-1-t; ++i) {
                    read_pipe_block(p8, &A_local[i]);
                }
                // swap
                for (j = 1; j <= Y-2; ++j) {
                    __attribute__((xcl_pipeline_loop))
                    for (i = 1; i <= X+T-1-t; ++i) {
                        A_local[j*(X+T) + i] = B_local[j*(X+T) + i];
                    }
                }
                // write to pipe
                for (j = 0; j <= Y-2; ++j) {
                    write_pipe_block(p7, &A_local[j*(X+T) + 1]);
                }
                for (i = 0; i <= X+T-1-t; ++i) {
                    write_pipe_block(p7, &A_local[X+T + i]);
                }
            }
            else {
                // compute
                for (j = 1; j <= Y-2; ++j) {
                    __attribute__((xcl_pipeline_loop))
                    for (i = 1; i <= X-2; ++i) {
                        B_local[j*X + i] = 0.2f * (A_local[j*X + i] + A_local[j*X + (i - 1)] + A_local[j*X + (i + 1)] + A_local[(j+1)*X + i] + A_local[(j-1)*X + i]);
                    }
                }
                // read from pipe
                for (j = 0; j <= Y-2; ++j) {
                    read_pipe_block(p8, &A_local[j*X]);
                }
                for (i = 0; i <= X-2; ++i) {
                    read_pipe_block(p8, &A_local[i]);
                }
                // swap
                for (j = 1; j <= Y-2; ++j) {
                    __attribute__((xcl_pipeline_loop))
                    for (i = 1; i <= X-2; ++i) {
                        A_local[j*X + i] = B_local[j*X + i];
                    }
                }
                // write to pipe
                for (j = 0; j <= Y-2; ++j) {
                    write_pipe_block(p7, &A_local[j*X + 1]);
                }
                for (i = 0; i <= X-2; ++i) {
                    write_pipe_block(p7, &A_local[X + i]);
                }
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gid_x < M/(2*X)-1) {
        for (j = 0; j <= Y-1; ++j) {
            async_work_group_copy(&A[(gid_y*2*Y+j+Y)*M + gid_x*2*X+X], &A_local[(X+T)*j], X, 0);
        }
    }
    else {
        for (j = 0; j <= Y-1; ++j) {
            async_work_group_copy(&A[(gid_y*2*Y+j+Y)*M + gid_x*2*X+X], &A_local[X*j], X, 0);
        }
    }
}

__kernel __attribute__ ((reqd_work_group_size(1,1,1)))
void runJacobi2D_connect_1_4(__global DATA_TYPE* A)
{
    __local float h1[2*(X+T)], h2[2*(X+T)], h2_buf[2*(X+T)], h3[2*(X+T)], h3_buf[2*(X+T)], h4[2*(X+T)];    
    __local float v1[2*(X+T)], v2[2*(X+T)], v2_buf[2*(X+T)], v3[2*(X+T)], v3_buf[2*(X+T)], v4[2*(X+T)];    

    int gid_x = get_group_id(0);
    int gid_y = get_group_id(1);
  
    int i, j;

    if (gid_x == 0 && gid_y ==0) {
        // initialization
        if (gid_x == 0) {
            async_work_group_copy(h1, &A[(gid_y*2*Y+Y-2)*M + gid_x*2*X], 2*X+T, 0);
            async_work_group_copy(h2_buf, &A[(gid_y*2*Y+Y-1)*M + gid_x*2*X], 2*X+T, 0);
            async_work_group_copy(h3_buf, &A[(gid_y*2*Y+Y)*M + gid_x*2*X], 2*X+T, 0);
            async_work_group_copy(h4, &A[(gid_y*2*Y+Y+1)*M + gid_x*2*X], 2*X+T, 0);
        }
        else if (gid_x < M/(2*X)-1) {
            async_work_group_copy(h1, &A[(gid_y*2*Y+Y-2)*M + gid_x*2*X-T], 2*(X+T), 0);
            async_work_group_copy(h2_buf, &A[(gid_y*2*Y+Y-1)*M + gid_x*2*X-T], 2*(X+T), 0);
            async_work_group_copy(h3_buf, &A[(gid_y*2*Y+Y)*M + gid_x*2*X-T], 2*(X+T), 0);
            async_work_group_copy(h4, &A[(gid_y*2*Y+Y+1)*M + gid_x*2*X-T], 2*X+T, 0);
        }
        else {
            async_work_group_copy(h1, &A[(gid_y*2*Y+Y-2)*M + gid_x*2*X-T], 2*X+T, 0);
            async_work_group_copy(h2_buf, &A[(gid_y*2*Y+Y-1)*M + gid_x*2*X-T], 2*X+T, 0);
            async_work_group_copy(h3_buf, &A[(gid_y*2*Y+Y)*M + gid_x*2*X-T], 2*X+T, 0);
            async_work_group_copy(h4, &A[(gid_y*2*Y+Y+1)*M + gid_x*2*X-T], 2*X+T, 0);
        }

        if (gid_y == 0) {
            __attribute__((xcl_pipeline_loop))
            for (j = 0; j <= 2*Y+T-1; ++j) {
                v1[j] = A[(gid_y*2*Y+j)*M + gid_x*2*X+X-2];
                v2_buf[j] = A[(gid_y*2*Y+j)*M + gid_x*2*X+X-1];
                v3[j] = A[(gid_y*2*Y+j)*M + gid_x*2*X+X];
                v4[j] = A[(gid_y*2*Y+j)*M + gid_x*2*X+X+1];
            }
        }
        else if (gid_y < M/(2*Y)-1) {
            __attribute__((xcl_pipeline_loop))
            for (j = 0; j <= 2*(Y+T)-1; ++j) {
                v1[j] = A[(gid_y*2*Y-T+j)*M + gid_x*2*X+X-2];
                v2_buf[j] = A[(gid_y*2*Y-T+j)*M + gid_x*2*X+X-1];
                v3[j] = A[(gid_y*2*Y-T+j)*M + gid_x*2*X+X];
                v4[j] = A[(gid_y*2*Y-T+j)*M + gid_x*2*X+X+1];
            }
        }
        else {
            __attribute__((xcl_pipeline_loop))
            for (j = 0; j <= 2*Y+T-1; ++j) {
                v1[j] = A[(gid_y*2*Y-T+j)*M + gid_x*2*X+X-2];
                v2_buf[j] = A[(gid_y*2*Y-T+j)*M + gid_x*2*X+X-1];
                v3[j] = A[(gid_y*2*Y-T+j)*M + gid_x*2*X+X];
                v4[j] = A[(gid_y*2*Y-T+j)*M + gid_x*2*X+X+1];
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int t, v_lower, v_upper, h_lower, h_upper;

    for (t = 1; t <= T; ++t) { 
        // compute
        if (gid_y == 0) {
            v_lower = 1;
            v_upper = 2*Y+T-1-t;
        }
        else if (gid_y < M/(2*Y)-1) {
            v_lower = t;
            v_upper = 2*(Y+T)-1-t;
        }
        else {
            v_lower = t;
            v_upper = 2*Y+T-2;
        }
        __attribute__((xcl_pipeline_loop))
        for (j = v_lower; j <= v_upper; ++j) {
            v2[j] = 0.2f * (v2_buf[j-1] + v2_buf[j] + v2_buf[j+1] + v1[j] + v3_buf[j]);
        }
        __attribute__((xcl_pipeline_loop))
        for (j = v_lower; j <= v_upper; ++j) {
            v3[j] = 0.2f * (v3_buf[j-1] + v3_buf[j] + v3_buf[j+1] + v2_buf[j] + v4[j]);
        }
        
        if (gid_x == 0) {
            h_lower = 1;
            h_upper = 2*X+T-1-t;
        }
        else if (gid_x < M/(2*X)-1) {
            h_lower = t;
            h_upper = 2*(X+T)-1-t;
        }
        else {
            h_lower = t;
            h_upper = 2*X+T-2;
        }
        __attribute__((xcl_pipeline_loop))
        for (i = h_lower; i <= h_upper; ++i) {
            h2[i] = 0.2f * (h2_buf[i-1] + h2_buf[i] + h2_buf[i+1] + h1[i] + h3_buf[i]);
        }

        __attribute__((xcl_pipeline_loop))
        for (i = h_lower; i <= h_upper; ++i) {
            h3[i] = 0.2f * (h3_buf[i-1] + h3_buf[i] + h3_buf[i+1] + h2_buf[i] + h4[i]);
        }

        // communicate with compute kernels
        // write vertical data
        if (gid_y == 0) {
            for (j = 1; j <= Y-1; ++j) {
                write_pipe_block(p2, &v2[j]);
                write_pipe_block(p4, &v3[j]);
            }
            for (j = 0; j <= Y+T-1-t; ++j) {
                write_pipe_block(p6, &v2[j+Y]);
                write_pipe_block(p8, &v3[j+Y]);
            }
        }
        else if (gid_y < M/(2*Y)-1) {
            for (j = t; j <= Y+T-1; ++j) {
                write_pipe_block(p2, &v2[j]);
                write_pipe_block(p4, &v3[j]);
            }
            for (j = 0; j <= Y+T-1-t; ++j) {
                write_pipe_block(p6, &v2[j+Y+T]);
                write_pipe_block(p8, &v3[j+Y+T]);
            }
        }
        else {
            for (j = t; j <= Y+T-1; ++j) {
                write_pipe_block(p2, &v2[j]);
                write_pipe_block(p4, &v3[j]);
            }
            for (j = 0; j <= Y-2; ++j) {
                write_pipe_block(p6, &v2[j+Y+T]);
                write_pipe_block(p8, &v3[j+Y+T]);
            }
        }
        // write horizontal data
        if (gid_x == 0) {
            for (i = 1; i <= X-1; ++i) {
                write_pipe_block(p2, &h2[i]);
                write_pipe_block(p6, &h3[i]);
            }
            for (i = 0; i <= X+T-1-t; ++i) {
                write_pipe_block(p4, &h2[i+X]);
                write_pipe_block(p8, &h3[i+X]);
            }
        }
        else if (gid_x < M/(2*X)-1) {
            for (i = t; i <= X+T-1; ++i) {
                write_pipe_block(p2, &h2[i]);
                write_pipe_block(p6, &h3[i]);
            }
            for (i = 0; i <= X+T-1-t; ++i) {
                write_pipe_block(p4, &h2[i+X+T]);
                write_pipe_block(p8, &h3[i+X+T]);
            }
        }
        else {
            for (i = t; i <= X+T-1; ++i) {
                write_pipe_block(p2, &h2[i]);
                write_pipe_block(p6, &h3[i]);
            }
            for (i = 0; i <= X-2; ++i) {
                write_pipe_block(p4, &h2[i+X+T]);
                write_pipe_block(p8, &h3[i+X+T]);
            }
        }

        // read vertical data
        if (gid_y == 0) {
            for (j = 1; j <= Y-1; ++j) {
                read_pipe_block(p1, &v1[j]);
                read_pipe_block(p3, &v4[j]);
            }
            for (j = 0; j <= Y+T-1-t; ++j) {
                read_pipe_block(p5, &v1[j+Y]);
                read_pipe_block(p7, &v4[j+Y]);
            }
        }
        else if (gid_y < M/(2*Y)-1) {
            for (j = t; j <= Y+T-1; ++j) {
                read_pipe_block(p1, &v1[j]);
                read_pipe_block(p3, &v4[j]);
            }
            for (j = 0; j <= Y+T-1-t; ++j) {
                read_pipe_block(p5, &v1[j+Y+T]);
                read_pipe_block(p7, &v4[j+Y+T]);
            }
        }
        else {
            for (j = t; j <= Y+T-1; ++j) {
                read_pipe_block(p1, &v1[j]);
                read_pipe_block(p3, &v4[j]);
            }
            for (j = 0; j <= Y-2; ++j) {
                read_pipe_block(p5, &v1[j+Y+T]);
                read_pipe_block(p7, &v4[j+Y+T]);
            }
        }

        // horizontal data
        if (gid_x == 0) {
            for (i = 1; i <= X-1; ++i) {
                read_pipe_block(p1, &h1[i]);
                read_pipe_block(p5, &h4[i]);
            }
            for (i = 0; i <= X+T-1-t; ++i) {
                read_pipe_block(p3, &h1[i+X]);
                read_pipe_block(p7, &h4[i+X]);
            }
        }
        else if (gid_x < M/(2*X)-1) {
            for (i = t; i <= X+T-1; ++i) {
                read_pipe_block(p1, &h1[i]);
                read_pipe_block(p5, &h4[i]);
            }
            for (i = 0; i <= X+T-1-t; ++i) {
                read_pipe_block(p3, &h1[i+X+T]);
                read_pipe_block(p7, &h4[i+X+T]);
            }
        }
        else {
            for (i = t; i <= X+T-1; ++i) {
                read_pipe_block(p1, &h1[i]);
                read_pipe_block(p5, &h4[i]);
            }
            for (i = 0; i <= X-2; ++i) {
                read_pipe_block(p3, &h1[i+X+T]);
                read_pipe_block(p7, &h4[i+X+T]);
            }
        }
    }
}
