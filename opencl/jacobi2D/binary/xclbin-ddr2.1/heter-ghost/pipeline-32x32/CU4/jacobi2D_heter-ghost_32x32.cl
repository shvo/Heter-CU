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
#define X 32   // local 0-dimension size
#define Y 32   // local 1-dimension size
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
            for (i = 0; i < Y; ++i) {
                async_work_group_copy(&A_local[X*i], &A[(gid_y*2*Y+i)*M + gid_x*2*X], X, 0);
            }
        }
        else if (gid_x < M/(2*X)-1) {
            for (i = 0; i < Y; ++i) {
                async_work_group_copy(&A_local[(X+T)*i], &A[(gid_y*2*Y+i)*M + gid_x*2*X-T], X+T, 0);
            }
        }
        else {
            for (i = 0; i < Y; ++i) {
                async_work_group_copy(&A_local[(X+T)*i], &A[(gid_y*2*Y+i)*M + gid_x*2*X-T], X+T, 0);
            }
        }
    }
    else if (gid_y < M/(2*Y)-1) {
        if (gid_x == 0) {
            for (i = 0; i < Y + T; ++i) {
                async_work_group_copy(&A_local[X*i], &A[(gid_y*2*Y+i-T)*M + gid_x*2*X], X, 0);
            }
        }
        else if (gid_x < M/(2*X)-1) {
            for (i = 0; i < Y + T; ++i) {
                async_work_group_copy(&A_local[(X+T)*i], &A[(gid_y*2*Y+i-T)*M + gid_x*2*X-T], X+T, 0);
            }
        }
        else {
            for (i = 0; i < Y + T; ++i) {
                async_work_group_copy(&A_local[(X+T)*i], &A[(gid_y*2*Y+i-T)*M + gid_x*2*X-T], X+T, 0);
            }
        }
    }
    else {
        if (gid_x == 0) {
            for (i = 0; i < Y + T; ++i) {
                async_work_group_copy(&A_local[X*i], &A[(gid_y*2*Y+i-T)*M + gid_x*2*X], X, 0);
            }
        }
        else if (gid_x < M/(2*X)-1) {
            for (i = 0; i < Y + T; ++i) {
                async_work_group_copy(&A_local[(X+T)*i], &A[(gid_y*2*Y+i-T)*M + gid_x*2*X-T], X+T, 0);
            }
        }
        else {
            for (i = 0; i < Y + T; ++i) {
                async_work_group_copy(&A_local[(X+T)*i], &A[(gid_y*2*Y+i-T)*M + gid_x*2*X-T], X+T, 0);
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // inital data for pipe
    if (gid_y == 0) {
        if (gid_x == 0) {
            for (j = 0; j <= Y-1; ++j) {
                write_pipe_block(p1, &A_local[j*X + X-2]);
                write_pipe_block(p1, &A_local[j*X + X-1]);
            }
            for (i = 0; i <= X-1; ++i) {
                write_pipe_block(p1, &A_local[(Y-2)*X + i]);
                write_pipe_block(p1, &A_local[(Y-1)*X + i]);
            }
        }
        else if (gid_x < M/(2*X)-1) {
            for (j = 0; j <= Y-1; ++j) {
                write_pipe_block(p1, &A_local[j*X + X-2]);
                write_pipe_block(p1, &A_local[j*X + X-1]);
            }
            for (i = 0; i <= X+T-1; ++i) {
                write_pipe_block(p1, &A_local[(Y-2)*X + i]);
                write_pipe_block(p1, &A_local[(Y-1)*X + i]);
            }
        }
        else {
            for (j = 0; j <= Y-1; ++j) {
                write_pipe_block(p1, &A_local[j*X + X-2]);
                write_pipe_block(p1, &A_local[j*X + X-1]);
            }
            for (i = 0; i <= X+T-1; ++i) {
                write_pipe_block(p1, &A_local[(Y-2)*X + i]);
                write_pipe_block(p1, &A_local[(Y-1)*X + i]);
            }
        }
    }
    else if (gid_y < M/(2*Y)-1) {
        if (gid_x == 0) {
            for (j = 0; j <= Y+T-1; ++j) {
                write_pipe_block(p1, &A_local[j*X + X-2]);
                write_pipe_block(p1, &A_local[j*X + X-1]);
            }
            for (i = 0; i <= X-1; ++i) {
                write_pipe_block(p1, &A_local[(Y-2)*X + i]);
                write_pipe_block(p1, &A_local[(Y-1)*X + i]);
            }
        }
        else if (gid_x < M/(2*X)-1) {
            for (j = 0; j <= Y+T-1; ++j) {
                write_pipe_block(p1, &A_local[j*X + X-2]);
                write_pipe_block(p1, &A_local[j*X + X-1]);
            }
            for (i = 0; i <= X+T-1; ++i) {
                write_pipe_block(p1, &A_local[(Y-2)*X + i]);
                write_pipe_block(p1, &A_local[(Y-1)*X + i]);
            }
        }
        else {
            for (j = 0; j <= Y+T-1; ++j) {
                write_pipe_block(p1, &A_local[j*X + X-2]);
                write_pipe_block(p1, &A_local[j*X + X-1]);
            }
            for (i = 0; i <= X+T-1; ++i) {
                write_pipe_block(p1, &A_local[(Y-2)*X + i]);
                write_pipe_block(p1, &A_local[(Y-1)*X + i]);
            }
        }
    }
    else {
        if (gid_x == 0) {
            for (j = 0; j <= Y+T-1; ++j) {
                write_pipe_block(p1, &A_local[j*X + X-2]);
                write_pipe_block(p1, &A_local[j*X + X-1]);
            }
            for (i = 0; i <= X-1; ++i) {
                write_pipe_block(p1, &A_local[(Y-2)*X + i]);
                write_pipe_block(p1, &A_local[(Y-1)*X + i]);
            }
        }
        else if (gid_x < M/(2*X)-1) {
            for (j = 0; j <= Y+T-1; ++j) {
                write_pipe_block(p1, &A_local[j*X + X-2]);
                write_pipe_block(p1, &A_local[j*X + X-1]);
            }
            for (i = 0; i <= X+T-1; ++i) {
                write_pipe_block(p1, &A_local[(Y-2)*X + i]);
                write_pipe_block(p1, &A_local[(Y-1)*X + i]);
            }
        }
        else {
            for (j = 0; j <= Y+T-1; ++j) {
                write_pipe_block(p1, &A_local[j*X + X-2]);
                write_pipe_block(p1, &A_local[j*X + X-1]);
            }
            for (i = 0; i <= X+T-1; ++i) {
                write_pipe_block(p1, &A_local[(Y-2)*X + i]);
                write_pipe_block(p1, &A_local[(Y-1)*X + i]);
            }
        }
    }

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
                for (j = 0; j <= Y-1; ++j) {
                    write_pipe_block(p1, &A_local[j*X + X-2]);
                }
                for (i = 0; i <= X-1; ++i) {
                    write_pipe_block(p1, &A_local[(Y-2)*X + i]);
                }
            }
            else if (gid_x < M/(2*X)-1) {
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
                for (j = 0; j <= Y-1; ++j) {
                    write_pipe_block(p1, &A_local[j*(X+T) + X+T-2]);
                }
                for (i = t; i <= X+T-1; ++i) {
                    write_pipe_block(p1, &A_local[(Y-2)*(X+T) + i]);
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
                for (j = 1; j <= Y+T-1-t; ++j) {
                    __attribute__((xcl_pipeline_loop))
                    for (i = t; i <= X+T-2; ++i) {
                         A_local[j*(X+T) + i] = B_local[j*(X+T) + i];
                    }
                }
                // write to pipe
                for (j = 0; j <= Y-1; ++j) {
                    write_pipe_block(p1, &A_local[j*(X+T) + X+T-2]);
                }
                for (i = t; i <= X+T-1; ++i) {
                    write_pipe_block(p1, &A_local[(Y-2)*(X+T) + i]);
                }
            }
        }
        else if (gid_y < M/(2*Y)-1) {
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
                for (i = 0; i <= X-1; ++i) {
                    write_pipe_block(p1, &A_local[(Y-2)*X + i]);
                }
            }
            else if (gid_x < M/(2*X)-1) {
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
                for (i = 0; i <= X-1; ++i) {
                    write_pipe_block(p1, &A_local[(Y-2)*X + i]);
                }
            }
            else if (gid_x < M/(2*X)-1) {
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
        else if (gid_x < M/(2*X)-1) {
            for (j = 0; j <= Y-1; ++j) {
                async_work_group_copy(&A[(gid_y*2*Y+j)*M + gid_x*2*X], &A_local[(X+T)*j+T], X, 0);
            }
        }
        else {
            for (j = 0; j <= Y-1; ++j) {
                async_work_group_copy(&A[(gid_y*2*Y+j)*M + gid_x*2*X], &A_local[(X+T)*j+T], X, 0);
            }
        }
    }
    else if (gid_y < M/(2*Y)-1) {
        if (gid_x == 0) {
            for (j = 0; j <= Y-1; ++j) {
                async_work_group_copy(&A[(gid_y*2*Y+j)*M + gid_x*2*X], &A_local[X*(j+T)], X, 0);
            }
        }
        else if (gid_x < M/(2*X)-1) {
            for (j = 0; j <= Y-1; ++j) {
                async_work_group_copy(&A[(gid_y*2*Y+j)*M + gid_x*2*X], &A_local[(X+T)*(j+T)+T], X, 0);
            }
        }
        else {
            for (j = 0; j <= Y-1; ++j) {
                async_work_group_copy(&A[(gid_y*2*Y+j)*M + gid_x*2*X], &A_local[(X+T)*(j+T)+T], X, 0);
            }
        }
    }
    else {
        if (gid_x == 0) {
            for (j = 0; j <= Y-1; ++j) {
                async_work_group_copy(&A[(gid_y*2*Y+j)*M + gid_x*2*X], &A_local[X*(j+T)], X, 0);
            }
        }
        else if (gid_x < M/X-1) {
            for (j = 0; j <= Y-1; ++j) {
                async_work_group_copy(&A[(gid_y*2*Y+j)*M + gid_x*2*X], &A_local[(X+T)*(j+T)+T], X, 0);
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
        if (gid_x == 0) {
            for (i = 0; i < Y; ++i) {
                async_work_group_copy(&A_local[(X+T)*i], &A[(gid_y*2*Y+i)*M + gid_x*2*X+X], X+T, 0);
            }
        }
        else if (gid_x < M/(2*X)-1) {
            for (i = 0; i < Y; ++i) {
                async_work_group_copy(&A_local[(X+T)*i], &A[(gid_y*2*Y+i)*M + gid_x*2*X+X], X+T, 0);
            }
        }
        else {
            for (i = 0; i < Y; ++i) {
                async_work_group_copy(&A_local[X*i], &A[(gid_y*2*Y+i)*M + gid_x*2*X+X], X, 0);
            }
        }
    }
    else if (gid_y < M/(2*Y)-1) {
        if (gid_x == 0) {
            for (i = 0; i < Y + T; ++i) {
                async_work_group_copy(&A_local[(X+T)*i], &A[(gid_y*2*Y+i-T)*M + gid_x*2*X+X], X+T, 0);
            }
        }
        else if (gid_x < M/(2*X)-1) {
            for (i = 0; i < Y + T; ++i) {
                async_work_group_copy(&A_local[(X+T)*i], &A[(gid_y*2*Y+i-T)*M + gid_x*2*X+X], X+T, 0);
            }
        }
        else {
            for (i = 0; i < Y + T; ++i) {
                async_work_group_copy(&A_local[X*i], &A[(gid_y*2*Y+i-T)*M + gid_x*2*X+X], X, 0);
            }
        }
    }
    else {
        if (gid_x == 0) {
            for (i = 0; i < Y + T; ++i) {
                async_work_group_copy(&A_local[(X+T)*i], &A[(gid_y*2*Y+i-T)*M + gid_x*2*X+X], X+T, 0);
            }
        }
        else if (gid_x < M/(2*X)-1) {
            for (i = 0; i < Y + T; ++i) {
                async_work_group_copy(&A_local[(X+T)*i], &A[(gid_y*2*Y+i-T)*M + gid_x*2*X+X], X+T, 0);
            }
        }
        else {
            for (i = 0; i < Y + T; ++i) {
                async_work_group_copy(&A_local[X*i], &A[(gid_y*2*Y+i-T)*M + gid_x*2*X+X], X, 0);
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // inital data for pipe
    if (gid_y == 0) {
        if (gid_x == 0) {
            for (j = 0; j <= Y-1; ++j) {
                write_pipe_block(p1, &A_local[j*(X+T)]);
                write_pipe_block(p1, &A_local[j*(X+T) + 1]);
            }
            for (i = 0; i <= X+T-1; ++i) {
                write_pipe_block(p1, &A_local[(Y-2)*(X+T) + i]);
                write_pipe_block(p1, &A_local[(Y-1)*(X+T) + i]);
            }
        }
        else if (gid_x < M/(2*X)-1) {
            for (j = 0; j <= Y-1; ++j) {
                write_pipe_block(p1, &A_local[j*(X+T)]);
                write_pipe_block(p1, &A_local[j*(X+T) + 1]);
            }
            for (i = 0; i <= X+T-1; ++i) {
                write_pipe_block(p1, &A_local[(Y-2)*(X+T) + i]);
                write_pipe_block(p1, &A_local[(Y-1)*(X+T) + i]);
            }
        }
        else {
            for (j = 0; j <= Y-1; ++j) {
                write_pipe_block(p1, &A_local[j*X]);
                write_pipe_block(p1, &A_local[j*X + 1]);
            }
            for (i = 0; i <= X-1; ++i) {
                write_pipe_block(p1, &A_local[(Y-2)*X + i]);
                write_pipe_block(p1, &A_local[(Y-1)*X + i]);
            }
        }
    }
    else if (gid_y < M/(2*Y)-1) {
        if (gid_x == 0) {
            for (j = 0; j <= Y+T-1; ++j) {
                write_pipe_block(p1, &A_local[j*(X+T)]);
                write_pipe_block(p1, &A_local[j*(X+T) + 1]);
            }
            for (i = 0; i <= X+T-1; ++i) {
                write_pipe_block(p1, &A_local[(Y-2)*(X+T) + i]);
                write_pipe_block(p1, &A_local[(Y-1)*(X+T) + i]);
            }
        }
        else if (gid_x < M/(2*X)-1) {
            for (j = 0; j <= Y+T-1; ++j) {
                write_pipe_block(p1, &A_local[j*(X+T)]);
                write_pipe_block(p1, &A_local[j*(X+T) + 1]);
            }
            for (i = 0; i <= X+T-1; ++i) {
                write_pipe_block(p1, &A_local[(Y-2)*(X+T) + i]);
                write_pipe_block(p1, &A_local[(Y-1)*(X+T) + i]);
            }
        }
        else {
            for (j = 0; j <= Y+T-1; ++j) {
                write_pipe_block(p1, &A_local[j*X]);
                write_pipe_block(p1, &A_local[j*X + 1]);
            }
            for (i = 0; i <= X-1; ++i) {
                write_pipe_block(p1, &A_local[(Y-2)*X + i]);
                write_pipe_block(p1, &A_local[(Y-1)*X + i]);
            }
        }
    }
    else {
        if (gid_x == 0) {
            for (j = 0; j <= Y+T-1; ++j) {
                write_pipe_block(p1, &A_local[j*(X+T)]);
                write_pipe_block(p1, &A_local[j*(X+T) + 1]);
            }
            for (i = 0; i <= X+T-1; ++i) {
                write_pipe_block(p1, &A_local[(Y-2)*(X+T) + i]);
                write_pipe_block(p1, &A_local[(Y-1)*(X+T) + i]);
            }
        }
        else if (gid_x < M/(2*X)-1) {
            for (j = 0; j <= Y+T-1; ++j) {
                write_pipe_block(p1, &A_local[j*(X+T)]);
                write_pipe_block(p1, &A_local[j*(X+T) + 1]);
            }
            for (i = 0; i <= X+T-1; ++i) {
                write_pipe_block(p1, &A_local[(Y-2)*(X+T) + i]);
                write_pipe_block(p1, &A_local[(Y-1)*(X+T) + i]);
            }
        }
        else {
            for (j = 0; j <= Y+T-1; ++j) {
                write_pipe_block(p1, &A_local[j*X]);
                write_pipe_block(p1, &A_local[j*X + 1]);
            }
            for (i = 0; i <= X-1; ++i) {
                write_pipe_block(p1, &A_local[(Y-2)*X + i]);
                write_pipe_block(p1, &A_local[(Y-1)*X + i]);
            }
        }
    }

    int t;
    for (t = 1; t <= T; ++t) { 
        if (gid_y == 0) {
            if (gid_x == 0) { 
                // compute
                for (j = 1; j <= Y-2; ++j) {
                    __attribute__((xcl_pipeline_loop))
                    for (i = 1; i <= X+T-1-t; ++i) {
                        B_local[j*(X+T) + i] = 0.2f * (A_local[j*(X+T) + i] + A_local[j*(X+T) + (i - 1)] + A_local[j*(X+T) + (i + 1)] + A_local[(j+1)*(X+T) + i] + A_local[(j-1)*(X+T) + i]);
                    }
                }
                // read from pipe
                for (j = 1; j <= Y-1; ++j) {
                    read_pipe_block(p2, &A_local[j*(X+T)]);
                }
                for (i = 1; i <= X+T-1-t; ++i) {
                    read_pipe_block(p2, &A_local[(Y-1)*(X+T) + i]);
                }
                // swap
                for (j = 1; j <= Y-2; ++j) {
                    __attribute__((xcl_pipeline_loop))
                    for (i = 1; i <= X+T-1-t; ++i) {
                        A_local[j*(X+T) + i] = B_local[j*(X+T) + i];
                    }
                }
                // write to pipe
                for (j = 0; j <= Y-1; ++j) {
                    write_pipe_block(p1, &A_local[j*(X+T) + 1]);
                }
                for (i = 0; i <= X+T-1-t; ++i) {
                    write_pipe_block(p1, &A_local[(Y-2)*(X+T) + i]);
                }
            }
            else if (gid_x < M/(2*X)-1) {
                // compute
                for (j = 1; j <= Y-2; ++j) {
                    __attribute__((xcl_pipeline_loop))
                    for (i = 1; i <= X+T-1-t; ++i) {
                        B_local[j*(X+T) + i] = 0.2f * (A_local[j*(X+T) + i] + A_local[j*(X+T) + (i - 1)] + A_local[j*(X+T) + (i + 1)] + A_local[(j+1)*(X+T) + i] + A_local[(j-1)*(X+T) + i]);
                    }
                }
                // read from pipe
                for (j = 1; j <= Y-1; ++j) {
                    read_pipe_block(p2, &A_local[j*(X+T)]);
                }
                for (i = 1; i <= X+T-1-t; ++i) {
                    read_pipe_block(p2, &A_local[(Y-1)*(X+T) + i]);
                }
                // swap
                for (j = 1; j <= Y-2; ++j) {
                    __attribute__((xcl_pipeline_loop))
                    for (i = 1; i <= X+T-1-t; ++i) {
                        A_local[j*(X+T) + i] = B_local[j*(X+T) + i];
                    }
                }
                // write to pipe
                for (j = 0; j <= Y-1; ++j) {
                    write_pipe_block(p1, &A_local[j*(X+T) + 1]);
                }
                for (i = 1; i <= X+T-1-t; ++i) {
                    write_pipe_block(p1, &A_local[(Y-2)*(X+T) + i]);
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
                    read_pipe_block(p2, &A_local[j*X]);
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
                for (j = 0; j <= Y-1; ++j) {
                    write_pipe_block(p1, &A_local[j*X + 1]);
                }
                for (i = 0; i <= X-1; ++i) {
                    write_pipe_block(p1, &A_local[(Y-2)*X + i]);
                }
            }
        }
        else if (gid_y < M/(2*Y)-1) {
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
                    read_pipe_block(p2, &A_local[j*(X+T) + X+T-1]);
                }
                for (i = 1; i <= X-1; ++i) {
                    read_pipe_block(p2, &A_local[(Y-1)*X + i]);
                }
                // swap
                for (j = t; j <= Y+T-1; ++j) {
                    __attribute__((xcl_pipeline_loop))
                    for (i = 1; i <= X-1; ++i) {
                        A_local[j*X + i] = B_local[j*X + i];
                    }
                }
                // write to pipe
                for (j = t; j <= Y+T-1; ++j) {
                    write_pipe_block(p1, &A_local[j*X + X-2]);
                }
                for (i = 0; i <= X-1; ++i) {
                    write_pipe_block(p1, &A_local[(Y-2)*X + i]);
                }
            }
            else if (gid_x < M/(2*X)-1) {
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
                    read_pipe_block(p2, &A_local[(Y-1)*X + i]);
                }
                // swap
                for (j = t; j <= Y+T-1; ++j) {
                    __attribute__((xcl_pipeline_loop))
                    for (i = t; i <= X+T-1; ++i) {
                        A_local[j*(X+T) + i] = B_local[j*(X+T) + i];
                    }
                }
                // write to pipe
                for (j = t; j <= Y+T-1; ++j) {
                    write_pipe_block(p1, &A_local[j*X + X-2]);
                }
                for (i = t; i <= X+T-1; ++i) {
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
                    read_pipe_block(p2, &A_local[(Y-1)*X + i]);
                }
                // swap
                for (j = t; j <= Y+T-1; ++j) {
                    __attribute__((xcl_pipeline_loop))
                    for (i = t; i <= X+T-2; ++i) {
                        A_local[j*(X+T) + i] = B_local[j*(X+T) + i];
                    }
                }
                // write to pipe
                for (j = t; j <= Y+T-1; ++j) {
                    write_pipe_block(p1, &A_local[j*X + X-2]);
                }
                for (i = t; i <= X+T-1; ++i) {
                    write_pipe_block(p1, &A_local[(Y-2)*X + i]);
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
                    read_pipe_block(p2, &A_local[j*(X+T) + X+T-1]);
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
                for (i = 0; i <= X-1; ++i) {
                    write_pipe_block(p1, &A_local[(Y-2)*X + i]);
                }
            }
            else if (gid_x < M/(2*X)-1) {
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
                    read_pipe_block(p2, &A_local[(Y-1)*X + i]);
                }
                // swap
                for (j = t; j <= Y+T-1; ++j) {
                    __attribute__((xcl_pipeline_loop))
                    for (i = t; i <= X+T-1; ++i) {
                        A_local[j*(X+T) + i] = B_local[j*(X+T) + i];
                    }
                }
                // write to pipe
                for (j = t; j <= Y+T-1; ++j) {
                    write_pipe_block(p1, &A_local[j*X + X-2]);
                }
                for (i = t; i <= X+T-1; ++i) {
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
                    read_pipe_block(p2, &A_local[(Y-1)*X + i]);
                }
                // swap
                for (j = t; j <= Y+T-1; ++j) {
                    __attribute__((xcl_pipeline_loop))
                    for (i = t; i <= X+T-1; ++i) {
                        A_local[j*(X+T) + i] = B_local[j*(X+T) + i];
                    }
                }
                // write to pipe
                for (j = t; j <= Y+T-1; ++j) {
                    write_pipe_block(p1, &A_local[j*X + X-2]);
                }
                for (i = t; i <= X+T-1; ++i) {
                    write_pipe_block(p1, &A_local[(Y-2)*X + i]);
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
        else if (gid_x < M/(2*X)-1) {
            for (j = 0; j <= Y-1; ++j) {
                async_work_group_copy(&A[(gid_y*2*Y+j)*M + gid_x*2*X], &A_local[(X+T)*j+T], X, 0);
            }
        }
        else {
            for (j = 0; j <= Y-1; ++j) {
                async_work_group_copy(&A[(gid_y*2*Y+j)*M + gid_x*2*X], &A_local[(X+T)*j+T], X, 0);
            }
        }
    }
    else if (gid_y < M/(2*Y)-1) {
        if (gid_x == 0) {
            for (j = 0; j <= Y-1; ++j) {
                async_work_group_copy(&A[(gid_y*2*Y+j)*M + gid_x*2*X], &A_local[X*(j+T)], X, 0);
            }
        }
        else if (gid_x < M/(2*X)-1) {
            for (j = 0; j <= Y-1; ++j) {
                async_work_group_copy(&A[(gid_y*2*Y+j)*M + gid_x*2*X], &A_local[(X+T)*(j+T)+T], X, 0);
            }
        }
        else {
            for (j = 0; j <= Y-1; ++j) {
                async_work_group_copy(&A[(gid_y*2*Y+j)*M + gid_x*2*X], &A_local[(X+T)*(j+T)+T], X, 0);
            }
        }
    }
    else {
        if (gid_x == 0) {
            for (j = 0; j <= Y-1; ++j) {
                async_work_group_copy(&A[(gid_y*2*Y+j)*M + gid_x*2*X], &A_local[X*(j+T)], X, 0);
            }
        }
        else if (gid_x < M/X-1) {
            for (j = 0; j <= Y-1; ++j) {
                async_work_group_copy(&A[(gid_y*2*Y+j)*M + gid_x*2*X], &A_local[(X+T)*(j+T)+T], X, 0);
            }
        }
        else {
            for (j = 0; j <= Y-1; ++j) {
                async_work_group_copy(&A[(gid_y*2*Y+j)*M + gid_x*2*X], &A_local[(X+T)*(j+T)+T], X, 0);
            }
        }
    }
}
