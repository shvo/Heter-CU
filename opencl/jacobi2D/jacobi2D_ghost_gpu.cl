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
#define T 16   // # merged iterations


__kernel __attribute__ ((reqd_work_group_size(1,1,1)))
void runJacobi2D_kernel1(__global DATA_TYPE* A, __global DATA_TYPE* B, int n)
{
    __local DATA_TYPE A_local[(X+2*T)*(Y+2*T)];
    __local DATA_TYPE B_local[(X+2*T)*(Y+2*T)];
    int gid_x = get_group_id(0);
    int gid_y = get_group_id(1);

    int i,j;
    if (gid_y == 0) {
        if (gid_x == 0) {
            for (i = 0; i < Y + T; ++i) {
                async_work_group_copy(&A_local[(X+T)*i], &A[(gid_y*Y+i)*M + gid_x*X], X+T, 0);
            }
        }
        else if (gid_x < M/X-1) {
            for (i = 0; i < Y + T; ++i) {
                async_work_group_copy(&A_local[(X+2*T)*i], &A[(gid_y*Y+i)*M + gid_x*X-T], X+2*T, 0);
            }
        }
        else {
            for (i = 0; i < Y + T; ++i) {
                async_work_group_copy(&A_local[(X+T)*i], &A[(gid_y*Y+i)*M + gid_x*X-T], X+T, 0);
        }
    }
    else if (gid_y > 0 && gid_y < M/Y-1) {
        if (gid_x == 0) {
            for (i = 0; i < Y + 2*T; ++i) {
                async_work_group_copy(&A_local[(X+T)*i], &A[(gid_y*Y+i-T)*M + gid_x*X], X+T, 0);
            }
        }
        else if (gid_x < M/X-1) {
            for (i = 0; i < Y + 2*T; ++i) {
                async_work_group_copy(&A_local[(X+2*T)*i], &A[(gid_y*Y+i-T)*M + gid_x*X-T], X+2*T, 0);
            }
        }
        else {
            for (i = 0; i < Y + 2*T; ++i) {
                async_work_group_copy(&A_local[(X+T)*i], &A[(gid_y*Y+i-T)*M + gid_x*X-T], X+T, 0);
        }
    }
    else {
        if (gid_x == 0) {
            for (i = 0; i < Y + T; ++i) {
                async_work_group_copy(&A_local[(X+T)*i], &A[(gid_y*Y+i-T)*M + gid_x*X], X+T, 0);
            }
        }
        else if (gid_x < M/X-1) {
            for (i = 0; i < Y + T; ++i) {
                async_work_group_copy(&A_local[(X+2*T)*i], &A[(gid_y*Y+i-T)*M + gid_x*X-T], X+2*T, 0);
        }
        else {
            for (i = 0; i < Y + T; ++i) {
                async_work_group_copy(&A_local[(X+T)*i], &A[(gid_y*Y+i-T)*M + gid_x*X-T], X+T, 0);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int t;
    for (t = 1; t <= T; ++t) { 
        if (gid_y == 0) {
            if (gid_x == 0) {
                for (j = 1; j <= Y+T-1-t; ++j) {
                    for (i = 1; i <= X+T-1-t; ++i) {
                        B_local[j*(X+T) + i] = 0.2f * (A_local[j*(X+T) + i] + A_local[j*(X+T) + (i - 1)] + A_local[j*(X+T) + (i + 1)] + A_local[(j+1)*(X+T) + i] + A_local[(j-1)*(X+T) + i]);
                    }
                }
            }
            else if (gid_x < M/X-1) {
                for (j = 1; j <= Y+T-1-t; ++j) {
                    for (i = t; i <= X+2*T-1-t; ++i) {
                        B_local[j*(X+2*T) + i] = 0.2f * (A_local[j*(X+2*T) + i] + A_local[j*(X+2*T) + (i - 1)] + A_local[j*(X+2*T) + (i + 1)] + A_local[(j+1)*(X+2*T) + i] + A_local[(j-1)*(X+2*T) + i]);
                    }
                }
            }
            else {
                for (j = 1; j <= Y+T-1-t; ++j) {
                    for (i = t; i <= X+T-2; ++i) {
                        B_local[j*(X+T) + i] = 0.2f * (A_local[j*(X+T) + i] + A_local[j*(X+T) + (i - 1)] + A_local[j*(X+T) + (i + 1)] + A_local[(j+1)*(X+T) + i] + A_local[(j-1)*(X+T) + i]);
                    }
                }
            }
        }
        else if (gid_y > 0 && gid_y < M/Y-1) {
            if (gid_x == 0) {
                for ( j = t; j <= Y+2*T-1-t; ++j) {
                    for (i = 1; i <= X+T-1-t; ++i) {
                        B_local[j*(X+T) + i] = 0.2f * (A_local[j*(X+T) + i] + A_local[j*(X+T) + (i - 1)] + A_local[j*(X+T) + (i + 1)] + A_local[(j+1)*(X+T) + i] + A_local[(j-1)*(X+T) + i]);
                    }
                }
            }
            else if (gid_x < M/X-1) {
                for ( j = t; j <= Y+2*T-1-t; ++j) {
                    for (i = t; i <= X+2*T-1-t; ++i) {
                        B_local[j*(X+2*T) + i] = 0.2f * (A_local[j*(X+2*T) + i] + A_local[j*(X+2*T) + (i - 1)] + A_local[j*(X+2*T) + (i + 1)] + A_local[(j+1)*(X+2*T) + i] + A_local[(j-1)*(X+2*T) + i]);
                    }
                }
            }
            else {
                for ( j = t; j <= Y+2*T-1-t; ++j) {
                    for (i = t; i <= X+T-2; ++i) {
                        B_local[j*(X+T) + i] = 0.2f * (A_local[j*(X+T) + i] + A_local[j*(X+T) + (i - 1)] + A_local[j*(X+T) + (i + 1)] + A_local[(j+1)*(X+T) + i] + A_local[(j-1)*(X+T) + i]);
                    }
                }
            }
        }
        else {
            if (gid_x == 0) {
                for ( j = t; j <= Y+T-2; ++j) {
                    for (i = 1; i <= X+T-1-t; ++i) {
                        B_local[j*(X+T) + i] = 0.2f * (A_local[j*(X+T) + i] + A_local[j*(X+T) + (i - 1)] + A_local[j*(X+T) + (i + 1)] + A_local[(j+1)*(X+T) + i] + A_local[(j-1)*(X+T) + i]);
                    }
                }
            }
            else if (gid_x < M/X-1) {
                for ( j = t; j <= Y+T-2; ++j) {
                    for (i = t; i <= X+2*T-1-t; ++i) {
                        B_local[j*(X+2*T) + i] = 0.2f * (A_local[j*(X+2*T) + i] + A_local[j*(X+2*T) + (i - 1)] + A_local[j*(X+2*T) + (i + 1)] + A_local[(j+1)*(X+2*T) + i] + A_local[(j-1)*(X+2*T) + i]);
                    }
                }
            }
            else {
                for ( j = t; j <= Y+T-2; ++j) {
                    for (i = t; i <= X+T-2; ++i) {
                        B_local[j*(X+T) + i] = 0.2f * (A_local[j*(X+T) + i] + A_local[j*(X+T) + (i - 1)] + A_local[j*(X+T) + (i + 1)] + A_local[(j+1)*(X+T) + i] + A_local[(j-1)*(X+T) + i]);
                    }
                }
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

}
