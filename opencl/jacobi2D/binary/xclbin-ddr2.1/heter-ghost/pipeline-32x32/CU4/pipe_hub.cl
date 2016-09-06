__kernel __attribute__ ((reqd_work_group_size(1,1,1)))
void runJacobi1D_connect_1_4(__global DATA_TYPE* A)
{
    float h1[2*(X+T)], h2[2*(X+T)], h2_buf[2*(X+T)], h3[2*(X+T)], h3_buf[2*(X+T)], h4[2*(X+T)];    
    float v1[2*(X+T)], v2[2*(X+T)], v2_buf[2*(X+T)], v3[2*(X+T)], v3_buf[2*(X+T)], v4[2*(X+T)];    

    int gid_x = get_group_id(0);
    int gid_y = get_group_id(1);
  
    int i, j;

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
        for (j = 0; j <= 2*Y+T-1; ++j) {
            async_work_group_copy(v1, &A[(gid_y*2*Y)*M + gid_x*2*X+X-2], 2*Y+T, 0);
            async_work_group_copy(v2_buf, &A[(gid_y*2*Y)*M + gid_x*2*X+X-1], 2*Y+T, 0);
            async_work_group_copy(v3_buf, &A[(gid_y*2*Y)*M + gid_x*2*X+X], 2*Y+T, 0);
            async_work_group_copy(v4, &A[(gid_y*2*Y)*M + gid_x*2*X+X+1], 2*Y+T, 0);
        }
    }
    else if (gid_y < M/(2*Y)-1) {
        for (j = 0; j <= 2*(Y+T)-1; ++j) {
            async_work_group_copy(v1, &A[(gid_y*2*Y-T)*M + gid_x*2*X+X-2], 2*(Y+T), 0);
            async_work_group_copy(v2_buf, &A[(gid_y*2*Y-T)*M + gid_x*2*X+X-1], 2*(Y+T), 0);
            async_work_group_copy(v3_buf, &A[(gid_y*2*Y-T)*M + gid_x*2*X+X], 2*(Y+T), 0);
            async_work_group_copy(v4, &A[(gid_y*2*Y-T)*M + gid_x*2*X+X+1], 2*(Y+T), 0);
        }
    }
    else {
        for (j = 0; j <= 2*Y+T-1; ++j) {
            async_work_group_copy(v1, &A[(gid_y*2*Y-T)*M + gid_x*2*X+X-2], 2*Y+T, 0);
            async_work_group_copy(v2_buf, &A[(gid_y*2*Y-T)*M + gid_x*2*X+X-1], 2*Y+T, 0);
            async_work_group_copy(v3_buf, &A[(gid_y*2*Y-T)*M + gid_x*2*X+X], 2*Y+T, 0);
            async_work_group_copy(v4, &A[(gid_y*2*Y-T)*M + gid_x*2*X+X+1], 2*Y+T, 0);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int t;

    for (t = 1; t <= T; ++t) { 
        // calculate
        if (gid_y == 0) {
            for (j = 1; j <= 2*Y+T-1-t; ++j) {
                v2[j] = 0.2f * (v2_buf[j-1] + v2_buf[j] + v2_buf[j+1] + v1[j] + v3_buf[j]);
                v3[j] = 0.2f * (v3_buf[j-1] + v3_buf[j] + v3_buf[j+1] + v2_buf[j] + v4[j]);
            }
        }
        else if (gid_y < M/(2*Y)-1) {
            for (j = t; j <= 2*(Y+T)-1-t; ++j) {
                v2[j] = 0.2f * (v2_buf[j-1] + v2_buf[j] + v2_buf[j+1] + v1[j] + v3_buf[j]);
                v3[j] = 0.2f * (v3_buf[j-1] + v3_buf[j] + v3_buf[j+1] + v2_buf[j] + v4[j]);
            }
        }
        else {
            for (j = 1; j <= 2*Y+T-2; ++j) {
                v2[j] = 0.2f * (v2_buf[j-1] + v2_buf[j] + v2_buf[j+1] + v1[j] + v3_buf[j]);
                v3[j] = 0.2f * (v3_buf[j-1] + v3_buf[j] + v3_buf[j+1] + v2_buf[j] + v4[j]);
            }
        }
        
        if (gid_x == 0) {
            for (i = 1; i <= 2*X+T-1-t; ++i) {
                h2[i] = 0.2f * (h2_buf[i-1] + h2_buf[i] + h2_buf[i+1] + h1[i] + h3_buf[i]);
                h3[i] = 0.2f * (h3_buf[i-1] + h3_buf[i] + h3_buf[i+1] + h2_buf[i] + h4[i]);
            }
        }
        else if (gid_x < M/(2*X)-1) {
            for (i = t; i <= 2*(X+T)-1-t; ++i) {
                h2[i] = 0.2f * (h2_buf[i-1] + h2_buf[i] + h2_buf[i+1] + h1[i] + h3_buf[i]);
                h3[i] = 0.2f * (h3_buf[i-1] + h3_buf[i] + h3_buf[i+1] + h2_buf[i] + h4[i]);
            }
        }
        else {
            for (i = t; i <= 2*X+T-2; ++i) {
                h2[i] = 0.2f * (h2_buf[i-1] + h2_buf[i] + h2_buf[i+1] + h1[i] + h3_buf[i]);
                h3[i] = 0.2f * (h3_buf[i-1] + h3_buf[i] + h3_buf[i+1] + h2_buf[i] + h4[i]);
        }

        // communicate with compute kernels
          // vertical data
        if (gid_y == 0) {
            for (j = 1; j <= Y-1; ++j) {
                write_pipe_block(p2, &v2[j]);
                write_pipe_block(p4, &v3[j]);
            }
            for (j = 0; j <= Y+T-1-t; ++j) {
                write_pipe_block(p6, &v2[j+Y]);
                write_pipe_block(p8, &v3[j+Y]);
            }
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
                write_pipe_block(p2, &v2[j]);
                write_pipe_block(p4, &v3[j]);
            }
            for (j = 0; j <= Y+T-1-t; ++j) {
                write_pipe_block(p6, &v2[j+Y+T]);
                write_pipe_block(p8, &v3[j+Y+T]);
            }
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
                write_pipe_block(p2, &v2[j]);
                write_pipe_block(p4, &v3[j]);
            }
            for (j = 0; j <= Y-2; ++j) {
                write_pipe_block(p6, &v2[j+Y+T]);
                write_pipe_block(p8, &v3[j+Y+T]);
            }
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
                write_pipe_block(p2, &h2[i]);
                write_pipe_block(p6, &h3[i]);
            }
            for (i = 0; i <= X+T-1-t; ++i) {
                write_pipe_block(p4, &h2[i+X]);
                write_pipe_block(p8, &h3[i+X]);
            }
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
                write_pipe_block(p2, &h2[i]);
                write_pipe_block(p6, &h3[i]);
            }
            for (i = 0; i <= X+T-1-t; ++i) {
                write_pipe_block(p4, &h2[i+X+T]);
                write_pipe_block(p8, &h3[i+X+T]);
            }
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
                write_pipe_block(p2, &h2[i]);
                write_pipe_block(p6, &h3[i]);
            }
            for (i = 0; i <= X-2; ++i) {
                write_pipe_block(p4, &h2[i+X+T]);
                write_pipe_block(p8, &h3[i+X+T]);
            }
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
