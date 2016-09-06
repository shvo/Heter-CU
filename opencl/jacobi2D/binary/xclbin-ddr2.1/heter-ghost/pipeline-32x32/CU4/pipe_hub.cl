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
        for (j = t; j <= 2*(X+T)-1-t; ++j) {
            v2[j] = 0.2f * (v2_buf[j-1] + v2_buf[j] + v2_buf[j+1] + v1[j] + v3_buf[j]);
            v3[j] = 0.2f * (v3_buf[j-1] + v3_buf[j] + v3_buf[j+1] + v2_buf[j] + v4[j]);
        }
        for (i = t; i <= 2*(X+T)-1-t; ++i) {
            h2[i] = 0.2f * (h2_buf[i-1] + h2_buf[i] + h2_buf[i+1] + h1[i] + h3_buf[i]);
            h3[i] = 0.2f * (h3_buf[i-1] + h3_buf[i] + h3_buf[i+1] + h2_buf[i] + h4[i]);
        }

    // write to kernels
        for (j = t; j <= 2*(X+T)-1-t; ++j) {
        }

    // read from kernels
    }
}
