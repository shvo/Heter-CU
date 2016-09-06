__kernel __attribute__ ((reqd_work_group_size(1,1,1)))
void runJacobi1D_connect_1_4(__global DATA_TYPE* A)
{
    float h1[2*(X+T)], h2[2*(X+T)], h3[2*(X+T)];    
    float v1[2*(X+T)], v2[2*(X+T)], v3[2*(X+T)];    

    int gid_x = get_group_id(0);
    int gid_y = get_group_id(1);
  
    int i, j;

    // initialization
    if (gid_x == 0) {
        async_work_group_copy(h1, &A[(gid_y*2*Y+Y-2)*M + gid_x*2*X], 2*X+T, 0);
        async_work_group_copy(h2, &A[(gid_y*2*Y+Y-1)*M + gid_x*2*X], 2*X+T, 0);
        async_work_group_copy(h3, &A[(gid_y*2*Y+Y)*M + gid_x*2*X], 2*X+T, 0);
    }
    else if (gid_x < M/(2*X)-1) {
        async_work_group_copy(h1, &A[(gid_y*2*Y+Y-2)*M + gid_x*2*X-T], 2*(X+T), 0);
        async_work_group_copy(h2, &A[(gid_y*2*Y+Y-1)*M + gid_x*2*X-T], 2*(X+T), 0);
        async_work_group_copy(h3, &A[(gid_y*2*Y+Y)*M + gid_x*2*X-T], 2*(X+T), 0);
    }
    else {
        async_work_group_copy(h1, &A[(gid_y*2*Y+Y-2)*M + gid_x*2*X-T], 2*X+T, 0);
        async_work_group_copy(h2, &A[(gid_y*2*Y+Y-1)*M + gid_x*2*X-T], 2*X+T, 0);
        async_work_group_copy(h3, &A[(gid_y*2*Y+Y)*M + gid_x*2*X-T], 2*X+T, 0);
    }

    if (gid_y == 0) {
        for (j = 0; j <= 2*Y+T-1; ++j) {
            async_work_group_copy(v1, &A[(gid_y*2*Y)*M + gid_x*2*X+X-2], 2*Y+T, 0);
            async_work_group_copy(v2, &A[(gid_y*2*Y)*M + gid_x*2*X+X-1], 2*Y+T, 0);
            async_work_group_copy(v3, &A[(gid_y*2*Y)*M + gid_x*2*X+X], 2*Y+T, 0);
        }
    }
    else if (gid_y < M/(2*Y)-1) {
        for (j = 0; j <= 2*(Y+T)-1; ++j) {
            async_work_group_copy(v1, &A[(gid_y*2*Y-T)*M + gid_x*2*X+X-2], 2*(Y+T), 0);
            async_work_group_copy(v2, &A[(gid_y*2*Y-T)*M + gid_x*2*X+X-1], 2*(Y+T), 0);
            async_work_group_copy(v3, &A[(gid_y*2*Y-T)*M + gid_x*2*X+X], 2*(Y+T), 0);
        }
    }
    else {
        for (j = 0; j <= 2*Y+T-1; ++j) {
            async_work_group_copy(v1, &A[(gid_y*2*Y-T)*M + gid_x*2*X+X-2], 2*Y+T, 0);
            async_work_group_copy(v2, &A[(gid_y*2*Y-T)*M + gid_x*2*X+X-1], 2*Y+T, 0);
            async_work_group_copy(v3, &A[(gid_y*2*Y-T)*M + gid_x*2*X+X], 2*Y+T, 0);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
      
}
