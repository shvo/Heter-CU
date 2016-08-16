#define AMB_TEMP 80.0
#define NX 512
#define NY 512
#define NZ 8
#define XY 512*512
__kernel __attribute__ ((reqd_work_group_size(1,1,1)))
void hotspot3D(__global float *p, 
                 __global float* tIn,
                 __global float *tOut, 
                float sdc,float ce, 
                float cw, float cn, 
                float cs, float ct,
                float cb, float cc) {

    __local float p_linebuf0[NY*NZ] __attribute__((xcl_array_partition(cyclic,64,1)));
    __local float tIn_linebuf0[NY*NZ] __attribute__((xcl_array_partition(cyclic,64,1)));
    __local float tIn_linebuf1[NY*NZ] __attribute__((xcl_array_partition(cyclic,64,1)));
    __local float tIn_linebuf2[NY*NZ] __attribute__((xcl_array_partition(cyclic,64,1)));
    __local float tOut_linebuf[NY*NZ] __attribute__((xcl_array_partition(cyclic,64,1)));

    int col = get_global_id(0);

    for (int layer = 0; layer < NZ; ++layer) {
        async_work_group_copy(&p_linebuf0[layer*NY], &p[col*NY + layer*NX*NY], NY, 0);
        async_work_group_copy(&tIn_linebuf0[layer*NY], &tIn[col*NY + layer*NX*NY], NY, 0);
        async_work_group_copy(&tIn_linebuf1[layer*NY], &tIn[(col+1)*NY + layer*NX*NY], NY, 0);
        async_work_group_copy(&tIn_linebuf2[layer*NY], &tIn[(col+2)*NY + layer*NX*NY], NY, 0);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int layer = 1; layer < NZ-1; ++layer) {

        __attribute__((xcl_pipeline_loop))
        for (int row = 1; row < NY-1; ++row) {

            float temp_p = p[NY*layer + row];
            float temp2_c = tIn_linebuf1[NY*layer + row]; 
            float temp2_w = tIn_linebuf1[NY*layer + row - 1];
            float temp2_e = tIn_linebuf1[NY*layer + row + 1];
            float temp2_n = tIn_linebuf0[NY*layer + row];
            float temp2_s = tIn_linebuf2[NY*layer + row];
            float temp1_c = tIn_linebuf1[NY*(layer-1) + row];
            float temp3_c = tIn_linebuf1[NY*(layer+1) + row];

            tOut_linebuf[NY*layer + row] = cc * temp2_c + cw * temp2_w + ce * temp2_e + cs * temp2_s
               + cn * temp2_n + cb * temp1_c + ct * temp3_c + sdc * temp_p + ct * AMB_TEMP;

        }

    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int layer = 1; layer < NZ-1; ++layer) {
        async_work_group_copy(&tOut[(col+1)*NY + layer*NX*NY + 1], &tOut_linebuf[layer*NY + 1], NY-2, 0);
    }
    return;
}
