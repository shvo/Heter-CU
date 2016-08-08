#define AMB_TEMP 80.0
#define NX 512
#define NY 512
#define NZ 8
#define XY 512*512
__kernel __attribute__ ((reqd_work_group_size(1,1,1)))
void hotspotOpt1(__global float *p, 
                 __global float* tIn,
                 __global float *tOut, 
                float sdc,float ce, 
                float cw, float cn, 
                float cs, float ct,
                float cb, float cc) {

    __local float p_linebuf0[NY*NZ];
    __local float tIn_linebuf0[NY*NZ], tIn_linebuf1[NY*NZ], tIn_linebuf2[NY*NZ];
    __local float tOut_linebuf0[NY*NZ];

    int col = get_global_id(0);

    for (int layer = 0; layer < NZ; ++layer) {
        async_work_group_copy(&p_linebuf0[layer*NY], &p[col*NY + layer*NX*NY], NY, 0);
        async_work_group_copy(&tIn_linebuf0[layer*NY], &tIn[col*NY + layer*NX*NY], NY, 0);
        async_work_group_copy(&tIn_linebuf1[layer*NY], &tIn[(col+1)*NY + layer*NX*NY], NY, 0);
        async_work_group_copy(&tIn_linebuf2[layer*NY], &tIn[(col+2)*NY + layer*NX*NY], NY, 0);
    }

    for (int row = 1; row < NY-1; ++row) {

        float temp2_c = tIn_linebuf1[row]; 
        float temp2_W = tIn_linebuf1[row - 1];
        float temp2_E = tIn_linebuf1[row + 1];
        float temp2_N = tIn_linebuf0[row];
        float temp2_S = tIn_linebuf2[row];
        float temp1_c = tIn_linebuf1[row];;
        float temp3_c = tIn_linebuf1[row];;

        for (int layer = 1; k < NZ-1; ++k) {
            tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S]
               + cn * tIn[N] + cb * temp1 + ct * temp3 + sdc * p[c] + ct * AMB_TEMP;
        }
    }



  float temp1, temp2, temp3;
  temp1 = temp2 = tIn[c];
  temp3 = tIn[c+xy];
  tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S]
    + cn * tIn[N] + cb * temp1 + ct * temp3 + sdc * p[c] + ct * AMB_TEMP;

  for (int k = 1; k < NZ-1; ++k) {
      temp1 = temp2;
      temp2 = temp3;
      temp3 = tIn[c+xy];
     
      tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S]
        + cn * tIn[N] + cb * temp1 + ct * temp3 + sdc * p[c] + ct * AMB_TEMP;
      
      c += xy;
      W += xy;
      E += xy;
      N += xy;
      S += xy;
  }
  return;
}
