XILINX_SDACCEL :=/opt/Xilinx/SDAccel/2016.1
XILINX_OPENCL := $(XILINX_SDACCEL)
DSA := xilinx:adm-pcie-7v3:1ddr:2.1
XOCC := $(XILINX_SDACCEL)/bin/xocc
CLFLAGS := --xdevice $(DSA)

all : jacobi2D_ghost_128x128_T1_cu3.xclbin \
      jacobi2D_ghost_128x128_T2_cu3.xclbin \
      jacobi2D_ghost_128x128_T4_cu3.xclbin \
      jacobi2D_ghost_128x128_T8_cu3.xclbin \
      jacobi2D_ghost_128x128_T16_cu3.xclbin \
      jacobi2D_ghost_128x128_T32_cu3.xclbin \
      jacobi2D_ghost_128x128_T64_cu3.xclbin \
      jacobi2D_ghost_128x128_T128_cu3.xclbin \

.PHONY : all clean

jacobi2D_ghost_128x128_T128_cu3.xclbin: ./jacobi2D_ghost_128x128.cl
	$(XOCC) $(CLFLAGS) -D T=128 --nk runJacobi2D_kernel1:3 -k runJacobi2D_kernel1 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T128.xtxt

jacobi2D_ghost_128x128_T64_cu3.xclbin: ./jacobi2D_ghost_128x128.cl
	$(XOCC) $(CLFLAGS) -D T=64 --nk runJacobi2D_kernel1:3 -k runJacobi2D_kernel1 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T64.xtxt

jacobi2D_ghost_128x128_T32_cu3.xclbin: ./jacobi2D_ghost_128x128.cl
	$(XOCC) $(CLFLAGS) -D T=32 --nk runJacobi2D_kernel1:3 -k runJacobi2D_kernel1 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T32.xtxt

jacobi2D_ghost_128x128_T16_cu3.xclbin: ./jacobi2D_ghost_128x128.cl
	$(XOCC) $(CLFLAGS) -D T=16 --nk runJacobi2D_kernel1:3 -k runJacobi2D_kernel1 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T16.xtxt

jacobi2D_ghost_128x128_T8_cu3.xclbin: ./jacobi2D_ghost_128x128.cl
	$(XOCC) $(CLFLAGS) -D T=8 --nk runJacobi2D_kernel1:3 -k runJacobi2D_kernel1 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T8.xtxt

jacobi2D_ghost_128x128_T4_cu3.xclbin: ./jacobi2D_ghost_128x128.cl
	$(XOCC) $(CLFLAGS) -D T=4 --nk runJacobi2D_kernel1:3 -k runJacobi2D_kernel1 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T4.xtxt

jacobi2D_ghost_128x128_T2_cu3.xclbin: ./jacobi2D_ghost_128x128.cl
	$(XOCC) $(CLFLAGS) -D T=2 --nk runJacobi2D_kernel1:3 -k runJacobi2D_kernel1 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T2.xtxt

jacobi2D_ghost_128x128_T1_cu3.xclbin: ./jacobi2D_ghost_128x128.cl
	$(XOCC) $(CLFLAGS) -D T=1 --nk runJacobi2D_kernel1:3 -k runJacobi2D_kernel1 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T1.xtxt

clean:
	rm *.xo
