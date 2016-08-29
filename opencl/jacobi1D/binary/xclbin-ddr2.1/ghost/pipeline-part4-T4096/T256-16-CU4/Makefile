XILINX_SDACCEL :=/opt/Xilinx/SDAccel/2016.1
XILINX_OPENCL := $(XILINX_SDACCEL)
DSA := xilinx:adm-pcie-7v3:1ddr:2.1
XOCC := $(XILINX_SDACCEL)/bin/xocc
CLFLAGS := --xdevice $(DSA)

all : jacobi1D_ghost_pipeline_part4_N4096_T256_cu4.xclbin \
      jacobi1D_ghost_pipeline_part4_N4096_T128_cu4.xclbin \
      jacobi1D_ghost_pipeline_part4_N4096_T64_cu4.xclbin \
      jacobi1D_ghost_pipeline_part4_N4096_T32_cu4.xclbin \
      jacobi1D_ghost_pipeline_part4_N4096_T16_cu4.xclbin \

.PHONY : all clean

jacobi1D_ghost_pipeline_part4_N4096_T256_cu4.xclbin: ./jacobi1D_ghost_pipeline_part4_N4096.cl
	$(XOCC) $(CLFLAGS) -D T=256 --nk runJacobi1D_kernel1:4 -k runJacobi1D_kernel1 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T256.xtxt

jacobi1D_ghost_pipeline_part4_N4096_T128_cu4.xclbin: ./jacobi1D_ghost_pipeline_part4_N4096.cl
	$(XOCC) $(CLFLAGS) -D T=128 --nk runJacobi1D_kernel1:4 -k runJacobi1D_kernel1 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T128.xtxt

jacobi1D_ghost_pipeline_part4_N4096_T64_cu4.xclbin: ./jacobi1D_ghost_pipeline_part4_N4096.cl
	$(XOCC) $(CLFLAGS) -D T=64 --nk runJacobi1D_kernel1:4 -k runJacobi1D_kernel1 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T64.xtxt

jacobi1D_ghost_pipeline_part4_N4096_T32_cu4.xclbin: ./jacobi1D_ghost_pipeline_part4_N4096.cl
	$(XOCC) $(CLFLAGS) -D T=32 --nk runJacobi1D_kernel1:4 -k runJacobi1D_kernel1 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T32.xtxt

jacobi1D_ghost_pipeline_part4_N4096_T16_cu4.xclbin: ./jacobi1D_ghost_pipeline_part4_N4096.cl
	$(XOCC) $(CLFLAGS) -D T=16 --nk runJacobi1D_kernel1:4 -k runJacobi1D_kernel1 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T16.xtxt
clean:
	rm *.xo
