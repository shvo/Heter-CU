XILINX_SDACCEL :=/opt/Xilinx/SDAccel/2016.1
XILINX_OPENCL := $(XILINX_SDACCEL)
DSA := xilinx:adm-pcie-7v3:1ddr:2.1
XOCC := $(XILINX_SDACCEL)/bin/xocc
CLFLAGS := --xdevice $(DSA)

all : jacobi1D_ghost_pipeline_part4_N4096_T8_cu4.xclbin \
      jacobi1D_ghost_pipeline_part4_N4096_T4_cu4.xclbin \
      jacobi1D_ghost_pipeline_part4_N4096_T2_cu4.xclbin \
      jacobi1D_ghost_pipeline_part4_N4096_T1_cu4.xclbin \

.PHONY : all clean

jacobi1D_ghost_pipeline_part4_N4096_T8_cu4.xclbin: ./jacobi1D_ghost_pipeline_part4_N4096.cl
	$(XOCC) $(CLFLAGS) -D T=8 --nk runJacobi1D_kernel1:4 -k runJacobi1D_kernel1 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T8.xtxt

jacobi1D_ghost_pipeline_part4_N4096_T4_cu4.xclbin: ./jacobi1D_ghost_pipeline_part4_N4096.cl
	$(XOCC) $(CLFLAGS) -D T=4 --nk runJacobi1D_kernel1:4 -k runJacobi1D_kernel1 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T4.xtxt

jacobi1D_ghost_pipeline_part4_N4096_T2_cu4.xclbin: ./jacobi1D_ghost_pipeline_part4_N4096.cl
	$(XOCC) $(CLFLAGS) -D T=2 --nk runJacobi1D_kernel1:4 -k runJacobi1D_kernel1 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T2.xtxt

jacobi1D_ghost_pipeline_part4_N4096_T1_cu4.xclbin: ./jacobi1D_ghost_pipeline_part4_N4096.cl
	$(XOCC) $(CLFLAGS) -D T=1 --nk runJacobi1D_kernel1:4 -k runJacobi1D_kernel1 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T1.xtxt

clean:
	rm *.xo
