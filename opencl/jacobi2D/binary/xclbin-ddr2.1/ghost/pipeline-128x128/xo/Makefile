XILINX_SDACCEL :=/opt/Xilinx/SDAccel/2016.1
XILINX_OPENCL := $(XILINX_SDACCEL)
DSA := xilinx:adm-pcie-7v3:1ddr:2.1
XOCC := $(XILINX_SDACCEL)/bin/xocc
CLFLAGS := --xdevice $(DSA)

all : jacobi2D_gpu_ghost.xo \

.PHONY : all clean

jacobi2D_gpu_ghost.xo: ./jacobi2D_gpu_ghost.cl
	$(XOCC) $(CLFLAGS) -c -D T=16 --nk runJacobi2D_kernel1:1 -k runJacobi2D_kernel1 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T16.xtxt
clean:
	rm *.xo
