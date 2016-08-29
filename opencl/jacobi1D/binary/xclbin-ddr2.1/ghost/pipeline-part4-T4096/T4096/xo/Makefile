XILINX_SDACCEL :=/opt/Xilinx/SDAccel/2016.1
XILINX_OPENCL := $(XILINX_SDACCEL)
DSA := xilinx:adm-pcie-7v3:1ddr:2.1
XOCC := $(XILINX_SDACCEL)/bin/xocc
CLFLAGS := --xdevice $(DSA)

all : jacobi1D_ghost_pipeline_part4_T4096_cu1.xo \

.PHONY : all clean

jacobi1D_ghost_pipeline_part4_T4096_cu1.xo: ./jacobi1D_ghost_pipeline_part4_T4096.cl
	$(XOCC) $(CLFLAGS) -c --nk runJacobi1D_kernel1:1 -k runJacobi1D_kernel1 --report system $< -o $@ ;\

clean:
	rm *.xo
