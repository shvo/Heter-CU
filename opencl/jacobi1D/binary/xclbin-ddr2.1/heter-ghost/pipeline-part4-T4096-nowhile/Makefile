XILINX_SDACCEL :=/opt/Xilinx/SDAccel/2016.1
XILINX_OPENCL := $(XILINX_SDACCEL)
DSA := xilinx:adm-pcie-7v3:1ddr:2.1
XOCC := $(XILINX_SDACCEL)/bin/xocc
CLFLAGS := -s --xdevice $(DSA)

all : jacobi1D_heter-ghost_pipeline_part4_T4096-nowhile.xclbin \

.PHONY : all clean

jacobi1D_heter-ghost_pipeline_part4_T4096-nowhile.xclbin: ./jacobi1D_heter-ghost_pipeline_part4_T4096-nowhile.cl
	$(XOCC) $(CLFLAGS) --report system $< -o $@ ;\

clean:
	rm *.xo
