XILINX_SDACCEL :=/opt/Xilinx/SDAccel/2016.1
XILINX_OPENCL := $(XILINX_SDACCEL)
DSA := xilinx:adm-pcie-7v3:1ddr:3.0
XOCC := $(XILINX_SDACCEL)/bin/xocc
CLFLAGS := -s --xdevice $(DSA)

all : hotspot3D_regular.xo \

.PHONY : all clean

hotspot3D_regular.xo: ./hotspot3D_regular_fpga.cl
	$(XOCC) $(CLFLAGS) --nk hotspot3D:1 -k hotspot3D -c --report estimate $< -o $@ ;\

clean:
	rm *.xo
