XILINX_SDACCEL :=/opt/Xilinx/SDAccel/2016.1
XILINX_OPENCL := $(XILINX_SDACCEL)
DSA := xilinx:adm-pcie-7v3:1ddr:2.1
XOCC := $(XILINX_SDACCEL)/bin/xocc
CLFLAGS := -s --xdevice $(DSA)

all : jacobi1D_ghost_pipeline_part4_T512_cu1.xclbin \
      jacobi1D_ghost_pipeline_part4_T512_cu2.xclbin \
      jacobi1D_ghost_pipeline_part4_T512_cu3.xclbin \
      jacobi1D_ghost_pipeline_part4_T512_cu4.xclbin \
      jacobi1D_ghost_pipeline_part4_T512_cu5.xclbin \
      jacobi1D_ghost_pipeline_part4_T512_cu6.xclbin \
      jacobi1D_ghost_pipeline_part4_T512_cu7.xclbin \
      jacobi1D_ghost_pipeline_part4_T512_cu8.xclbin \

.PHONY : all clean

jacobi1D_ghost_pipeline_part4_T512_cu1.xclbin: ./jacobi1D_ghost_pipeline_part4_T512.cl
	$(XOCC) $(CLFLAGS) --nk runJacobi1D_kernel1:1 -k runJacobi1D_kernel1 --report system $< -o $@ ;\

jacobi1D_ghost_pipeline_part4_T512_cu2.xclbin: ./jacobi1D_ghost_pipeline_part4_T512.cl
	$(XOCC) $(CLFLAGS) --nk runJacobi1D_kernel1:2 -k runJacobi1D_kernel1 --report system $< -o $@ ;\

jacobi1D_ghost_pipeline_part4_T512_cu3.xclbin: ./jacobi1D_ghost_pipeline_part4_T512.cl
	$(XOCC) $(CLFLAGS) --nk runJacobi1D_kernel1:3 -k runJacobi1D_kernel1 --report system $< -o $@ ;\

jacobi1D_ghost_pipeline_part4_T512_cu4.xclbin: ./jacobi1D_ghost_pipeline_part4_T512.cl
	$(XOCC) $(CLFLAGS) --nk runJacobi1D_kernel1:4 -k runJacobi1D_kernel1 --report system $< -o $@ ;\

jacobi1D_ghost_pipeline_part4_T512_cu5.xclbin: ./jacobi1D_ghost_pipeline_part4_T512.cl
	$(XOCC) $(CLFLAGS) --nk runJacobi1D_kernel1:5 -k runJacobi1D_kernel1 --report system $< -o $@ ;\

jacobi1D_ghost_pipeline_part4_T512_cu6.xclbin: ./jacobi1D_ghost_pipeline_part4_T512.cl
	$(XOCC) $(CLFLAGS) --nk runJacobi1D_kernel1:6 -k runJacobi1D_kernel1 --report system $< -o $@ ;\

jacobi1D_ghost_pipeline_part4_T512_cu7.xclbin: ./jacobi1D_ghost_pipeline_part4_T512.cl
	$(XOCC) $(CLFLAGS) --nk runJacobi1D_kernel1:7 -k runJacobi1D_kernel1 --report system $< -o $@ ;\

jacobi1D_ghost_pipeline_part4_T512_cu8.xclbin: ./jacobi1D_ghost_pipeline_part4_T512.cl
	$(XOCC) $(CLFLAGS) --nk runJacobi1D_kernel1:8 -k runJacobi1D_kernel1 --report system $< -o $@ ;\

clean:
	rm *.xo
