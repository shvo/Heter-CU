XILINX_SDACCEL :=/opt/Xilinx/SDAccel/2016.1
XILINX_OPENCL := $(XILINX_SDACCEL)
DSA := xilinx:adm-pcie-7v3:1ddr:2.1
XOCC := $(XILINX_SDACCEL)/bin/xocc
CLFLAGS := -s --xdevice $(DSA)

all : jacobi1D_heter-ghost_pipeline_part4_N2048_T2048_cu8.xclbin \
      jacobi1D_heter-ghost_pipeline_part4_N2048_T1024_cu8.xclbin \
      jacobi1D_heter-ghost_pipeline_part4_N2048_T512_cu8.xclbin \
      jacobi1D_heter-ghost_pipeline_part4_N2048_T256_cu8.xclbin \
      jacobi1D_heter-ghost_pipeline_part4_N2048_T128_cu8.xclbin \
      jacobi1D_heter-ghost_pipeline_part4_N2048_T64_cu8.xclbin \
      jacobi1D_heter-ghost_pipeline_part4_N2048_T32_cu8.xclbin \
      jacobi1D_heter-ghost_pipeline_part4_N2048_T16_cu8.xclbin \
      jacobi1D_heter-ghost_pipeline_part4_N2048_T8_cu8.xclbin \
      jacobi1D_heter-ghost_pipeline_part4_N2048_T4_cu8.xclbin \
      jacobi1D_heter-ghost_pipeline_part4_N2048_T2_cu8.xclbin \
      jacobi1D_heter-ghost_pipeline_part4_N2048_T1_cu8.xclbin \

.PHONY : all clean

jacobi1D_heter-ghost_pipeline_part4_N2048_T2048_cu8.xclbin: ./jacobi1D_heter-ghost_pipeline_part4_N2048_cu8.cl
	$(XOCC) $(CLFLAGS) -D T=2048 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T2048.xtxt ;\

jacobi1D_heter-ghost_pipeline_part4_N2048_T1024_cu8.xclbin: ./jacobi1D_heter-ghost_pipeline_part4_N2048_cu8.cl
	$(XOCC) $(CLFLAGS) -D T=1024 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T1024.xtxt ;\

jacobi1D_heter-ghost_pipeline_part4_N2048_T512_cu8.xclbin: ./jacobi1D_heter-ghost_pipeline_part4_N2048_cu8.cl
	$(XOCC) $(CLFLAGS) -D T=512 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T512.xtxt ;\

jacobi1D_heter-ghost_pipeline_part4_N2048_T256_cu8.xclbin: ./jacobi1D_heter-ghost_pipeline_part4_N2048_cu8.cl
	$(XOCC) $(CLFLAGS) -D T=256 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T256.xtxt ;\

jacobi1D_heter-ghost_pipeline_part4_N2048_T128_cu8.xclbin: ./jacobi1D_heter-ghost_pipeline_part4_N2048_cu8.cl
	$(XOCC) $(CLFLAGS) -D T=128 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T128.xtxt ;\

jacobi1D_heter-ghost_pipeline_part4_N2048_T64_cu8.xclbin: ./jacobi1D_heter-ghost_pipeline_part4_N2048_cu8.cl
	$(XOCC) $(CLFLAGS) -D T=64 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T64.xtxt ;\

jacobi1D_heter-ghost_pipeline_part4_N2048_T32_cu8.xclbin: ./jacobi1D_heter-ghost_pipeline_part4_N2048_cu8.cl
	$(XOCC) $(CLFLAGS) -D T=32 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T2048.xtxt ;\

jacobi1D_heter-ghost_pipeline_part4_N2048_T16_cu8.xclbin: ./jacobi1D_heter-ghost_pipeline_part4_N2048_cu8.cl
	$(XOCC) $(CLFLAGS) -D T=16 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T16.xtxt ;\

jacobi1D_heter-ghost_pipeline_part4_N2048_T8_cu8.xclbin: ./jacobi1D_heter-ghost_pipeline_part4_N2048_cu8.cl
	$(XOCC) $(CLFLAGS) -D T=8 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T8.xtxt ;\

jacobi1D_heter-ghost_pipeline_part4_N2048_T4_cu8.xclbin: ./jacobi1D_heter-ghost_pipeline_part4_N2048_cu8.cl
	$(XOCC) $(CLFLAGS) -D T=4 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T4.xtxt ;\

jacobi1D_heter-ghost_pipeline_part4_N2048_T2_cu8.xclbin: ./jacobi1D_heter-ghost_pipeline_part4_N2048_cu8.cl
	$(XOCC) $(CLFLAGS) -D T=2 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T2.xtxt ;\

jacobi1D_heter-ghost_pipeline_part4_N2048_T1_cu8.xclbin: ./jacobi1D_heter-ghost_pipeline_part4_N2048_cu8.cl
	$(XOCC) $(CLFLAGS) -D T=1 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T1.xtxt ;\

clean:
	rm *.xo
