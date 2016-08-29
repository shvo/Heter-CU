XILINX_SDACCEL :=/opt/Xilinx/SDAccel/2016.1
XILINX_OPENCL := $(XILINX_SDACCEL)
DSA := xilinx:adm-pcie-7v3:1ddr:2.1
XOCC := $(XILINX_SDACCEL)/bin/xocc
CLFLAGS := -s --xdevice $(DSA)

all : jacobi1D_heter-ghost_pipeline_part4_N2340_T2048_cu7.xclbin \
      jacobi1D_heter-ghost_pipeline_part4_N2340_T1024_cu7.xclbin \
      jacobi1D_heter-ghost_pipeline_part4_N2340_T512_cu7.xclbin \
      jacobi1D_heter-ghost_pipeline_part4_N2340_T256_cu7.xclbin \
      jacobi1D_heter-ghost_pipeline_part4_N2340_T128_cu7.xclbin \
      jacobi1D_heter-ghost_pipeline_part4_N2340_T64_cu7.xclbin \
      jacobi1D_heter-ghost_pipeline_part4_N2340_T32_cu7.xclbin \
      jacobi1D_heter-ghost_pipeline_part4_N2340_T16_cu7.xclbin \
      jacobi1D_heter-ghost_pipeline_part4_N2340_T8_cu7.xclbin \
      jacobi1D_heter-ghost_pipeline_part4_N2340_T4_cu7.xclbin \
      jacobi1D_heter-ghost_pipeline_part4_N2340_T2_cu7.xclbin \
      jacobi1D_heter-ghost_pipeline_part4_N2340_T1_cu7.xclbin \

.PHONY : all clean

jacobi1D_heter-ghost_pipeline_part4_N2340_T2048_cu7.xclbin: ./jacobi1D_heter-ghost_pipeline_part4_N2340_cu7.cl
	$(XOCC) $(CLFLAGS) -D T=2048 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T2048.xtxt ;\

jacobi1D_heter-ghost_pipeline_part4_N2340_T1024_cu7.xclbin: ./jacobi1D_heter-ghost_pipeline_part4_N2340_cu7.cl
	$(XOCC) $(CLFLAGS) -D T=1024 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T1024.xtxt ;\

jacobi1D_heter-ghost_pipeline_part4_N2340_T512_cu7.xclbin: ./jacobi1D_heter-ghost_pipeline_part4_N2340_cu7.cl
	$(XOCC) $(CLFLAGS) -D T=512 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T512.xtxt ;\

jacobi1D_heter-ghost_pipeline_part4_N2340_T256_cu7.xclbin: ./jacobi1D_heter-ghost_pipeline_part4_N2340_cu7.cl
	$(XOCC) $(CLFLAGS) -D T=256 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T256.xtxt ;\

jacobi1D_heter-ghost_pipeline_part4_N2340_T128_cu7.xclbin: ./jacobi1D_heter-ghost_pipeline_part4_N2340_cu7.cl
	$(XOCC) $(CLFLAGS) -D T=128 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T128.xtxt ;\

jacobi1D_heter-ghost_pipeline_part4_N2340_T64_cu7.xclbin: ./jacobi1D_heter-ghost_pipeline_part4_N2340_cu7.cl
	$(XOCC) $(CLFLAGS) -D T=64 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T64.xtxt ;\

jacobi1D_heter-ghost_pipeline_part4_N2340_T32_cu7.xclbin: ./jacobi1D_heter-ghost_pipeline_part4_N2340_cu7.cl
	$(XOCC) $(CLFLAGS) -D T=32 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T32.xtxt ;\

jacobi1D_heter-ghost_pipeline_part4_N2340_T16_cu7.xclbin: ./jacobi1D_heter-ghost_pipeline_part4_N2340_cu7.cl
	$(XOCC) $(CLFLAGS) -D T=16 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T16.xtxt ;\

jacobi1D_heter-ghost_pipeline_part4_N2340_T8_cu7.xclbin: ./jacobi1D_heter-ghost_pipeline_part4_N2340_cu7.cl
	$(XOCC) $(CLFLAGS) -D T=8 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T8.xtxt ;\

jacobi1D_heter-ghost_pipeline_part4_N2340_T4_cu7.xclbin: ./jacobi1D_heter-ghost_pipeline_part4_N2340_cu7.cl
	$(XOCC) $(CLFLAGS) -D T=4 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T4.xtxt ;\

jacobi1D_heter-ghost_pipeline_part4_N2340_T2_cu7.xclbin: ./jacobi1D_heter-ghost_pipeline_part4_N2340_cu7.cl
	$(XOCC) $(CLFLAGS) -D T=2 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T2.xtxt ;\

jacobi1D_heter-ghost_pipeline_part4_N2340_T1_cu7.xclbin: ./jacobi1D_heter-ghost_pipeline_part4_N2340_cu7.cl
	$(XOCC) $(CLFLAGS) -D T=1 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T1.xtxt ;\

clean:
	rm *.xo
