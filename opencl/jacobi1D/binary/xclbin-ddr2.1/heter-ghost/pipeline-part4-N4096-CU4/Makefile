XILINX_SDACCEL :=/opt/Xilinx/SDAccel/2016.1
XILINX_OPENCL := $(XILINX_SDACCEL)
DSA := xilinx:adm-pcie-7v3:1ddr:2.1
XOCC := $(XILINX_SDACCEL)/bin/xocc
CLFLAGS := -s --xdevice $(DSA)

all : jacobi1D_heter-ghost_pipeline_part4_N4096_T4096_cu4.xclbin \
      jacobi1D_heter-ghost_pipeline_part4_N4096_T2048_cu4.xclbin \
      jacobi1D_heter-ghost_pipeline_part4_N4096_T1024_cu4.xclbin \
      jacobi1D_heter-ghost_pipeline_part4_N4096_T512_cu4.xclbin \
      jacobi1D_heter-ghost_pipeline_part4_N4096_T256_cu4.xclbin \
      jacobi1D_heter-ghost_pipeline_part4_N4096_T128_cu4.xclbin \
      jacobi1D_heter-ghost_pipeline_part4_N4096_T64_cu4.xclbin \
      jacobi1D_heter-ghost_pipeline_part4_N4096_T32_cu4.xclbin \
      jacobi1D_heter-ghost_pipeline_part4_N4096_T16_cu4.xclbin \
      jacobi1D_heter-ghost_pipeline_part4_N4096_T8_cu4.xclbin \
      jacobi1D_heter-ghost_pipeline_part4_N4096_T4_cu4.xclbin \
      jacobi1D_heter-ghost_pipeline_part4_N4096_T2_cu4.xclbin \
      jacobi1D_heter-ghost_pipeline_part4_N4096_T1_cu4.xclbin \

.PHONY : all clean

jacobi1D_heter-ghost_pipeline_part4_N4096_T4096_cu4.xclbin: ./jacobi1D_heter-ghost_pipeline_part4_N4096_cu4.cl
	$(XOCC) $(CLFLAGS) -D T=4096 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T4096.xtxt ;\

jacobi1D_heter-ghost_pipeline_part4_N4096_T2048_cu4.xclbin: ./jacobi1D_heter-ghost_pipeline_part4_N4096_cu4.cl
	$(XOCC) $(CLFLAGS) -D T=2048 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T2048.xtxt ;\

jacobi1D_heter-ghost_pipeline_part4_N4096_T1024_cu4.xclbin: ./jacobi1D_heter-ghost_pipeline_part4_N4096_cu4.cl
	$(XOCC) $(CLFLAGS) -D T=1024 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T1024.xtxt ;\

jacobi1D_heter-ghost_pipeline_part4_N4096_T512_cu4.xclbin: ./jacobi1D_heter-ghost_pipeline_part4_N4096_cu4.cl
	$(XOCC) $(CLFLAGS) -D T=512 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T512.xtxt ;\

jacobi1D_heter-ghost_pipeline_part4_N4096_T256_cu4.xclbin: ./jacobi1D_heter-ghost_pipeline_part4_N4096_cu4.cl
	$(XOCC) $(CLFLAGS) -D T=256 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T256.xtxt ;\

jacobi1D_heter-ghost_pipeline_part4_N4096_T128_cu4.xclbin: ./jacobi1D_heter-ghost_pipeline_part4_N4096_cu4.cl
	$(XOCC) $(CLFLAGS) -D T=128 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T128.xtxt ;\

jacobi1D_heter-ghost_pipeline_part4_N4096_T64_cu4.xclbin: ./jacobi1D_heter-ghost_pipeline_part4_N4096_cu4.cl
	$(XOCC) $(CLFLAGS) -D T=64 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T64.xtxt ;\

jacobi1D_heter-ghost_pipeline_part4_N4096_T32_cu4.xclbin: ./jacobi1D_heter-ghost_pipeline_part4_N4096_cu4.cl
	$(XOCC) $(CLFLAGS) -D T=32 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T32.xtxt ;\

jacobi1D_heter-ghost_pipeline_part4_N4096_T16_cu4.xclbin: ./jacobi1D_heter-ghost_pipeline_part4_N4096_cu4.cl
	$(XOCC) $(CLFLAGS) -D T=16 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T16.xtxt ;\

jacobi1D_heter-ghost_pipeline_part4_N4096_T8_cu4.xclbin: ./jacobi1D_heter-ghost_pipeline_part4_N4096_cu4.cl
	$(XOCC) $(CLFLAGS) -D T=8 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T8.xtxt ;\

jacobi1D_heter-ghost_pipeline_part4_N4096_T4_cu4.xclbin: ./jacobi1D_heter-ghost_pipeline_part4_N4096_cu4.cl
	$(XOCC) $(CLFLAGS) -D T=4 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T4.xtxt ;\

jacobi1D_heter-ghost_pipeline_part4_N4096_T2_cu4.xclbin: ./jacobi1D_heter-ghost_pipeline_part4_N4096_cu4.cl
	$(XOCC) $(CLFLAGS) -D T=2 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T2.xtxt ;\

jacobi1D_heter-ghost_pipeline_part4_N4096_T1_cu4.xclbin: ./jacobi1D_heter-ghost_pipeline_part4_N4096_cu4.cl
	$(XOCC) $(CLFLAGS) -D T=1 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T1.xtxt ;\

clean:
	rm *.xo
