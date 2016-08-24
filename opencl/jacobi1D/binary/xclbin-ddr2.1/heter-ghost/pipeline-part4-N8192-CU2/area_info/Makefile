XILINX_SDACCEL :=/opt/Xilinx/SDAccel/2016.1
XILINX_OPENCL := $(XILINX_SDACCEL)
DSA := xilinx:adm-pcie-7v3:1ddr:2.1
XOCC := $(XILINX_SDACCEL)/bin/xocc
CLFLAGS := -s --xdevice $(DSA)

all : jacobi1D_heter-ghost_pipeline_part4_N8192_T8192_cu2.xo \
      jacobi1D_heter-ghost_pipeline_part4_N8192_T4096_cu2.xo \
      jacobi1D_heter-ghost_pipeline_part4_N8192_T2048_cu2.xo \
      jacobi1D_heter-ghost_pipeline_part4_N8192_T1024_cu2.xo \
      jacobi1D_heter-ghost_pipeline_part4_N8192_T512_cu2.xo \
      jacobi1D_heter-ghost_pipeline_part4_N8192_T256_cu2.xo \
      jacobi1D_heter-ghost_pipeline_part4_N8192_T128_cu2.xo \

.PHONY : all clean

jacobi1D_heter-ghost_pipeline_part4_N8192_T8192_cu2.xo: ./jacobi1D_heter-ghost_pipeline_part4_N8192_cu2.cl
	$(XOCC) $(CLFLAGS) -c -D T=8192 --report system $< -o $@ ;\
	mv system_estimate.xtxt  system_estimate_T8192.xtxt ;\

jacobi1D_heter-ghost_pipeline_part4_N8192_T4096_cu2.xo: ./jacobi1D_heter-ghost_pipeline_part4_N8192_cu2.cl
	$(XOCC) $(CLFLAGS) -c -D T=4096 --report system $< -o $@ ;\
	mv system_estimate.xtxt  system_estimate_T4096.xtxt ;\

jacobi1D_heter-ghost_pipeline_part4_N8192_T2048_cu2.xo: ./jacobi1D_heter-ghost_pipeline_part4_N8192_cu2.cl
	$(XOCC) $(CLFLAGS) -c -D T=2048 --report system $< -o $@ ;\
	mv system_estimate.xtxt  system_estimate_T2048.xtxt ;\

jacobi1D_heter-ghost_pipeline_part4_N8192_T1024_cu2.xo: ./jacobi1D_heter-ghost_pipeline_part4_N8192_cu2.cl
	$(XOCC) $(CLFLAGS) -c -D T=1024 --report system $< -o $@ ;\
	mv system_estimate.xtxt  system_estimate_T1024.xtxt ;\

jacobi1D_heter-ghost_pipeline_part4_N8192_T512_cu2.xo: ./jacobi1D_heter-ghost_pipeline_part4_N8192_cu2.cl
	$(XOCC) $(CLFLAGS) -c -D T=512 --report system $< -o $@ ;\
	mv system_estimate.xtxt  system_estimate_T512.xtxt ;\

jacobi1D_heter-ghost_pipeline_part4_N8192_T256_cu2.xo: ./jacobi1D_heter-ghost_pipeline_part4_N8192_cu2.cl
	$(XOCC) $(CLFLAGS) -c -D T=256 --report system $< -o $@ ;\
	mv system_estimate.xtxt  system_estimate_T256.xtxt ;\

jacobi1D_heter-ghost_pipeline_part4_N8192_T128_cu2.xo: ./jacobi1D_heter-ghost_pipeline_part4_N8192_cu2.cl
	$(XOCC) $(CLFLAGS) -c -D T=128 --report system $< -o $@ ;\
	mv system_estimate.xtxt  system_estimate_T128.xtxt ;\

clean:
	rm *.xo
