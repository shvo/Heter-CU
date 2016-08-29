XILINX_SDACCEL :=/opt/Xilinx/SDAccel/2016.1
XILINX_OPENCL := $(XILINX_SDACCEL)
DSA := xilinx:adm-pcie-7v3:1ddr:2.1
XOCC := $(XILINX_SDACCEL)/bin/xocc
CLFLAGS := --xdevice $(DSA)

all : jacobi1D_heter-ghost_pipeline_part4_N3276_T2048_cu5.xo \
      jacobi1D_heter-ghost_pipeline_part4_N3276_T1024_cu5.xo \
      jacobi1D_heter-ghost_pipeline_part4_N3276_T512_cu5.xo \
      jacobi1D_heter-ghost_pipeline_part4_N3276_T256_cu5.xo \
      jacobi1D_heter-ghost_pipeline_part4_N3276_T128_cu5.xo \
      jacobi1D_heter-ghost_pipeline_part4_N3276_T64_cu5.xo \
      jacobi1D_heter-ghost_pipeline_part4_N3276_T32_cu5.xo \
      jacobi1D_heter-ghost_pipeline_part4_N3276_T16_cu5.xo \
      jacobi1D_heter-ghost_pipeline_part4_N3276_T8_cu5.xo \
      jacobi1D_heter-ghost_pipeline_part4_N3276_T4_cu5.xo \
      jacobi1D_heter-ghost_pipeline_part4_N3276_T2_cu5.xo \
      jacobi1D_heter-ghost_pipeline_part4_N3276_T1_cu5.xo \

.PHONY : all clean

jacobi1D_heter-ghost_pipeline_part4_N3276_T2048_cu5.xo: ./jacobi1D_heter-ghost_pipeline_part4_N3276_cu5.cl
	$(XOCC) $(CLFLAGS) -c -D T=2048 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T2048.xtxt ;\

jacobi1D_heter-ghost_pipeline_part4_N3276_T1024_cu5.xo: ./jacobi1D_heter-ghost_pipeline_part4_N3276_cu5.cl
	$(XOCC) $(CLFLAGS) -c -D T=1024 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T1024.xtxt ;\

jacobi1D_heter-ghost_pipeline_part4_N3276_T512_cu5.xo: ./jacobi1D_heter-ghost_pipeline_part4_N3276_cu5.cl
	$(XOCC) $(CLFLAGS) -c -D T=512 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T512.xtxt ;\

jacobi1D_heter-ghost_pipeline_part4_N3276_T256_cu5.xo: ./jacobi1D_heter-ghost_pipeline_part4_N3276_cu5.cl
	$(XOCC) $(CLFLAGS) -c -D T=256 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T256.xtxt ;\

jacobi1D_heter-ghost_pipeline_part4_N3276_T128_cu5.xo: ./jacobi1D_heter-ghost_pipeline_part4_N3276_cu5.cl
	$(XOCC) $(CLFLAGS) -c -D T=128 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T128.xtxt ;\

jacobi1D_heter-ghost_pipeline_part4_N3276_T64_cu5.xo: ./jacobi1D_heter-ghost_pipeline_part4_N3276_cu5.cl
	$(XOCC) $(CLFLAGS) -c -D T=64 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T64.xtxt ;\

jacobi1D_heter-ghost_pipeline_part4_N3276_T32_cu5.xo: ./jacobi1D_heter-ghost_pipeline_part4_N3276_cu5.cl
	$(XOCC) $(CLFLAGS) -c -D T=32 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T32.xtxt ;\

jacobi1D_heter-ghost_pipeline_part4_N3276_T16_cu5.xo: ./jacobi1D_heter-ghost_pipeline_part4_N3276_cu5.cl
	$(XOCC) $(CLFLAGS) -c -D T=16 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T16.xtxt ;\

jacobi1D_heter-ghost_pipeline_part4_N3276_T8_cu5.xo: ./jacobi1D_heter-ghost_pipeline_part4_N3276_cu5.cl
	$(XOCC) $(CLFLAGS) -c -D T=8 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T8.xtxt ;\

jacobi1D_heter-ghost_pipeline_part4_N3276_T4_cu5.xo: ./jacobi1D_heter-ghost_pipeline_part4_N3276_cu5.cl
	$(XOCC) $(CLFLAGS) -c -D T=4 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T4.xtxt ;\

jacobi1D_heter-ghost_pipeline_part4_N3276_T2_cu5.xo: ./jacobi1D_heter-ghost_pipeline_part4_N3276_cu5.cl
	$(XOCC) $(CLFLAGS) -c -D T=2 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T2.xtxt ;\

jacobi1D_heter-ghost_pipeline_part4_N3276_T1_cu5.xo: ./jacobi1D_heter-ghost_pipeline_part4_N3276_cu5.cl
	$(XOCC) $(CLFLAGS) -c -D T=1 --report system $< -o $@ ;\
	mv system_estimate.xtxt system_estimate_T1.xtxt ;\

clean:
	rm *.xo
