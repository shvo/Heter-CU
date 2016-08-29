PATH_TO_COMMON = ../../common
include $(PATH_TO_COMMON)/make.config

ifndef CC
	CC = gcc
endif

OUTPUT = *.out
CCFLAGS=-O3 -lm 
LIBS = -lOpenCL 

GPU_SRC = jacobi2D_gpu.c
GPU_GHOST_SRC = jacobi2D_ghost_gpu.c

all: jacobi2D_gpu.exe \
     jacobi2D_ghost_gpu.exe \

jacobi2D_gpu.exe: $(GPU_SRC)
	$(CC) $< $(CCFLAGS) $(LIBS) -I$(OPENCL_INC) -I$(PATH_TO_COMMON) -L$(OPENCL_LIB) -o $@

jacobi2D_ghost_gpu.exe: $(GPU_GHOST_SRC)
	$(CC) $<  $(CCFLAGS) $(LIBS) -I$(OPENCL_INC) -I$(PATH_TO_COMMON) -L$(OPENCL_LIB) -o $@

clean:
	rm -f *.exe $(OUTPUT) 