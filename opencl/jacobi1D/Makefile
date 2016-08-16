PATH_TO_COMMON = ../../common
include $(PATH_TO_COMMON)/make.config

ifndef CC
	CC = gcc
endif

OUTPUT = *.out
CCFLAGS=-O3 -lm 
LIBS = -lOpenCL 

SRC = jacobi1D.c
GPU_SRC = jacobi1D_gpu.c
#FPGA_SRC = 3D_fpga.c CL_helper.c ./util/opencl/opencl.o

all: jacobi1D_gpu.exe \
     jacobi1D.c \

jacobi1D.exe: $(SRC)
	$(CC) $(SRC) $(CCFLAGS) $(LIBS) -I$(OPENCL_INC) -I$(PATH_TO_COMMON) -L$(OPENCL_LIB) -o $@

jacobi1D_gpu.exe: $(GPU_SRC)
	$(CC) $(GPU_SRC) $(CCFLAGS) $(LIBS) -I$(OPENCL_INC) -I$(PATH_TO_COMMON) -L$(OPENCL_LIB) -o $@

clean:
	rm -f *.exe $(OUTPUT) 
