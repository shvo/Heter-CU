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
GPU_GHOST_SRC = jacobi1D_gpu_ghost.c
FPGA_SRC = jacobi1D_fpga.c
FPGA_GHOST_SRC = jacobi1D_fpga_ghost.c
FPGA_HETER-GHOST_SRC = jacobi1D_fpga_heter-ghost.c
FPGA_HETER-GHOST-NOWHILE_SRC = jacobi1D_fpga_heter-ghost-nowhile.c
FPGA_HETER-GHOST-NOWHILE-N4096_SRC = jacobi1D_fpga_heter-ghost-nowhile-N4096.c
FPGA_HETER-GHOST-NOWHILE-N5460_SRC = jacobi1D_fpga_heter-ghost-nowhile-N5460.c
FPGA_BOARD_SRC = jacobi1D_fpga_board.c
WI_FPGA_BOARD_SRC = jacobi1D_wi_fpga_board.c

all: jacobi1D.exe \
     jacobi1D_gpu.exe \
     jacobi1D_gpu_ghost.exe \
     jacobi1D_fpga.exe \
     jacobi1D_fpga_ghost.exe \
     jacobi1D_fpga_heter-ghost.exe \
     jacobi1D_fpga_heter-ghost-nowhile-N4096.exe \
     jacobi1D_fpga_heter-ghost-nowhile-N5460.exe \
     jacobi1D_fpga_board.exe \
     jacobi1D_wi_fpga_board.exe \

jacobi1D.exe: $(SRC)
	$(CC) $(SRC) $(CCFLAGS) $(LIBS) -I$(OPENCL_INC) -I$(PATH_TO_COMMON) -L$(OPENCL_LIB) -o $@

jacobi1D_gpu.exe: $(GPU_SRC)
	$(CC) $(GPU_SRC) $(CCFLAGS) $(LIBS) -I$(OPENCL_INC) -I$(PATH_TO_COMMON) -L$(OPENCL_LIB) -o $@

jacobi1D_gpu_ghost.exe: $(GPU_GHOST_SRC)
	$(CC) $(GPU_GHOST_SRC) $(CCFLAGS) $(LIBS) -I$(OPENCL_INC) -I$(PATH_TO_COMMON) -L$(OPENCL_LIB) -o $@

jacobi1D_fpga.exe: $(FPGA_SRC)
	$(CC) $(FPGA_SRC) $(CCFLAGS) $(LIBS) -I$(OPENCL_INC) -I$(PATH_TO_COMMON) -L$(OPENCL_LIB) -o $@

jacobi1D_fpga_ghost.exe: $(FPGA_GHOST_SRC)
	$(CC) $(FPGA_GHOST_SRC) $(CCFLAGS) $(LIBS) -I$(OPENCL_INC) -I$(PATH_TO_COMMON) -L$(OPENCL_LIB) -o $@

jacobi1D_fpga_heter-ghost.exe: $(FPGA_HETER-GHOST_SRC)
	$(CC) $(FPGA_HETER-GHOST_SRC) $(CCFLAGS) $(LIBS) -I$(OPENCL_INC) -I$(PATH_TO_COMMON) -L$(OPENCL_LIB) -o $@

jacobi1D_fpga_heter-ghost-nowhile.exe: $(FPGA_HETER-GHOST-NOWHILE_SRC)
	$(CC) $(FPGA_HETER-GHOST-NOWHILE_SRC) $(CCFLAGS) $(LIBS) -I$(OPENCL_INC) -I$(PATH_TO_COMMON) -L$(OPENCL_LIB) -o $@

jacobi1D_fpga_heter-ghost-nowhile-N4096.exe: $(FPGA_HETER-GHOST-NOWHILE-N4096_SRC)
	$(CC) $(FPGA_HETER-GHOST-NOWHILE-N4096_SRC) $(CCFLAGS) $(LIBS) -I$(OPENCL_INC) -I$(PATH_TO_COMMON) -L$(OPENCL_LIB) -o $@

jacobi1D_fpga_heter-ghost-nowhile-N5460.exe: $(FPGA_HETER-GHOST-NOWHILE-N5460_SRC)
	$(CC) $(FPGA_HETER-GHOST-NOWHILE-N5460_SRC) $(CCFLAGS) $(LIBS) -I$(OPENCL_INC) -I$(PATH_TO_COMMON) -L$(OPENCL_LIB) -o $@

jacobi1D_fpga_board.exe: $(FPGA_BOARD_SRC)
	$(CC) $(FPGA_BOARD_SRC) $(CCFLAGS) $(LIBS) -I$(OPENCL_INC) -I$(PATH_TO_COMMON) -L$(OPENCL_LIB) -o $@

jacobi1D_wi_fpga_board.exe: $(WI_FPGA_BOARD_SRC)
	$(CC) $(WI_FPGA_BOARD_SRC) $(CCFLAGS) $(LIBS) -I$(OPENCL_INC) -I$(PATH_TO_COMMON) -L$(OPENCL_LIB) -o $@

clean:
	rm -f *.exe $(OUTPUT) 
