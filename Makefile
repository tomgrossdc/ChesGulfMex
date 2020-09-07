# Minimal Makefile for particles
# make particles
# make clean


# Location of the CUDA Toolkit
CUDA_PATH ?= /usr/local/cuda

# /usr/bin/cuda-g++  no longer is constructed, or needed.  Don't know if that is a problem
#HOST_COMPILER := /usr/bin/g++
HOST_COMPILER := g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# Common includes and paths for CUDA
INCLUDES  := -I/usr/local/cuda/samples/common/inc
#LIBRARIES := -L/usr/local/cuda/samples/common/lib/linux/x86_64 -L/usr/lib/nvidia-compute-utils-440
LIBRARIES := -L/usr/lib/nvidia-compute-utils-440

################################################################################
# OpenGL libraries
# Makefile include to help find GL Libraries
# include ./findgllib.mk

#LIBRARIES += $(GLLINK)
LIBRARIES += -lGL -lGLU -lglut
LIBRARIES += -lGLEW

################################################################################
##  The particle routine makefile
#LIBS += -L/home/tom/code/NETCDFvcpkg/lib
#INCLUDES += -I/home/tom/code/NETCDFvcpkg/include

LIBS	= -L/usr/local/lib -L/usr/local/hdf4/lib
#LIBS += -L/usr/lib/x86_64-linux-gnu
#LIBS += -L/home/tom/code/vcpkg-master/packages/netcdf-cxx4_x64-linux/lib
INCLUDES += -I/usr/local/include -I/usr/local/hdf4/include
#INCLUDES += -I/usr/include
#INCLUDES += -I/home/tom/code/vcpkg-master/packages/netcdf-cxx4_x64-linux/include
INCLUDES += -Idelaunator-cpp-master/include

LIBRARIES += -lnetcdf_c++4 -lnetcdf
ARFLAGS = rv
CPPFLAGS        = $(INCLUDES) -g
CPP             =nvcc 
	
SRCS = MMeshC.cpp DDataC.cpp PParticle.cpp MakeShader.cpp MMeshG.cpp DDataG.cpp
#HINCS = MMesh.h DData.h Main.h PPart.h
#triangulation.cpp
#date.cpp 
OBJS = ${SRCS:.cpp=.o  }
	#g++ -g -c -o mesh.o mesh.cpp $(INCLUDES)
	#g++ -g -c -o data.o data.cpp $(INCLUDES)
	#g++ -g -c -o particle.o particle.cpp  $(INCLUDES)
	#g++ -g -c -o MakeShader.o MakeShader.cpp  $(INCLUDES)
	#g++ -g -c -o triangulation.o triangulation.cpp
	#ar rv libparticle.a mesh.o data.o particle.o triangulation.o MakeShader.o 

################################################################################
# Compilation

Mainpart: $(SRCS)  Mainpart.o libparticles.a
	$(NVCC) -o Mainpart  Mainpart.cu  libparticles.a  $(INCLUDES) $(LIBS) $(LIBRARIES)
#nvcc mainpart.cu -o mainpart.x libparticles.a $(CPPFLAGS) $(LIBS)

Mainpart.o:Mainpart.cu Main.h ChesPartGL.cu
	$(NVCC) $(INCLUDES) -o Mainpart.o -c Mainpart.cu
    

libparticles.a:  ${OBJS} ${OBJSC} 
	ar ${ARFLAGS} $@ $?

run: libparticles.a 

clean:
	rm -f core  Mainpart  Mainpart.o  libparticles.a
	rm -f MakeShader.o  DDataC.o  MMeshC.o  PParticle.o  DDataG.o MMeshG.o
	
clobber: clean
