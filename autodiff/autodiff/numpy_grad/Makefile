include Makefile.variable

FLAGS ?=

UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
MKLROOT ?= /opt/intel/mkl
MKLEXT ?= a
CXXFLAGS :=
endif

ifeq ($(UNAME_S),Linux)
MKLROOT ?= ${HOME}/opt/conda
MKLEXT ?= so
CXXFLAGS := -Wl,--no-as-needed
endif

CXX = g++
CXXFLAGS := ${CXXFLAGS} \
	-std=c++17 -O3 -m64 \
	${FLAGS}

# For MKL

MKLLINKLINE := ${MKLROOT}/lib/libmkl_intel_lp64.${MKLEXT} \
	${MKLROOT}/lib/libmkl_sequential.${MKLEXT} \
	${MKLROOT}/lib/libmkl_core.${MKLEXT} \
	-lpthread -lm -ldl

INCLUDES = 	-I${MKLROOT}/include

path = autodiff/autodiff/numpy_grad

python3_config_path = ${MKLROOT}/bin/python3-config

python3_path = ${MKLROOT}/bin/python3

# For pybind11
all: build

build: TARGET = ${path}/_vjp`${python3_config_path} --extension-suffix`

build: INCLUDES += `${python3_path} -m pybind11 --includes`

build: CXXFLAGS += -shared -Wall -fPIC

build: CXXFLAGS += -DPYTHON_LIB

# For numpy

build: INCLUDES += -I${MKLROOT}/lib/python3.7/site-packages/numpy/core/include/

build: ${path}/__vjps.cpp

	${CXX} ${CXXFLAGS} ${INCLUDES} -o ${TARGET} $< ${MKLLINKLINE}

.PHONY: clean
clean:
	rm -rf *.so
