include autodiff/autodiff/numpy_grad/Makefile

CONDAROOT ?= ${HOME}/opt/conda

all: test

test:
	${CONDAROOT}/bin/pytest autodiff/tests
