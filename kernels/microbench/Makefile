CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
ARCH ?= sm_86
NVCCFLAGS ?= -O3 -arch=$(ARCH) -std=c++17 -lineinfo

build/peak: peak.cu
	mkdir -p build
	$(NVCC) $(NVCCFLAGS) -o $@ $< -lnvidia-ml

# Laptop on AC, performance profile, nothing else on the GPU.
run: build/peak
	mkdir -p ../../results/cuda
	./build/peak | tee ../../results/cuda/peak.json

clean:
	rm -rf build

.PHONY: run clean
