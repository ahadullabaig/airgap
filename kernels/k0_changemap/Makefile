# K0 block change map: CPU versions (serial reference + OpenMP) and the CUDA version.
# `make` builds the CPU versions only (no CUDA toolkit needed); `make cuda` builds the GPU test and benchmark.

CXX      ?= g++
CXXFLAGS ?= -O3 -march=native -std=c++17 -Wall -Wextra
OMPFLAGS := -fopenmp
BUILD    := build
SRCS     := k0_serial.cpp k0_omp.cpp
FLAGS_STR = $(CXX) $(CXXFLAGS) $(OMPFLAGS)
RESULTS  := ../../results/k0

CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc
ARCH      ?= sm_86
NVCCFLAGS ?= -O3 -arch=$(ARCH) -std=c++17 -lineinfo
CUDA_SRCS := k0_cuda.cu k0_serial.cpp

all: $(BUILD)/libk0.so $(BUILD)/k0_test $(BUILD)/k0_bench

cuda: $(BUILD)/k0_cuda_test $(BUILD)/k0_cuda_bench

$(BUILD):
	mkdir -p $@

$(BUILD)/libk0.so: $(SRCS) k0.h | $(BUILD)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -fPIC -shared $(SRCS) -o $@

$(BUILD)/k0_test: k0_test.cpp $(SRCS) k0.h | $(BUILD)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) k0_test.cpp $(SRCS) -o $@

$(BUILD)/k0_bench: k0_bench.cpp $(SRCS) k0.h | $(BUILD)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -DK0_FLAGS='"$(FLAGS_STR)"' k0_bench.cpp $(SRCS) -o $@

$(BUILD)/k0_cuda_test: k0_cuda_test.cu $(CUDA_SRCS) k0_cuda.h k0.h | $(BUILD)
	$(NVCC) $(NVCCFLAGS) k0_cuda_test.cu $(CUDA_SRCS) -o $@

$(BUILD)/k0_cuda_bench: k0_cuda_bench.cu $(CUDA_SRCS) k0_cuda.h k0.h ../harness/harness.h | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -I../harness k0_cuda_bench.cu $(CUDA_SRCS) -o $@ -lnvidia-ml

test: $(BUILD)/k0_test
	./$(BUILD)/k0_test

test-cuda: $(BUILD)/k0_cuda_test
	./$(BUILD)/k0_cuda_test

# Laptop on AC, performance profile, nothing else using the GPU.
bench-cuda: $(BUILD)/k0_cuda_bench
	mkdir -p $(RESULTS)
	./$(BUILD)/k0_cuda_bench $(RESULTS)/bench_gpu.json

bench: $(BUILD)/k0_bench
	mkdir -p $(RESULTS)
	./$(BUILD)/k0_bench $(RESULTS)/bench_cpu.json

# One frame pair re-read: cache-resident up to 1080p (for comparison with the ring default).
bench-hot: $(BUILD)/k0_bench
	mkdir -p $(RESULTS)
	./$(BUILD)/k0_bench --hot $(RESULTS)/bench_cpu_hot.json

clean:
	rm -rf $(BUILD)

.PHONY: all cuda test bench bench-hot test-cuda bench-cuda clean
