# K0 block change map: CPU versions (serial reference + OpenMP).
# The CUDA version is added once the CUDA toolkit is set up.

CXX      ?= g++
CXXFLAGS ?= -O3 -march=native -std=c++17 -Wall -Wextra
OMPFLAGS := -fopenmp
BUILD    := build
SRCS     := k0_serial.cpp k0_omp.cpp
FLAGS_STR = $(CXX) $(CXXFLAGS) $(OMPFLAGS)
RESULTS  := ../../results/k0

all: $(BUILD)/libk0.so $(BUILD)/k0_test $(BUILD)/k0_bench

$(BUILD):
	mkdir -p $@

$(BUILD)/libk0.so: $(SRCS) k0.h | $(BUILD)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -fPIC -shared $(SRCS) -o $@

$(BUILD)/k0_test: k0_test.cpp $(SRCS) k0.h | $(BUILD)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) k0_test.cpp $(SRCS) -o $@

$(BUILD)/k0_bench: k0_bench.cpp $(SRCS) k0.h | $(BUILD)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -DK0_FLAGS='"$(FLAGS_STR)"' k0_bench.cpp $(SRCS) -o $@

test: $(BUILD)/k0_test
	./$(BUILD)/k0_test

bench: $(BUILD)/k0_bench
	mkdir -p $(RESULTS)
	./$(BUILD)/k0_bench $(RESULTS)/bench_cpu.json

clean:
	rm -rf $(BUILD)

.PHONY: all test bench clean
