CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
ARCH ?= sm_86
NVCCFLAGS ?= -O3 -arch=$(ARCH) -std=c++17 -lineinfo

build/selftest: selftest.cu harness.h
	mkdir -p build
	$(NVCC) $(NVCCFLAGS) -o $@ $< -lnvidia-ml

# Laptop on AC, performance profile, nothing else using the GPU.
selftest: build/selftest
	./build/selftest
	../../.venv/bin/python -I -c "import json; d = json.load(open('../../results/harness/selftest.json')); print('selftest.json parses:', len(d['ops']), 'ops')"

# Also checks recovery from a lost CUDA context, by faulting the GPU on purpose (a driver Xid error).
selftest-fault: build/selftest
	./build/selftest --with-fault

clean:
	rm -rf build

.PHONY: selftest selftest-fault clean
