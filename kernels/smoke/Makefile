CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
ARCH ?= sm_86
NVCCFLAGS ?= -O3 -arch=$(ARCH) -std=c++17 -lineinfo

build/smoke: smoke.cu
	mkdir -p build
	$(NVCC) $(NVCCFLAGS) -o $@ $<

run: build/smoke
	mkdir -p ../../results/cuda
	./build/smoke | tee ../../results/cuda/smoke.json

clean:
	rm -rf build

.PHONY: run clean

# Toolchain versions, for the record (results/cuda/toolchain.txt)
toolchain:
	mkdir -p ../../results/cuda
	{ echo "# nvcc"; $(NVCC) --version | tail -2; \
	  echo "# host compiler"; g++ --version | head -1; \
	  echo "# driver"; nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader; \
	  echo "# torch"; ../../.venv/bin/python -I -c "import torch; print(torch.__version__, 'cuda', torch.version.cuda)"; \
	} > ../../results/cuda/toolchain.txt
	cat ../../results/cuda/toolchain.txt

.PHONY: toolchain

# Nsight Compute check: DRAM traffic of vec_add should be 2 x 64 MiB read, 64 MiB written
profile: build/smoke
	mkdir -p ../../results/cuda
	$(CUDA_HOME)/bin/ncu --kernel-name vec_add --launch-count 1 \
	  --metrics dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum ./build/smoke \
	  2>&1 | grep -A8 "vec_add(" > ../../results/cuda/smoke_ncu.txt
	cat ../../results/cuda/smoke_ncu.txt

.PHONY: profile
