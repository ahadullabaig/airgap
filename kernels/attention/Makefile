# Attention kernels (PLAN.md section 3.4). FP32, D = 64.
CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc
ARCH      ?= sm_86
NVCCFLAGS ?= -O3 -arch=$(ARCH) -std=c++17 -lineinfo
BUILD     := build
PY        := ../../.venv/bin/python -I
RESULTS   := ../../results/attention

all: $(BUILD)/v1_test $(BUILD)/v1_bench

$(BUILD):
	mkdir -p $@

$(BUILD)/testdata/rand_h12_n1024/meta.json: ../../bench/attn_testdata.py | $(BUILD)
	$(PY) ../../bench/attn_testdata.py

$(BUILD)/v1_test: v1_test.cu v1.cu attn.h | $(BUILD)
	$(NVCC) $(NVCCFLAGS) v1_test.cu v1.cu -o $@

$(BUILD)/v1_bench: v1_bench.cu v1.cu attn.h ../harness/harness.h | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -I../harness v1_bench.cu v1.cu -o $@ -lnvidia-ml

test: $(BUILD)/v1_test $(BUILD)/testdata/rand_h12_n1024/meta.json
	./$(BUILD)/v1_test

# Laptop on AC, performance profile, nothing else using the GPU.
bench: $(BUILD)/v1_bench
	mkdir -p $(RESULTS)
	./$(BUILD)/v1_bench $(RESULTS)/v1.json

$(BUILD)/v1_profile: v1_profile.cu v1.cu attn.h | $(BUILD)
	$(NVCC) $(NVCCFLAGS) v1_profile.cu v1.cu -o $@

# What limits each V1 kernel: Nsight Compute on one launch of each (12 heads, N = 1024), caches and
# clocks left alone. Needs performance-counter access (PLAN.md checklist).
NCU_METRICS := gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__throughput.avg.pct_of_peak_sustained_active,lts__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,dram__bytes_write.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active
profile: $(BUILD)/v1_profile
	mkdir -p $(RESULTS)
	$(CUDA_HOME)/bin/ncu --launch-skip 3 --launch-count 3 --cache-control none --clock-control none \
	  --metrics $(NCU_METRICS) ./$(BUILD)/v1_profile 2>&1 \
	  | grep -E "::v1_|Section|Metric|----|duration|pct|bytes|sectors|requests" > $(RESULTS)/v1_ncu.txt
	cat $(RESULTS)/v1_ncu.txt

clean:
	rm -rf $(BUILD)

.PHONY: all test bench profile clean
