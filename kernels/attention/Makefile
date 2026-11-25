# Attention kernels (PLAN.md section 3.4). FP32, D = 64. One test, benchmark and profiling driver for
# every version (versions.h); VERSION picks the one to benchmark or profile.
CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc
ARCH      ?= sm_86
NVCCFLAGS ?= -O3 -arch=$(ARCH) -std=c++17 -lineinfo
BUILD     := build
PY        := ../../.venv/bin/python -I
RESULTS   := ../../results/attention
VERSION   ?= v2
KERNELS   := v1.cu v2.cu v3.cu
HEADERS   := attn.h versions.h

all: $(BUILD)/attn_test $(BUILD)/attn_bench

$(BUILD):
	mkdir -p $@

$(BUILD)/testdata/cases.txt: ../../bench/attn_testdata.py | $(BUILD)
	$(PY) ../../bench/attn_testdata.py

$(BUILD)/attn_test: attn_test.cu $(KERNELS) $(HEADERS) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) attn_test.cu $(KERNELS) -o $@

$(BUILD)/attn_bench: attn_bench.cu $(KERNELS) $(HEADERS) ../harness/harness.h | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -I../harness attn_bench.cu $(KERNELS) -o $@ -lnvidia-ml

$(BUILD)/attn_profile: attn_profile.cu $(KERNELS) $(HEADERS) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) attn_profile.cu $(KERNELS) -o $@

# Every version, every test case, every internal path.
test: $(BUILD)/attn_test $(BUILD)/testdata/cases.txt
	./$(BUILD)/attn_test

# Laptop on AC, performance profile, nothing else using the GPU.
bench: $(BUILD)/attn_bench
	mkdir -p $(RESULTS)
	./$(BUILD)/attn_bench $(VERSION) $(RESULTS)/$(VERSION).json

# Tile sweep at 12 heads, N = 1024, for V2 or V3 (selection rule in attn_bench.cu).
tune: $(BUILD)/attn_bench
	mkdir -p $(RESULTS)
	./$(BUILD)/attn_bench $(VERSION) $(RESULTS)/$(VERSION)_tune.json --tune

# What limits each kernel: Nsight Compute on the fourth run's launches (12 heads, N = 1024), caches and
# clocks left alone. Needs performance-counter access (PLAN.md checklist). The filter keeps kernel names
# and metric rows only (no process ids or paths).
NCU_METRICS := gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__throughput.avg.pct_of_peak_sustained_active,lts__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,dram__bytes_write.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum,sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread
KERNELS_PER_RUN := $(if $(filter v3,$(VERSION)),1,3)
SKIP_LAUNCHES   := $(if $(filter v3,$(VERSION)),3,9)
profile: $(BUILD)/attn_profile
	mkdir -p $(RESULTS)
	$(CUDA_HOME)/bin/ncu --launch-skip $(SKIP_LAUNCHES) --launch-count $(KERNELS_PER_RUN) \
	  --cache-control none --clock-control none \
	  --metrics $(NCU_METRICS) ./$(BUILD)/attn_profile $(VERSION) 2>&1 \
	  | grep -E "::v[0-9]_|Section|Metric|----|duration|pct|bytes|sectors|requests|conflicts|wavefronts|registers" \
	  > $(RESULTS)/$(VERSION)_ncu.txt
	cat $(RESULTS)/$(VERSION)_ncu.txt

clean:
	rm -rf $(BUILD)

.PHONY: all test bench tune profile clean
