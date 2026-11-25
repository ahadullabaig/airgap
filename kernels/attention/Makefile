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

# X2: fixed costs (empty kernels with each version's launch configuration) and host-device copies at
# 12 heads, N = 1024 (x2_bench.cu), then warp-stall profiles of every kernel, including the sweeps'
# alternative tile shapes as interventions: V2 scores 64x64x8/4x4 (conflict-free shared stores) with
# output 64x64x16/4x4 (twice the occupancy), and V3 64x32/128 (two blocks per SM instead of one).
$(BUILD)/x2_bench: x2_bench.cu attn.h ../harness/harness.h | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -I../harness x2_bench.cu -o $@ -lnvidia-ml

x2: $(BUILD)/x2_bench
	mkdir -p $(RESULTS)
	./$(BUILD)/x2_bench $(RESULTS)/x2.json

comma := ,
empty :=
space := $(empty) $(empty)
X2_STALLS := barrier branch_resolving dispatch_stall drain imc_miss lg_throttle long_scoreboard math_pipe_throttle membar mio_throttle misc no_instruction not_selected selected short_scoreboard sleeping tex_throttle wait
X2_METRICS := gpu__time_duration.sum,smsp__average_warp_latency_per_inst_issued.ratio,$(subst $(space),$(comma),$(foreach r,$(X2_STALLS),smsp__average_warps_issue_stalled_$(r)_per_issue_active.ratio)),sm__warps_active.avg.pct_of_peak_sustained_active,sm__maximum_warps_per_active_cycle_pct,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,launch__occupancy_limit_warps,launch__occupancy_limit_blocks,launch__waves_per_multiprocessor,launch__registers_per_thread,sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,dram__bytes_read.sum,dram__bytes_write.sum
# name | attn_profile arguments | launches to skip | launches to profile
X2_PROFILES := v1:v1:9:3 v2:v2:9:3 v2alt:v2_1_2:9:3 v3:v3:3:1 v3alt:v3_1:3:1
x2-profile: $(BUILD)/attn_profile
	mkdir -p $(RESULTS)
	for spec in $(X2_PROFILES); do \
	  name=$${spec%%:*}; rest=$${spec#*:}; args=$${rest%%:*}; rest=$${rest#*:}; skip=$${rest%%:*}; count=$${rest#*:}; \
	  $(CUDA_HOME)/bin/ncu --launch-skip $$skip --launch-count $$count --cache-control none --clock-control none \
	    --metrics $(X2_METRICS) ./$(BUILD)/attn_profile $$(echo $$args | tr _ ' ') 2>&1 \
	    | grep -E "::v[0-9]_|Section|Metric|----|duration|pct|bytes|conflicts|stalled|latency|occupancy|warps|registers|waves" \
	    > $(RESULTS)/x2_ncu_$$name.txt; \
	done
	ls -l $(RESULTS)/x2_ncu_*.txt

clean:
	rm -rf $(BUILD)

.PHONY: all test bench tune profile x2 x2-profile clean
