NVCC := nvcc
CXX_STD := --std=c++17

build:
	@echo "=========================================================="
	@echo "                 BUILDING  IMPLEMENTATIONS                "
	@echo "=========================================================="
	@echo "\n--- 01 CPU SEQUENTIAL ---"
	@$(NVCC) -arch=compute_75 $(CXX_STD) 01_sequential.cu -Iinclude -Iexternal -o 01_sequential
	@echo "\n--- 02 GPU NAIVE ---"
	@$(NVCC) -arch=compute_75 $(CXX_STD) 02_naive.cu -Iinclude -Iexternal -o 02_naive
	@echo "\n--- 03 GPU SHARED ---"
	@$(NVCC) -arch=compute_75 $(CXX_STD) 03_shared.cu -Iinclude -Iexternal -o 03_shared
	@echo "\n--- 04 GPU BEYOND ---"
	@$(NVCC) -arch=compute_75 $(CXX_STD) 04_beyond.cu -Iinclude -Iexternal -o 04_beyond
	@echo "\n--- 05 GPU CUTOFF ---"
	@$(NVCC) -arch=compute_75 $(CXX_STD) 05_cutoff.cu -Iinclude -Iexternal -o 05_cutoff
	@echo "==========================================================\n\n"

info:
	@echo "=========================================================="
	@echo "                SYSTEM HARDWARE INFORMATION               "
	@echo "=========================================================="
	@echo "\n--- SOFTWARE & COMPILER ---"
	@$(NVCC) --version | grep "release"
	@echo "C++ Standard: C++17"
	
	@echo "\n--- CPU ---"
	@lscpu | grep -E "Model name|CPU\(s\):|Core\(s\) per socket" | sed 's/^[ \t]*//'
	
	@echo "\n--- RAM ---"
	@free -h | awk '/^Mem:/ {printf "Total: %-10s Used: %-10s Free: %-10s Available: %-10s\n", $$2, $$3, $$4, $$7}'
	
	@echo "\n--- GPU ---"
	@nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | \
		awk -F', ' '{printf "Model: %-25s Driver: %-10s VRAM: %-10s\n", $$1, $$2, $$3}'
	@echo "==========================================================\n\n"

run: build
	@echo "=========================================================="
	@echo "                  RUNNING IMPLEMENTATIONS                 "
	@echo "=========================================================="
	./01_sequential
	./02_naive
	./03_shared
	./04_beyond
	./05_cutoff
	@echo "==========================================================\n\n"