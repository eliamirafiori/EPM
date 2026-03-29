build:
	nvcc -arch=compute_75 --std=c++11 01_sequential.cu -Iinclude -Iexternal -o 01_sequential
	nvcc -arch=compute_75 --std=c++11 02_naive.cu -Iinclude -Iexternal -o 02_naive

info:
	echo "--- CPU ---";
	lscpu | grep "Model name";
	lscpu | grep -E "Model name|CPU\(s\)|Core\(s\) per socket"
	echo "--- RAM ---";
	free -h | grep "Mem:";
	echo "--- GPU ---";
	nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

run: build
	./01_sequential
	./02_naive