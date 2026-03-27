build:
	g++ 01_sequential.cpp -o 01_sequential
	nvcc --std=c++11 02_naive.cu -Iinclude -Iexternal -o 02_naive
	nvcc --std=c++11 02_naive.cu -Iinclude -Iexternal -o 02_naive
	nvcc --std=c++11 03_optimized.cu -Iinclude -Iexternal -o 03_optimized

run: build
	./01_sequential
	./02_naive
	./03_optimized