#!/bin/bash
DIR=`dirname $0`

nvcc -w -std=c++11 -arch=sm_62 "$DIR"/02_naive.cu -I"$DIR"/external -o 02_naive
