

#About CPU
CC=gcc
CC_FLAGS=-O3

OMP_FLAGS= -O3 -fopenmp
OMP_DIR=omp-solution
BASE_DIR=baseline

# About GPU
NVCC=nvcc
NVCC_FLAGS= #-ftz=true #--use_fast_math
NSYS=nsys profile
NSYSFLAGS=--stats=true --force-overwrite=true
INCLUDE_DIR=include

#About nbodies
version?=0
nbodies?=1000


#Executable fname
EX_NAME=nbody

.PHONY: nbody_gpu gpu_profile 


nbody_gpu:
	$(NVCC) $(NVCC_FLAGS) -o $(EX_NAME)-v$(version) -I ./$(INCLUDE_DIR) nbody_v$(version)/$(EX_NAME)-cuda.cu

gpu_profile: input.txt nbody_gpu
	mkdir -p nbody_v$(version)/Profiling
	$(NSYS) $(NSYSFLAGS) -o nbody_v$(version)/Profiling/nbody-report ./nbody-v$(version) < input.txt

nbody_omp:
	$(CC) $(OMP_FLAGS) -o $(EX_NAME)-omp $(OMP_DIR)/$(EX_NAME)-omp.c -lm 

nbody_c:
	$(CC) $(CC_FLAGS) -o $(EX_NAME)-c $(BASE_DIR)/$(EX_NAME)-c.c -lm

input: $(INCLUDE_DIR)/input-generator.c
	$(CC) $(CC_FLAGS) -o input-generator $(INCLUDE_DIR)/input-generator.c
	./input-generator $(nbodies)



clean:
	rm input-generator nbody-* output-*
