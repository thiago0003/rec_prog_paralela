#pragma once

#include <iostream>
#include <fstream>
#include <cstdint>
#include <string>
#include <stdio.h>

//Definitions About NBody
#define N_ITER 10 // number of simulation iterations
#define DT 0.01f // time step
#define SOFTENING 1e-9f // to avoid zero divisors
#define EPSILON (0.5f)
#define FLOAT_EQ(X,Y)( (fabs((X) - fabs(Y)) <= EPSILON) ? 1 : 0)

//Definitions about GPU
#define N_THREADS 32
#define N_STREAMS 64

/*
 * Each body holds coordinate positions (i.e., x, y, and z) and
 * velocities (i.e., vx, vy, and vz).
 */

// Structures used in this example
 typedef struct { 
	float x, y, z, vx, vy, vz;
	} Body;

typedef struct { 
    int x, y, z, vx, vy, vz; 
    } Body_int;



uint64_t sdiv (uint64_t a, uint64_t b) {
    return (a+b-1)/b;
}


int AreEqual(Body &p0, Body &p1){
	bool equal = true;
	equal = equal && FLOAT_EQ(p0.x , p1.x);
	equal = equal && FLOAT_EQ(p0.y , p1.y);
	equal = equal && FLOAT_EQ(p0.z , p1.z);
	equal = equal && FLOAT_EQ(p0.vx, p1.vx);
	equal = equal && FLOAT_EQ(p0.vy, p1.vy);
	equal = equal && FLOAT_EQ(p0.vz, p1.vz);
	
	return equal;
}
/*
 * Compute the gravitational impact among all pairs of bodies in 
 * the system.
 */

 int checkResults(Body* b_GPU, int nbodies){
 	

	FILE *file =NULL;
	char fname[] = "output-c.txt";
	file = fopen(fname, "r"); 
	if(file == NULL){
		printf("\nArquivo de Comparação nao encontrado - %s\n", fname);
		exit(1);
	}
	Body *p = (Body*) malloc(nbodies * sizeof(Body));


	for(int i = 0; i < nbodies; i++){
	fscanf(file, "%f %f %f %f %f %f\n",&p[i].x,  &p[i].y,  &p[i].z, 
										&p[i].vx, &p[i].vy, &p[i].vz);
	
		if(!AreEqual(p[i], b_GPU[i])){
			printf("Divergent Val in pos %i \nExpected:\n", i);
			printf("pos = %.5f,  %.5f,  %.5f\n",p[i].x, p[i].y, p[i].z);
			printf("vel = %.5f,  %.5f,  %.5f\n",p[i].vx, p[i].vy, p[i].vz);

			printf("Obtained:\n");
			printf("pos = %.5f,  %.5f,  %.5f\n",b_GPU[i].x, b_GPU[i].y, b_GPU[i].z);
			printf("vel = %.5f,  %.5f,  %.5f\n",b_GPU[i].vx, b_GPU[i].vy, b_GPU[i].vz);
			return 0;
		}
 	}
	return 1;
 }


inline void check_last_error ( ) {

    cudaError_t err;
    if ((err = cudaGetLastError()) != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(err) << " : "
                  << __FILE__ << ", line " << __LINE__ << std::endl;
            exit(1);
    }
}



class Timer {

    float time;
    const uint64_t gpu;
    cudaEvent_t ying, yang;

public:

    Timer (uint64_t gpu=0) : gpu(gpu) {
        cudaSetDevice(gpu);
        cudaEventCreate(&ying);
        cudaEventCreate(&yang);
    }

    ~Timer ( ) {
        cudaSetDevice(gpu);
        cudaEventDestroy(ying);
        cudaEventDestroy(yang);
    }

    void start ( ) {
        cudaSetDevice(gpu);
        cudaEventRecord(ying, 0);
    }

    void stop (std::string label) {
        cudaSetDevice(gpu);
        cudaEventRecord(yang, 0);
        cudaEventSynchronize(yang);
        cudaEventElapsedTime(&time, ying, yang);
        std::cout << "TIMING: " << time << " ms (" << label << ")" << std::endl;
    }
};
