#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "cuda-kernels.cuh"


inline void debugMode(){
	#ifndef ONLINE_JUDGE
	freopen("input.txt", "r", stdin);
	// freopen("output.txt", "w", stdout);
	#endif //ONLINE_JUDGE
}

/*
 * Read a dataset with initilized bodies.
 */
 Body* read_dataset(int nbodies) {

	// Body *p = (Body *)malloc(nbodies * sizeof(Body));
	Body *p; 
	//Allocate in RAM 
	cudaMallocHost(&p, nbodies*sizeof(Body));
	
	for(int i = 0; i < nbodies; i++){
	fscanf(stdin, "%f %f %f %f %f %f\n",&p[i].x,  &p[i].y,  &p[i].z, 
										&p[i].vx, &p[i].vy, &p[i].vz);

	}
	return p;
}

/*
 * Write simulation results into a dataset.
 */
 void write_dataset(const int nbodies, Body *bodies, char *fname) {

	// Body_int *bodies_int = (Body_int *)malloc(nbodies * sizeof(Body_int)); 
	FILE *fp;
	fp = fopen(fname, "w");

	for (int i = 0; i < nbodies; i++) {
		fprintf(fp, "%f %f %f %f %f %f\n", bodies[i].x,  bodies[i].y,  bodies[i].z, 
										   bodies[i].vx, bodies[i].vy, bodies[i].vz);
	}

}

int main(int argc,char **argv) {

	debugMode();	
	char file_gpu[] = "output-cuda_v1.txt";
	int nbodies;
	
	fscanf(stdin,"%d",&(nbodies));

	Body *bodies = read_dataset(nbodies);
	
	/*
	 * At each simulation iteration, interbody forces are computed,
	 * and bodies' positions are integrated.
	 */
	Nbody_wrapper(bodies, nbodies, N_ITER);
	//copyToHost - Receive values from GPU

	if(checkResults(bodies, nbodies) == 0 ){
		printf("Test not passed!\n");
		exit(EXIT_FAILURE);
	}else {
		printf("Test Passed!\n");
		write_dataset(nbodies, bodies, file_gpu);
	}

	cudaFree(bodies);

	
	exit(EXIT_SUCCESS);
}
