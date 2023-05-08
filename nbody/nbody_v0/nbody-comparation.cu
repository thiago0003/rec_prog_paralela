#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>	

#define N_ITER 10 // number of simulation iterations
#define DT 0.01f // time step
#define SOFTENING 1e-9f // to avoid zero divisors
#define EPSILON (0.000005f)
#define N_THREADS 32
#define FLOAT_EQ(X,Y)( (fabs((X) - (Y)) <= EPSILON) ? 1 : 0)

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

inline void debugMode(){
	#ifndef ONLINE_JUDGE
	freopen("input.txt", "r", stdin);
	freopen("output.txt", "a", stdout);
	#endif //ONLINE_JUDGE
}
int AreEqual(Body &p0, Body &p1){
	if(p0.x != p1.x || 
		 p0.y != p1.y ||
		 p0.z != p1.z ||
		 p0.vx != p1.vx ||
		 p0.vy != p1.vy ||
		 p0.vz != p1.vz ) return 0;
	return 1;
}
/*
 * Compute the gravitational impact among all pairs of bodies in 
 * the system.
 */

 int checkResults(Body* b_GPU, Body* b_CPU, int nbodies){
 	int c = 1;
 	for(int i = 0 ; i < nbodies; i++){
 		if(!AreEqual(b_CPU[i], b_GPU[i])){
 			printf("They are not\n");
 			return 0;
 		}
 	}
 	return c;

 }

/*
 * Read a binary dataset with initilized bodies.
 */
 Body* read_dataset(int nbodies) {

	Body *p = (Body *)malloc(nbodies * sizeof(Body));

	for(int i = 0; i < nbodies; i++)
	fscanf(stdin, "%f %f %f %f %f %f\n",&p[i].x,  &p[i].y,  &p[i].z, 
					&p[i].vx, &p[i].vy, &p[i].vz);
	return p;
}

/*
 * Write simulation results into a binary dataset.
 */
 void write_dataset(const int nbodies, Body *bodies, char *fname) {

	// Body_int *bodies_int = (Body_int *)malloc(nbodies * sizeof(Body_int)); 
	FILE *fp;
	fp = fopen(fname, "w");

	for (int i = 0; i < nbodies; i++) {
		fprintf(fp, "%f %f %f %f %f %f\n",bodies[i].x,  bodies[i].y,  bodies[i].z, 
		bodies[i].vx, bodies[i].vy, bodies[i].vz);
	}

}

__global__ void NbodyForceGPU(Body *p,float dt,int nbodies){


	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i<nbodies){

		float dx,dy,dz, sqrd_dist, inv_dist, inv_dist3;
		float fx = 0.0f; 
		float fy = 0.0f; 
		float fz = 0.0f;
		//Calculate body forces
		for(int j = 0; j < nbodies; j++){
			dx = p[j].x - p[i].x;
			dy = p[j].y - p[i].y;
			dz = p[j].z - p[i].z;
			sqrd_dist = dx*dx + dy*dy + dz*dz + SOFTENING;
			inv_dist = 1 / sqrt(sqrd_dist);
			inv_dist3 = inv_dist * inv_dist * inv_dist;

			fx += dx * inv_dist3; 
			fy += dy * inv_dist3; 
			fz += dz * inv_dist3;
		}
		__syncthreads();
		p[i].vx += dt*fx; 
		p[i].vy += dt*fy; 
		p[i].vz += dt*fz;
	}

}

__global__ void NbodyIteractGPU(Body *bodies,float dt,int nbodies){

	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if(i < nbodies){
		
		bodies[i].x += bodies[i].vx * dt;
		bodies[i].y += bodies[i].vy * dt;
		bodies[i].z += bodies[i].vz * dt;
	}
}

void body_force(Body *p, float dt, int n) {
  //Big-O --> n^2
  for (int i = 0; i < n; ++i) {
    float fx = 0.0f; 
    float fy = 0.0f; 
    float fz = 0.0f;


    //Parallel Loop -> OpenMP

    
    {
      for (int j = 0; j < n; j++) {
        float dx = p[j].x - p[i].x;
        float dy = p[j].y - p[i].y;
        float dz = p[j].z - p[i].z;
        float sqrd_dist = dx*dx + dy*dy + dz*dz + SOFTENING;
        float inv_dist = 1 / sqrt(sqrd_dist);
        float inv_dist3 = inv_dist * inv_dist * inv_dist;


        fx += dx * inv_dist3; 
        fy += dy * inv_dist3; 
        fz += dz * inv_dist3;
      }
    }
    p[i].vx += dt*fx; 
    p[i].vy += dt*fy; 
    p[i].vz += dt*fz;
  }
}

int main(int argc,char **argv) {

	debugMode();

	
	// char file_gpu[] = "output-cuda.txt";
	int nbodies;
	fscanf(stdin,"%d",&(nbodies));

	int N_BLOCKS =(int) (N_THREADS + nbodies - 1)/N_THREADS;

	Body *bodies = read_dataset(nbodies);
	Body *GPU_bodies = (Body *) malloc(nbodies * sizeof(Body));

	float gpu_time_used=0;
	cudaEvent_t start, stop;

	(cudaEventCreate(&start));
	(cudaEventCreate(&stop));

	(cudaEventRecord(start, 0));

	//Allocate Dev pointers
	Body *d_bodies;
	cudaMalloc(&d_bodies, nbodies*sizeof(Body));
	//Copy to device
	cudaMemcpy(d_bodies, bodies, nbodies*sizeof(Body), cudaMemcpyHostToDevice);

	 for (int iter = 0; iter < N_ITER; iter++) {
	 	NbodyForceGPU<<<N_BLOCKS, N_THREADS>>>(d_bodies, DT, nbodies);
		cudaDeviceSynchronize();	//Wait NBodyForce
		NbodyIteractGPU<<<N_BLOCKS, N_THREADS>>>(d_bodies, DT, nbodies); 
	}
	cudaDeviceSynchronize();// Wait Last Result

	//copyToHost - Receive values from GPU
	cudaMemcpy(GPU_bodies, d_bodies, nbodies*sizeof(Body), cudaMemcpyDeviceToHost);

	(cudaEventRecord(stop, 0));
	(cudaEventSynchronize (stop) );
	
	(cudaEventElapsedTime(&gpu_time_used, start, stop) );

	(cudaEventDestroy(start));
	(cudaEventDestroy(stop));



	float t;
	float cpu_time_used;

	//CPU time
	t = clock();
	for (int iter = 0; iter < N_ITER; iter++) {
	  body_force(bodies, DT, nbodies);
	  for (int i = 0; i < nbodies; i++) {
	    bodies[i].x += bodies[i].vx * DT;
	    bodies[i].y += bodies[i].vy * DT;
	    bodies[i].z += bodies[i].vz * DT;
	  }
	}

	t = clock() -t;
 	cpu_time_used = ((float)t/CLOCKS_PER_SEC)*1000; //returns mili-seconds :)

 	// printf("Nbodies - %d\n", nbodies);
 	// printf("Time elapsed:\nGPU -%8.2fms\nCPU -%8.2fms\n",gpu_time_used ,cpu_time_used);
 	// printf("Total Speed-up: %f", cpu_time_used/gpu_time_used );
	printf("Nbodies, time_cpu, time_gpu, speed_up\n");
	printf("%d, %.2f, %.2f, %.2f", nbodies, cpu_time_used, gpu_time_used, cpu_time_used/gpu_time_used);




	// write_dataset(nbodies, GPU_bodies, file_gpu);

	free(bodies);
	
	exit(EXIT_SUCCESS);
}
