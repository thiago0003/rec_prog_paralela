#ifndef __CUDAKERNEL_H__
#define __CUDAKERNEL_H__


#include "helpers.cuh"

//Kernel Declaration
__global__ void NbodyForceGPU(Body *p,float dt,int nbodies){

	const int tx = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride_x = gridDim.x * blockDim.x;


    for(int i = tx; i < nbodies; i += stride_x){

		float dx,dy,dz, sqrd_dist, inv_dist, inv_dist3;
		float fx = 0.0f; 
		float fy = 0.0f; 
		float fz = 0.0f;
		//Calculate body forces
		for(int j = 0; j < nbodies; j++){
            if(i != j){//avoiding redundant calc
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
        }
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
		__syncthreads();
	}
}


void Nbody_wrapper(Body *bodies, int nbodies, int num_iter){
    
    const int NThreads = 32;
    const int Blocks = 80 * 32;
    
    Timer OverAll, T;

    
    Body *d_bodies;
    
    OverAll.start();
    cudaMalloc(&d_bodies, nbodies * sizeof(Body));
    check_last_error();

    for(int iter=0; iter < num_iter; iter++){

        //copy bodies to GPU
        T.start();
        cudaMemcpy(d_bodies, bodies,
                        nbodies*sizeof(Body), 
                        cudaMemcpyHostToDevice);
        check_last_error();
        T.stop("Memory Copy H->D");
        
        T.start();
        NbodyForceGPU   <<<Blocks,NThreads>>>
        (d_bodies, DT, nbodies);
        check_last_error();
        T.stop("Kernel - NbodyForceGPU");
        
        T.start();
        NbodyIteractGPU <<<Blocks, NThreads>>>
        (d_bodies, DT, nbodies);
        T.stop("Kernel - NbodyIteractGPU");

        T.start();
        cudaMemcpy(bodies, d_bodies, nbodies*sizeof(Body), cudaMemcpyHostToHost);
        check_last_error();
        T.stop("Memory Copy D->H");
    }


    cudaFree(d_bodies);
    check_last_error();
    OverAll.stop("Execução Total do Kernel");
    
}



#endif // __CUDAKERNEL_H__ 