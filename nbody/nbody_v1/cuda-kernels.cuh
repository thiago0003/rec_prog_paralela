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
                inv_dist = rsqrtf(sqrd_dist);
                inv_dist3 = inv_dist * inv_dist * inv_dist;

                fx += dx * inv_dist3; 
                fy += dy * inv_dist3; 
                fz += dz * inv_dist3;
            }
        }
        //write velocity   
        p[i].vx += dt*fx;
        p[i].vy += dt*fy;
        p[i].vz += dt*fz;
        p[i].x += p[i].vx * dt;
		p[i].y += p[i].vy * dt;
		p[i].z += p[i].vz * dt;
    }
}


void Nbody_wrapper(Body *bodies, int nbodies, int num_iter){
    
    const int NThreads = 32;
    const int Blocks = 80 * 32;
    const int sizeof_bodies = sizeof(Body)* nbodies; 
    uint64_t chunk_size, lower, upper, width;
    // Timer T;
    Timer OverAll;
    chunk_size = sdiv(nbodies, N_STREAMS);
    
    //Declaring Streams
    cudaStream_t streams[N_STREAMS];
    for(uint64_t i = 0; i < N_STREAMS; i++)
        cudaStreamCreate(&streams[i]);
    check_last_error();


    //Allocating Memory    
    // T.start();
    Body *d_bodies;
    
    cudaMalloc    (&d_bodies, sizeof_bodies);
    check_last_error();

    OverAll.start();
    for(int iter=0; iter < num_iter; iter++){
        for(uint64_t stream_id = 0; stream_id < N_STREAMS; stream_id ++){

            lower = stream_id * chunk_size;
            upper = std::min((uint64_t)nbodies,(uint64_t) lower + chunk_size);
            width = upper - lower;

            //copy bodies to GPU
            cudaMemcpyAsync(d_bodies + lower, bodies + lower,
                            width*sizeof(Body), 
                            cudaMemcpyHostToDevice);
            check_last_error();

            // T.stop("Memory Copy");
            // T.start();
            NbodyForceGPU   <<<Blocks,NThreads, 0, streams[stream_id]>>>
            (d_bodies + lower, DT, width);
            check_last_error();

            cudaMemcpyAsync(bodies + lower, d_bodies + lower, width*sizeof(Body), cudaMemcpyHostToHost);
            check_last_error();
        }
    }

    //Destroy All streams
    for(uint64_t i = 0; i < N_STREAMS; i++)
        cudaStreamDestroy(streams[i]);
    
    cudaFree(d_bodies);
    check_last_error();
    OverAll.stop("Execução Total do Kernel");

}



#endif // __CUDAKERNEL_H__ 