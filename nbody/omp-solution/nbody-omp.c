#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define N_ITER 10 // number of simulation iterations
#define DT 0.01f // time step
#define SOFTENING 1e-9f // to avoid zero divisors

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


/*
 * Compute the gravitational impact among all pairs of bodies in 
 * the system.
 */

void body_force(Body *p, float dt, int n) {
  //Big-O --> n^2
  int i,j;
  #pragma omp parallel for private(i,j)
  for (i = 0; i < n; ++i) {
    float fx = 0.0f; 
    float fy = 0.0f; 
    float fz = 0.0f;

    for (j = 0; j < n; j++) {
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

    p[i].vx += dt*fx; 
    p[i].vy += dt*fy; 
    p[i].vz += dt*fz;
  }
}

/*
 * Read a dataset with initilized bodies.
 */

Body* read_dataset(int nbodies) {
 
  Body *p = (Body *)malloc(nbodies * sizeof(Body));

  for(int i = 0; i < nbodies; i++)
    fscanf(stdin, "%f %f %f %f %f %f\n",&p[i].x,  &p[i].y,  &p[i].z, 
                                        &p[i].vx, &p[i].vy, &p[i].vz);

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

    fprintf(fp, "%f %f %f %f %f %f\n",bodies[i].x,  bodies[i].y,  bodies[i].z, 
                                      bodies[i].vx, bodies[i].vy, bodies[i].vz);
  }
}


int main(int argc,char **argv) {

  omp_set_num_threads(8);
  char file_cpu[] = "output-omp.txt";
  int nbodies;
  fscanf(stdin,"%d",&(nbodies));

  
  Body *bodies = read_dataset(nbodies);


   /*
   * At each simulation iteration, interbody forces are computed,
   * and bodies' positions are integrated.
   */
  int iter, i;
  #pragma omp parallel for private(iter) shared(bodies)
  for (iter = 0; iter < N_ITER; iter++) {    
    body_force(bodies, DT, nbodies);
    for (i = 0; i < nbodies; i++) {
      bodies[i].x += bodies[i].vx * DT;
      bodies[i].y += bodies[i].vy * DT;
      bodies[i].z += bodies[i].vz * DT;
    }
  }
  

  write_dataset(nbodies, bodies, file_cpu);

  printf("\nArquivo gerado - %s\n", file_cpu);
  free(bodies);
  
  exit(EXIT_SUCCESS);
}
