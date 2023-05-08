#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>


#define SEED 777

typedef struct {float x,y,z, vx,vy,vz;} Body;

int main(int argc, char **argv){

	FILE *fp;
	int nbodies;
	if (argc <= 1) {
		printf ("ERROR: ./input-generator <n_bodies> \n");
		return 1;
	}

	fp = fopen( "input.txt" , "w" );
	nbodies = atoi(argv[1]);
	// nbodies = 10;
	fprintf(fp, "%d\n", nbodies);
	Body *p = (Body*)malloc(nbodies*sizeof(Body));
	for(int i = 0; i < nbodies; i++){
		p[i].x  =(float) (rand() %1000); 
		p[i].y  =(float) (rand() %1000); 
		p[i].z  =(float) (rand() %1000); 
		p[i].vx =(float) (rand() %1000); 
		p[i].vy =(float) (rand() %1000); 
		p[i].vz =(float) (rand() %1000); 
	    fprintf(fp, "%f %f %f %f %f %f\n",p[i].x,  p[i].y,  p[i].z, 
                                            p[i].vx, p[i].vy, p[i].vz);
   }
   fclose(fp);

   printf("\nArquivo Gerado com %i Elementos\n", nbodies);
  
   return(0);
}
