/*
 * Cognome: Ragazzi
 * Nome: Luca
 * Matricola: 0000 753758
 * 
 * This version of Larger than Life is implemented in CUDA and uses dynamic shared memory and should work correctly with any domain size. 
 */
#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <ctype.h> /* for isdigit */

#define BLKSIZE 32

typedef unsigned char cell_t;

/* This struct is defined here as an example; it is possible to modify
   this definition, or remove it altogether if you prefer to pass
   around the pointer to the bitmap directly. */
typedef struct {
	int n;
	cell_t *bmap;
} bmap_t;

/* Returns a pointer to the cell of coordinates (i,j) in the bitmap bmap. */
cell_t *IDX( cell_t *bmap, int n, int i, int j ) {
	return bmap + i*n + j;
}

/**
 * Returns a pointer to the cell of coordinates (i+R,j+R).
 * This function is usefull for mapping better the indexes.
 */
__host__ __device__ cell_t* MAP(cell_t *bmap, int width, int i, int j, int R){
	return bmap + (i+R)*(width + 2*R) + j + R;
}

void read_ltl( bmap_t *ltl, FILE* f );
void init_map( cell_t *ghost, cell_t *current, int width, int newWidth, int R );
void fill_ghosts_cell( cell_t *ghost, int width, int newWidth, int R );
void processGeneration( int width, int newWidth, cell_t *ghost, int B1, int B2, int D1, int D2, int R, cell_t *temp );
int isCellDead( int index_cond );
int hasEnoughNeighborsToComeToLife( int neighbors, int B1, int B2 );
int hasEnoughNeighborsToSurvive( int neighbors, int D1, int D2 );
void makeCellAlive( int i, int j, int width, int R, cell_t *next );
void makeCellDead( int i, int j, int width, int R, cell_t *next );
void write_ltl( cell_t *ltl, FILE *f, int width, int newWIdth, int R ); 

/**
 * Read a PBM file from file f. The caller is responsible for passing
 * a pointer f to a file opened for reading. This function is not very
 * robust; it may fail on perfectly legal PBM images, but should work
 * for the images produced by gen-input.c. Also, it should work with
 * PBM images produced by Gimp (you must save them in "ASCII format"
 * when prompted).
 */
void read_ltl( bmap_t *ltl, FILE* f ) {
	char buf[2048]; 
	char *s; 
	int n, i, j;
	int width, height;

	/* Get the file type (must be "P1") */
	s = fgets(buf, sizeof(buf), f);
	if (0 != strcmp(s, "P1\n")) {
		fprintf(stderr, "FATAL: Unsupported file type \"%s\"\n", buf);
		exit(-1);
	}
	/* Get any comment and ignore it; does not work if there are
		 leading spaces in the comment line */
	do {
		s = fgets(buf, sizeof(buf), f);
	} while (s[0] == '#');
	/* Get width, height; since we are assuming square images, we
		 reject the input if width != height. */
	sscanf(s, "%d %d", &width, &height);
	if ( width != height ) {
		fprintf(stderr, "FATAL: image width (%d) and height (%d) must be equal\n", width, height);
		exit(-1);
	}        
	ltl->n = n = width;
	ltl->bmap = (cell_t*)malloc( n * n * sizeof(cell_t));
	/* scan bitmap; each pixel is represented by a single numeric
		 character ('0' or '1'); spaces and other separators are ignored
		 (Gimp produces PBM files with no spaces between digits) */
	for (i=0; i<n; i++) {
		for (j=0; j<n; j++) {
			int val;
			do {
				val = fgetc(f);
				if ( EOF == val ) {
					fprintf(stderr, "FATAL: error reading input\n");
					exit(-1);
				}
			} while ( !isdigit(val) );
			*IDX(ltl->bmap, n, i, j) = (val - '0');
		}
	}
}


/**
 * Initialize new grid having empty ghost cells.
 */
void init_map( cell_t *ghost, cell_t *current, int width, int newWidth, int R ) {
	int i,j;
	
	for(i=0; i<width; i++) {
		for(j=0; j<width; j++) {
			ghost[(i + R)*newWidth + j + R] = current[i*width + j];
		}
	}
}

/**
 * Fill ghost cells rows in the new grid.
 */
__global__ void fill_ghost_rows_cells( cell_t *ghost, int width, int R ) {
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y;
	
	if (x < width){
		/* fill top row */
		*MAP(ghost, width, width + y, x, R) = *MAP(ghost, width, y, x, R);
		/* fill bottom row */
		*MAP(ghost, width, - R + y, x, R) = *MAP(ghost, width, width - R + y, x, R);
	}
}

/**
 * Fill ghost cells columns in the new grid.
 */
__global__ void fill_ghost_columns_cells( cell_t *ghost, int width, int R ){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = threadIdx.y;
	
  if (x < width){
    *MAP(ghost, width, x, width + y, R) = *MAP(ghost, width, x, y, R);
    *MAP(ghost, width, x, - R + y, R) = *MAP(ghost, width, x, width - R + y, R);
  }
  if (x < R){
    *MAP(ghost, width, - R + x, width + y, R) = *MAP(ghost, width, - R + x, y, R);

    *MAP(ghost, width, - R + x, - R + y, R) = *MAP(ghost, width, - R + x, width - R + y, R);
  }
  if (x >= width - R){
    *MAP(ghost, width, x + R, width + y, R) = *MAP(ghost, width, x + R, y, R);
    *MAP(ghost, width, x + R, - R + y, R) = *MAP(ghost, width, x + R, width - R + y,  R);
  }
}

/**
 * Compute the next grid given the current configuration.
 * Updates are written to a temporary grid and then rewrite into original grid.
 */
__global__ void processGeneration( cell_t *ghost, cell_t *next, int B1, int B2, int D1, int D2, int width, int newWidth, int R ) {
	int global_i, global_j, local_i, local_j;
	int neighbors = 0;
	int cell_cond;	
	/* Declare the shared memory */
	extern __shared__ cell_t s_grid[];

	global_j = threadIdx.x + blockIdx.x*blockDim.x;
	global_i = threadIdx.y + blockIdx.y*blockDim.y;
	local_j = threadIdx.x + R;
	local_i = threadIdx.y + R;
	if (global_i < newWidth && global_j < newWidth){
		s_grid[local_i*(BLKSIZE + 2*R) + local_j] = *MAP(ghost, width, global_i, global_j, R);
		if (local_i < 2*R){
			s_grid[(local_i - R)*(BLKSIZE + 2*R) + local_j] = *MAP(ghost, width, global_i - R , global_j, R); 						//top
			s_grid[(local_i + BLKSIZE)*(BLKSIZE + 2*R) + local_j] = *MAP(ghost, width, BLKSIZE + global_i, global_j, R); //bottom
		}
		if (local_j < 2*R){
			s_grid[local_i*(BLKSIZE + 2*R) + 0 - R + local_j] = *MAP(ghost, width, global_i, 0 - R + global_j, R);          // left
			s_grid[local_i*(BLKSIZE + 2*R) + BLKSIZE + local_j] = *MAP(ghost, width, global_i, BLKSIZE + global_j, R);     //right
		}
		if (local_i < 2*R && local_j < 2*R){
			s_grid[(local_i - R)*(BLKSIZE + 2*R) + local_j - R] = *MAP(ghost, width, global_i - R, global_j - R, R);                          //top-left
			s_grid[(local_i - R)*(BLKSIZE + 2*R) + BLKSIZE + local_j] = *MAP(ghost, width, global_i - R, BLKSIZE + global_j, R);              //top-right
			s_grid[(BLKSIZE + local_i)*(BLKSIZE + 2*R) + local_j - R] = *MAP(ghost, width, BLKSIZE + global_i , global_j - R, R);             //bottom-left
			s_grid[(BLKSIZE + local_i)*(BLKSIZE + 2*R) + BLKSIZE + local_j] = *MAP(ghost, width, BLKSIZE + global_i, BLKSIZE + global_j, R);  //bottom-right
		}
	}
	__syncthreads();

	if (global_i < width + R && global_j < width + R){
		for(int i = local_i - R; i <= local_i + R; i++){
			for(int j = local_j - R; j <= local_j + R ; j++){
				/* count neighbors */
				if(s_grid[i*(BLKSIZE + 2*R) + j] == 1) {
					neighbors++;
				}
			}
		}

		cell_cond = ghost[(newWidth*(global_i + R) + global_j + R)];
		/* apply rules of the larger than life to cell (i, j) */
		if (isCellDead(cell_cond) && hasEnoughNeighborsToComeToLife(neighbors, B1, B2)) {
			makeCellAlive(global_i, global_j, width, R, next);			
		} else if (!isCellDead(cell_cond) && hasEnoughNeighborsToSurvive(neighbors, D1, D2)) {
			makeCellAlive(global_i, global_j, width, R, next);
		} else {
			makeCellDead(global_i, global_j, width, R, next);
		}
	}

}


/**
 * Write the content of the bmap_t structure pointed to by ltl to the
 * file f in PBM format. The caller is responsible for passing a
 * pointer f to a file opened for writing.
 */
void write_ltl( cell_t *ltl, FILE *f, int width, int newWidth, int R ) {
	int i, j;
	const int n = width;
	
	fprintf(f, "P1\n");
	fprintf(f, "# produced by ltl\n");
	fprintf(f, "%d %d\n", n, n);
	for (i=0; i<n; i++) {
		for (j=0; j<n; j++) {
			fprintf(f, "%d ", ltl[newWidth*(R + i) + R + j]);
		}
		fprintf(f, "\n");
	}
}

int main( int argc, char* argv[] ) {
	int R, B1, B2, D1, D2, nsteps, width, newWidth;
	int inc = 0;
	int i;
	const char *infile, *outfile;
	FILE *in, *out;
	bmap_t cur;
	double tstart, tstop;	
	
	cell_t *ghost;
	
	cell_t *d_ghost;
	cell_t *d_next;
	cell_t *d_temp;
	
	if ( argc != 9 ) {
		fprintf(stderr, "Usage: %s R B1 B2 D1 D2 nsteps infile outfile\n", argv[0]);
		return -1;
  }
	R = atoi(argv[1]);
	B1 = atoi(argv[2]);
	B2 = atoi(argv[3]);
	D1 = atoi(argv[4]);
	D2 = atoi(argv[5]);
	nsteps = atoi(argv[6]);
	infile = argv[7];
	outfile = argv[8];

	assert(  R <= 8  );
	assert(  0 <= B1 );
	assert( B1 <= B2 );
	assert(  1 <= D1 );
	assert( D1 <= D2 );
		
	in = fopen(infile, "r");
	if (in == NULL) {
		fprintf(stderr, "FATAL: can not open \"%s\" for reading\n", infile);
		exit(-1);
	}
	read_ltl(&cur, in);
	fclose(in);
	fprintf(stderr, "Size of input image: %d x %d\n", cur.n, cur.n);
	fprintf(stderr, "Model parameters: R=%d B1=%d B2=%d D1=%d D2=%d nsteps=%d\n",
					R, B1, B2, D1, D2, nsteps);
				
	width = cur.n;
	newWidth = width + 2*R;

	const size_t new_size = newWidth*newWidth*sizeof(cell_t);
	/* Useful for create general solutions having general inputs */
	inc = (width % BLKSIZE) > 0 ? 1 : 0;

	ghost = (cell_t*)malloc(new_size);
	
	/* Allocate space for device copies */
	cudaMalloc((void **)&d_ghost, new_size);
	cudaMalloc((void **)&d_next, new_size); 
	
	/* Setting up the thread blocks */
	dim3 block(BLKSIZE, BLKSIZE);
	dim3 grid(width/BLKSIZE + inc, width/BLKSIZE + inc);
	dim3 copyBlock(BLKSIZE, R);
	dim3 copyGrid(width/BLKSIZE + inc);

	out = fopen(outfile, "w");
	
	init_map(ghost, cur.bmap, width, newWidth, R);
	/* Copy input to device */
	cudaMemcpy(d_ghost, ghost, new_size, cudaMemcpyHostToDevice);
    
	tstart = hpc_gettime();
    
	for (i = 0; i<nsteps; i++) {
		fill_ghost_rows_cells<<<copyGrid, copyBlock>>>(d_ghost, width, R);
    fill_ghost_columns_cells<<<copyGrid, copyBlock>>>(d_ghost, width, R);		
		processGeneration<<<grid, block, (BLKSIZE + 2*R)*(BLKSIZE + 2*R)*sizeof(cell_t)>>>(d_ghost, d_next, B1, B2, D1, D2, width, newWidth, R);		
		cudaDeviceSynchronize();
		cudaMemcpy(ghost, d_next, new_size, cudaMemcpyDeviceToHost);
		d_temp = d_ghost;
		d_ghost = d_next;
		d_next = d_temp;
	}
	/* Wait for kernel to finish */
  cudaDeviceSynchronize();
	tstop = hpc_gettime();
	printf("Elapsed time %f\n", tstop - tstart);	
	// Copy back results
	cudaMemcpy(ghost, d_ghost, new_size, cudaMemcpyDeviceToHost);
	if ( out == NULL ) {
		fprintf(stderr, "FATAL: can not open \"%s\" for writing", outfile);
		exit(-1);
	}
	write_ltl(ghost, out, width, newWidth, R);		
	fclose(out);	
	/* Cleanup */
	cudaFree(d_ghost); 
	cudaFree(d_next); 
	free(cur.bmap);
	free(ghost);
	return 0;
}

/**
 * Check if cell is 0.
 */
 __device__ int isCellDead(int cell_cond) { 
	return cell_cond == 0;
}

/**
 * Check if a dead cell has enough neighbors to come to life. 
 */
 __device__ int hasEnoughNeighborsToComeToLife( int neighbors, int B1, int B2 ) { 
	return (neighbors <= B2 && neighbors >= B1); 
}

/** 
 * Check if an alive cell has enough neighbors to survive.
 */
 __device__ int hasEnoughNeighborsToSurvive( int neighbors, int D1, int D2 ) {
	return (neighbors <= D2 && neighbors >= D1);	
}

/** 
 * Make the cell alive.
 */
 __device__ void makeCellAlive( int global_i, int global_j, int width, int R, cell_t *next ) { 
	*MAP(next, width, global_i, global_j, R) = 1; 
}

/**
 * Make the cell dead.
 */
 __device__ void makeCellDead( int global_i, int global_j, int width, int R, cell_t *next ) {
	*MAP(next, width, global_i, global_j, R) = 0;
}
