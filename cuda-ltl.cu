#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <ctype.h> /* for isdigit */

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

/* Returns a pointer to the cell of coordinates (i+R,j+R). */
__device__ __host__ cell_t* MAP(cell_t *bmap, int width, int i, int j, int R){
	return bmap + (i+R)*(width + 2*R) + j + R;
}

void read_ltl( bmap_t *ltl, FILE* f );
void init_map( cell_t *ghost, cell_t *current, int width, int newWidth, int R );
void fill_ghosts_cell( cell_t *ghost, int width, int newWidth, int R );
void processGeneration( int newWidth, cell_t *ghost, int B1, int B2, int D1, int D2, int R, cell_t *temp );
int countNeighbors( int row, int col, int R, int newWidth, cell_t *ghost );
int isCellDead( int i, int j, int newWidth, cell_t *ghost );
int hasEnoughNeighborsToComeToLife( int neighbors, int B1, int B2 );
int hasEnoughNeighborsToSurvive( int neighbors, int D1, int D2 );
void makeCellAlive( int i, int j, int newWidth, cell_t *temp );
void makeCellDead( int i, int j, int newWidth, cell_t *temp );
void write_ltl( bmap_t *ltl, FILE *f );
void final_map( cell_t *ghost, cell_t *final, int width, int newWidth, int R ); 


/**
 * Read a PBM file from file f. The caller is responsible for passing
 * a pointer f to a file opened for reading. This function is not very
 * robust; it may fail on perfectly legal PBM images, but should work
 * for the images produced by gen-input.c. Also, it should work with
 * PBM images produced by Gimp (you must save them in "ASCII format"
 * when prompted).
 */
__global__ void read_ltl( bmap_t *ltl, FILE* f ) {
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
void init_map( cell_t *ghost, cell_t *current, int width, int newWidth, int R, int n ) {
	int i,j;
	
	for (i=0; i<width; i++) {
		for (j=0; j<width; j++) {
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
		*MAP(ghost, width, width + y, x, R) = *MAP(ghost, width, 0 + y, x, R);
		*MAP(ghost, width, 0 - R + y, x, R) = *MAP(ghost, width, width - R + y, x, R);
	}
}

/**
 * Fill ghost cells columns in the new grid.
 */
__global__ void fill_ghost_columns_cells( cell_t *ghost, int width, int R ){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = threadIdx.y;
	
  if (x < width){
    *MAP(ghost, width, x, width + y, R) = *MAP(ghost, width, x, 0 + y, R);
    *MAP(ghost, width, x, 0 - R + y, R) = *MAP(ghost, width, x, width - R + y, R);
  }
  if (x < R){
    *MAP(ghost, width, 0 - R + x, width + y, R) = *MAP(ghost, width, 0 - R + x, 0 + y, R);
    *MAP(ghost, width, 0 - R + x, 0 - R + y, R) = *MAP(ghost, width, 0 - R + x, width - R + y, R);
  }
  if (x >= width - R){
    *MAP(ghost, width, x + R, width + y, R) = *MAP(ghost, width, x + R, 0 + y, R);
		*MAP(ghost, width, x + R, 0 - R + y, R) = *MAP(ghost, width, x + R, width - R + y,  R);
  }
}

/**
 * Compute the next grid given the current configuration.
 * Updates are written to a temporary grid and then rewrite into original grid.
 */
__global__ void processGeneration( int width, int newWidth, cell_t *ghost, int B1, int B2, int D1, int D2, int R, cell_t *temp ) {
	int gi, gj, li, lj, neighbors = 0;	
	// Declare the shared memory on a per block level
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
		neighbors = countNeighbors(i, j, R, newWidth, shared_grid);
		for(int i = local_i - R; i <= local_i + R; i++){
			for(int j = local_j - R; j <= local_j + R ; j++){
				neighbors = countNeighbors(i, j, R, newWidth, s_grid);
			}
		}
		/* apply rules of the larger than life to cell (i, j) */
		if (isCellDead(global_i, global_j, newWidth, s_grid) && hasEnoughNeighborsToComeToLife(neighbors, B1, B2)) {
			makeCellAlive(global_i, global_j, newWidth, temp);			
		} else if (!isCellDead(global_i, global_j, newWidth, s_grid) && hasEnoughNeighborsToSurvive(neighbors, D1, D2)) {
			makeCellAlive(global_i, global_j, newWidth, temp);
		} else {
			makeCellDead(global_i, global_j, newWidth, temp);
		}
	}
	__syncthreads();	
	if (global_i < newWidth && global_j < newWidth) {
		ghost[global_i*newWidth + global_j] = s_grid[global_i*newWidth + global_j];
	}
}

/**
 * Check if cell is 0.
 */
__device__ int isCellDead( int i, int j, int newWidth, cell_t *s_grid ) { 
	return s_grid[i*newWidth + j] == 0;
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
	return (neighbors <= D2-1 && neighbors >= D1-1);	
}

/** 
 * Make the cell alive.
 */
__device__ void makeCellAlive( int i, int j, int newWidth, cell_t *temp ) { 
	temp[i*newWidth + j] = 1; 
}

/**
 * Make the cell dead.
 */
__device__ void makeCellDead( int i, int j, int newWidth, cell_t *temp ) {
	temp[i*newWidth + j] = 0;
}

/**
 * Write the content of the bmap_t structure pointed to by ltl to the
 * file f in PBM format. The caller is responsible for passing a
 * pointer f to a file opened for writing
 */
__global__ void write_ltl( bmap_t *ltl, FILE *f ) {
	int i, j;
	const int n = ltl->n;
	
	fprintf(f, "P1\n");
	fprintf(f, "# produced by ltl\n");
	fprintf(f, "%d %d\n", n, n);
	for (i=0; i<n; i++) {
		for (j=0; j<n; j++) {
			fprintf(f, "%d ", *IDX(ltl->bmap, n, i, j));
		}
		fprintf(f, "\n");
	}
}

/**
 * Initialize the final grid having correct values (without ghost cells).
 */
__global__ void final_map( cell_t *ghost, cell_t *final, int width, int newWidth, int R ) {
	int i,j;
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if(id<newWidth-R) {
		final[id] = ghost[(blockIdx.x + R)*blockIdx.x + threadIdx.x + R];
	}
}


#define BLKSIZE 32

int main( int argc, char* argv[] ) {
	int R, B1, B2, D1, D2, nsteps, width, newWidth;
	int i;
	int inc;
	const char *infile, *outfile;
	FILE *in, *out;
	bmap_t cur;
	double tstart, tstop;	
	
	cell_t *current;
	cell_t *ghost;
	cell_t *temp;
	cell_t *final;
	
	cell_t *d_current;
	cell_t *d_ghost;
	cell_t *d_temp;
	cell_t *d_final;
	
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

	const size_t size = width*width*sizeof(cell_t);
	const size_t new_size = newWidth*newWidth*sizeof(cell_t);
	inc = (width % BLKSIZE) > 0 ? 1 : 0;

	current = (cell_t*)malloc(size));
	ghost = (cell_t*)malloc(new_size);
	temp = (cell_t*)malloc(new_size);
	final = (cell_t*)malloc(size);
	
	/* Allocate space for device copies of current, ghost, temp, final */
	cudaMalloc((void **)&d_current, size);
	cudaMalloc((void **)&d_ghost, new_size);
	cudaMalloc((void **)&d_temp, new_size);
	cudaMalloc((void **)&d_final, size);
 
	/* Setup input value */
	current = cur.bmap; 
	
	dim3 block(BLKSIZE, BLKSIZE);
	dim3 grid(width/BLKSIZE + inc, width/BLKSIZE + inc);
	dim3 copyBlock(BLKSIZE, R);
	dim3 copyGrid(width/BLKSIZE + inc);

	out = fopen(outfile, "w");
	
	init_map(ghost, current, width, newWidth, R);
	/* Copy input to device */
	cudaMemcpy(d_current, &current, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_ghost, &ghost, new_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_temp, &temp, new_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_final, &final, size, cudaMemcpyHostToDevice);
    
	tstart = hpc_gettime();
    
	for (i = 0; i<nsteps; i++) {
		fill_ghost_rows_cells<<<copyGrid, copyBlock>>>(d_ghost, width, R);
    fill_ghost_columns_cells<<<copyGrid, copyBlock>>>(d_cur_ghost, width, R);
		processGeneration<<<grid, block, (BLKSIZE + 2 * R)*(BLKSIZE + 2 * R)*sizeof(cell_t)>>>(newWidth, d_ghost, B1, B2, D1, D2, R, d_temp);
		cudaDeviceSynchronize();
	}
  final_map<<<grid,block>>>(d_ghost, d_final, width, newWidth, R);
  cudaDeviceSynchronize(); /* wait for kernel to finish */
	tstop = hpc_gettime();
	printf("Elapsed time %f\n", tstop - tstart);	
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess)
		printf("CUDA error %s\n",cudaGetErrorString(error));
	// Copy back results
	cudaMemcpy(&final, d_final, size, cudaMemcpyDeviceToHost);
	if ( out == NULL ) {
		fprintf(stderr, "FATAL: can not open \"%s\" for writing", outfile);
		exit(-1);
	}
	cur.bmap = final;
	write_ltl(&cur, out);		
	fclose(out);	
	/* Cleanup */
	cudaFree(d_current); 
	cudaFree(d_ghost); 
	cudaFree(d_temp);
	cudaFree(d_final);
	free(cur.bmap);
	return 0;
}