#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <ctype.h> /* for isdigit */

#define BLOCK_SIZE 32

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
__global__ void init_map( cell_t *ghost, cell_t *current, int width, int newWidth, int R, int n ) {
    int i = threadIdx.y + blockIdx.y*blockDim.y;
	int j = threadIdx.x + blockIdx.x*blockDim.x;	 
    if(i < n && j < n) {
        ghost[(i + R)*newWidth + j + R] = current[i*width + j];
    }
}

/**
 * Fill ghost cells in the new grid.
 */
__global__ void fill_ghost_cells( cell_t **ghost, int *width, int *newWidth, int *R, int n ) {
    int i = threadIdx.y + blockIdx.y*blockDim.y; 
    int j = threadIdx.x + blockIdx.x*blockDim.x;

    if (i < n && j < n) {
        ghost[newWidth*i + R + j] = ghost[newWidth*(width + i) + R + j];			   //top
		ghost[newWidth*(newWidth - R + i) + R + j] = ghost[newWidth*(R + i) + R + j];  //bottom
    } 
    if (i < n && j < n) {
        ghost[newWidth*(i + R) + newWidth - R + j] = ghost[newWidth*(i+R) + R + j];    //right
        ghost[(i + R)*newWidth + j] = ghost[(i + R)*newWidth + width + j];			   //left  
    }
}

/**
 * Compute the next grid given the current configuration.
 * Updates are written to a temporary grid and then rewrite into original grid.
 */
__gloabl__ void processGeneration( int newWidth, cell_t *ghost, int B1, int B2, int D1, int D2, int R, cell_t *temp ) {
	int i,j;
    int iy = (blockDim.y - R) * blockIdx.y + threadIdx.y;
    int ix = (blockDim.x - R) * blockIdx.x + threadIdx.x;
    int id = iy * (newWidth) + ix;

    int i = threadIdx.y;
    int j = threadIdx.x;
    int neighbors;

    // Declare the shared memory on a per block level
    __shared__ cell_t shared_grid[BLOCK_SIZE_y][BLOCK_SIZE_x];

    // Copy cells into shared memory
    if (ix < newWidth && iy < newWidth) {
        shared_grid[i][j] = ghost[id];
    }
 
    //Sync all threads in block
    __syncthreads();

    if (iy < newWidth && ix < newWidth) {
        if(i != 0 && i !=blockDim.y && j != 0 && j !=blockDim.x) {
            neighbors = countNeighbors(i, j, R, newWidth, shared_grid);
            /* apply rules of the larger than life to cell (i, j) */
            if (isCellDead(i, j, newWidth, shared_grid) && hasEnoughNeighborsToComeToLife(neighbors, B1, B2)) {
			    makeCellAlive(i, j, newWidth, temp);
			} else if (!isCellDead(i, j, newWidth, shared_grid) && hasEnoughNeighborsToSurvive(neighbors, D1, D2)) {
                makeCellAlive(i, j, newWidth, temp);
			} else {
			    makeCellDead(i, j, newWidth, temp);
			}
        }
    }
    
    if (ix < newWidth && iy < newWidth) {
        ghost[id] = shared_grid[i][j];
    }

}

/**
 * Count alive neighbors of cell (row, col).
 */
__device__ int countNeighbors( int row, int col, int R, int newWidth, cell_t *shared_grid ) {
    int neighbors = 0;
    int i,j;
    for(i = row - R; i < row + R + 1; i++) {
        for(j = col - R; j < col + R + 1; j++) {
            if ((i != row && j!= col) && i > -1 && j > -1 && i < newWidth && j < newWidth) {
                neighbors += shared_grid[i][j];
            }
        }
    }
    return neighbors;
}

/**
 * Check if cell is 0.
 */
__device__ int isCellDead( int i, int j, int newWidth, cell_t *shared_grid ) { 
	return shared_grid[i][j] == 0;
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


#define BLKSIZE 512
int main( int argc, char* argv[] ) {
	int R, B1, B2, D1, D2, nsteps;
	int width, newWidth;
	int i;
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

	current = (cell_t*)malloc( width * width * sizeof(cell_t));
	ghost = (cell_t*)malloc( newWidth * newWidth * sizeof(cell_t));
	temp = (cell_t*)malloc( newWidth * newWidth * sizeof(cell_t));
	final = (cell_t*)malloc( width * width * sizeof(cell_t));
	
	/* Allocate space for device copies of current, ghost, temp, final */
	cudaMalloc((void **)&d_current, width * width * sizeof(cell_t));
	cudaMalloc((void **)&d_ghost, newWidth * newWidth * sizeof(cell_t));
	cudaMalloc((void **)&d_temp, newWidth * newWidth * sizeof(cell_t));
	cudaMalloc((void **)&d_final, width * width * sizeof(cell_t));
 
	/* Setup input value */
	current = cur.bmap; 

    cudaFuncSetCacheConfig(processGeneration, cudaFuncCachePreferShared);
	
	/* Copy input to device */
	cudaMemcpy(d_current, &current, width * width * sizeof(cell_t), cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE_x, BLOCK_SIZE_y,1);
    int linGrid_x = (int)ceil(width/(float)(BLOCK_SIZE_x-R));
    int linGrid_y = (int)ceil(width/(float)(BLOCK_SIZE_y-R));
    dim3 gridSize(linGrid_x,linGrid_y,1);
 
    dim3 cpyBlockSize(BLOCK_SIZE_x,1,1);
    dim3 cpyGridSize((int)ceil((newWidth)/(float)cpyBlockSize.x),1,1);

	out = fopen(outfile, "w");
	
	/* Launch init_map() kernel on GPU */
	init_map<<<cpyGridSize,cpyBlockSize>>>(d_ghost, d_current, width, newWidth, R);
    
	tstart = hpc_gettime();
    
	for (i = 0; i<nsteps; i++) {
		fill_ghost_cells<<<cpyGridSize,cpyBlockSize>>>(d_ghost, width, newWidth, R);
		processGeneration<<<gridSize, blockSize>>>(newWidth, d_ghost, B1, B2, D1, D2, R, d_temp);
	}
    final_map<<<cpyGridSize,cpyBlockSize>>>(d_ghost, d_final, width, newWidth, R);
    cudaDeviceSynchronize(); /* wait for kernel to finish */
	tstop = hpc_gettime();

	printf("Elapsed time %f\n", tstop - tstart);
    count_total_alive_cells<<<cpyGridSize,cpyBlockSize>>>(d_final, width);

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
        printf("CUDA error %s\n",cudaGetErrorString(error));
 
    // Copy back results
    cudaMemcpy(&final, d_final, width * width * sizeof(cell_t), cudaMemcpyDeviceToHost);

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
    free(current);
    free(ghost);
    free(temp);
    free(final);

	return 0;
}
