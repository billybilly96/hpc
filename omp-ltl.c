#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <omp.h>
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
void count_total_alive_cells( cell_t *final, int width);

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
	
	#pragma omp parallel for collapse(2) default(none) shared(width,newWidth,R,ghost,current) schedule(static,width)
	for (i=0; i<width; i++) {
		for (j=0; j<width; j++) {
			ghost[(i + R)*newWidth + j + R] = current[i*width + j];
		}
	}
}

/**
 * Fill ghost cells in the new grid.
 */
void fill_ghost_cells( cell_t *ghost, int width, int newWidth, int R ) {
	int i,j;
	
	#pragma omp parallel default(none) shared(width,newWidth,R,ghost)
	{
		#pragma omp for collapse(2)
		for (i=0; i<R; i++) {
			for (j=0; j<width; j++) {
				/* fill top rows */
				ghost[newWidth*i + R + j] = ghost[newWidth*(width + i) + R + j];
				/* fill bottom rows */
				ghost[newWidth*(newWidth - R + i) + R + j] = ghost[newWidth*(R + i) + R + j];
			}
		}
		#pragma omp for collapse(2)
		for (i=0-R; i<width+R; i++) {
			for (j=0; j<R; j++) {
				/* fill right columns */
				ghost[newWidth*(i + R) + newWidth - R + j] = ghost[newWidth*(i+R) + R + j];
				/* fill left columns */
				ghost[(i + R)*newWidth + j] = ghost[(i + R)*newWidth + width + j];
			}
		}
	}
	#pragma omp barrier
}

/**
 * Compute the next grid given the current configuration.
 * Updates are written to a temporary grid and then rewrite into original grid.
 */
void processGeneration( int newWidth, cell_t *ghost, int B1, int B2, int D1, int D2, int R, cell_t *temp ) {
	int i,j,neighbors;
	
	#pragma omp parallel default(none) private(i,j,neighbors) shared(newWidth,R,ghost,temp,B1,B2,D1,D2)
	{
		#pragma omp for collapse(2) schedule(static,newWidth)     
		for(i = 0; i < newWidth; i++) {
			for(j = 0; j < newWidth; j++) {
				neighbors = countNeighbors(i, j, R, newWidth, ghost);
				/* apply rules of the larger than life to cell (i, j) */
				if (isCellDead(i, j, newWidth, ghost) && hasEnoughNeighborsToComeToLife(neighbors, B1, B2)) {
					makeCellAlive(i, j, newWidth, temp);
				} else if (!isCellDead(i, j, newWidth, ghost) && hasEnoughNeighborsToSurvive(neighbors, D1, D2)) {
					makeCellAlive(i, j, newWidth, temp);
				} else {
					makeCellDead(i, j, newWidth, temp);
				}
			}
		}
		#pragma omp for collapse(2) schedule(static,newWidth) 
		for(i = 0; i < newWidth; i++) {
			for(j = 0; j < newWidth; j++) {
				ghost[i*newWidth + j] = temp[i*newWidth + j];
			}
		}
	}
	#pragma omp barrier
}

/**
 * Count alive neighbors of cell (row, col).
 */
int countNeighbors( int row, int col, int R, int newWidth, cell_t *ghost ) {
	int neighbors = 0;
	int i,j;
	
	for(i = row - R; i <= row + R; i++) {
		for(j = col - R; j <= col + R; j++) {
			if ((i*newWidth + j) != (row*newWidth + col) && i > -1 && j > -1 && i < newWidth && j < newWidth) {
					neighbors += ghost[i*newWidth + j];
			}
		}
	}
	return neighbors;
}

/**
 * Check if cell is 0.
 */
int isCellDead( int i, int j, int newWidth, cell_t *ghost ) { 
	return ghost[i*newWidth + j] == 0;
}

/**
 * Check if a dead cell has enough neighbors to come to life. 
 */
int hasEnoughNeighborsToComeToLife( int neighbors, int B1, int B2 ) { 
	return (neighbors <= B2 && neighbors >= B1);																								 
}

/** 
 * Check if an alive cell has enough neighbors to survive.
 */
int hasEnoughNeighborsToSurvive( int neighbors, int D1, int D2 ) {
	return (neighbors+1 <= D2 && neighbors+1 >= D1);	
}

/** 
 * Make the cell alive.
 */
void makeCellAlive( int i, int j, int newWidth, cell_t *temp ) { 
	temp[i*newWidth + j] = 1; 
}

/**
 * Make the cell dead.
 */
void makeCellDead( int i, int j, int newWidth, cell_t *temp ) {
	temp[i*newWidth + j] = 0;
}

/**
 * Write the content of the bmap_t structure pointed to by ltl to the
 * file f in PBM format. The caller is responsible for passing a
 * pointer f to a file opened for writing
 */
void write_ltl( bmap_t *ltl, FILE *f ) {
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
void final_map( cell_t *ghost, cell_t *final, int width, int newWidth, int R ) {
	int i,j;
	
	for (i=0; i<newWidth-R; i++) {
		for (j=0; j<newWidth-R; j++) {
			final[i*width + j] = ghost[(i + R)*newWidth + j + R];
		} 
	}
}


/**
 * Count alive cells in final configuration.
 * Used for testing purpose.
 */
/*
void count_total_alive_cells( cell_t *final, int width ) {
	int i,j;
	int tot = 0;
	
	for (i=0; i<width; i++) {
		for (j=0; j<width; j++) {
			tot += final[i*width + j];
		}
	}
	printf("Total Alive: %d\n", tot);
}
*/

int main( int argc, char* argv[] ) {
	int R, B1, B2, D1, D2, nsteps, width, newWidth;
	int i;
	const char *infile, *outfile;
	FILE *in, *out;
	bmap_t cur;
	double tstart, tstop;	
	cell_t *current, *ghost, *temp, *final;
	
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
	printf("Max threads : %d\n", omp_get_max_threads());	
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
	current = (cell_t*)malloc(size);
	ghost = (cell_t*)malloc(new_size);
	temp = (cell_t*)malloc(new_size);
	final = (cell_t*)malloc(size);
	current = cur.bmap;  
	out = fopen(outfile, "w");
	tstart = omp_get_wtime();
	init_map(ghost, current, width, newWidth, R);			
	for (i = 0; i<nsteps; i++) {
		fill_ghost_cells(ghost, width, newWidth, R);
		processGeneration(newWidth, ghost, B1, B2, D1, D2, R, temp);
	}	
	final_map(ghost, final, width, newWidth, R);	
	tstop = omp_get_wtime();
	printf("Elapsed time %f\n", tstop - tstart);	
	count_total_alive_cells(final, width);	
	cur.bmap = final;	
	if ( out == NULL ) {
		fprintf(stderr, "FATAL: can not open \"%s\" for writing", outfile);
		exit(-1);
	}	
	write_ltl(&cur, out);	
	fclose(out);
	free(cur.bmap);
	return 0;
}
