/* This project is based on COMS30006 - Advanced High Performance Computing 
   - Lattice Boltzmann in the University of Bristol. The project is been 
   modified to solve open boundaries for pipe flow and be more friendly 
   to CS110 students in ShanghaiTech University by Lei Jia in 2023.
*/

#include "types.h"
#include "utils.h"
#include "calc.h"
#include "d2q9_bgk.h"
#include <chrono>

void print_stream_and_boundary(const t_param& params, t_speed* cells, unsigned int speed) {
  std::cout << "Current speed is " << speed << std::endl;
  for(int jj = params.ny - 1 ; jj >= 0 ; --jj) {
    for(int ii = 0 ; ii < params.nx ; ++ii) {
      speed_identifier identifer {cells[ii + jj * params.nx].speeds[speed]};
      printf("|(%u, %u, %u)", identifer.get_x(), identifer.get_y(), identifer.get_speed());
    }
    printf("|\n");  
  }
} 

/* output usage examples */
void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile> [output_directory]\n", exe);
  exit(EXIT_FAILURE);
}

void print_speed(const t_param& params, t_speed* speeds) {
    for(int i = 0 ; i < 9 ; ++i) {
      printf("speeds: %d\n", i);
      for(int yy = params.ny - 1 ; yy >= 0 ; --yy) {
        for(int xx = 0 ; xx < params.nx ; ++xx) {
          printf("%.5f ", speeds[yy * params.nx + xx].speeds[i]);
        }
        printf("\n");
      }
    }
}

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
  char*    paramfile = nullptr;    /* name of the input parameter file */
  char*    obstaclefile = nullptr; /* name of a the input obstacle file */
  char*    out_dir = nullptr;      /* name of output directory */
  t_param  params;              /* struct to hold parameter values */
  t_speed* cells     = nullptr;    /* grid containing fluid densities */
  t_speed* tmp_cells = nullptr;    /* scratch space */
  int*     obstacles = nullptr;    /* grid indicating which cells are blocked */
  float*   inlets    = nullptr;    /* inlet velocity */  
  char buf[128];                /* a string buffer for specific filename */

  /* parse the command line */
  if (argc < 3)
  {
    usage(argv[0]);
  }
  else
  {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }
  /* set the output directory */
  if (argc >= 4)
    out_dir = argv[3];
  else
    out_dir = (char*)"./results";

  /* Display load parameters */
  printf("==load==\n");
  printf("Params file:   %s\n", paramfile);
  printf("Obstacle file: %s\n", obstaclefile);
  printf("Out directory: %s\n", out_dir);
  /* Total/init time starts here */
  auto t_start = std::chrono::high_resolution_clock::now();
  /* initialise our data structures and load values from file */
  initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &inlets);
  /* Set the inlet speed */
  set_inlets(params, inlets);
  /* Init time stops here */
  auto t_init = std::chrono::high_resolution_clock::now();
  auto init_duration = std::chrono::duration_cast<std::chrono::milliseconds>(t_init - t_start);
  /* Display simulation parameters */
  printf("==start==\n");
  printf("Number of cells:\t\t\t%d (%d x %d)\n",params.nx*params.ny,params.nx,params.ny);
  printf("Max iterations:\t\t\t\t%d\n", params.maxIters);
  printf("Density:\t\t\t\t%.6lf\n", params.density);
  printf("Kinematic viscosity:\t\t\t%.6lf\n", params.viscosity);
  printf("Inlet velocity:\t\t\t\t%.6lf\n", params.velocity);
  printf("Inlet type:\t\t\t\t%d\n", params.type);
  printf("Relaxtion parameter:\t\t\t%.6lf\n", params.omega);

  /* Compute time starts */
  t_start = std::chrono::high_resolution_clock::now();

  /* timestep loop */
  for (int tt = 0; tt < params.maxIters; tt++)
  {
    timestep(params, cells, tmp_cells, inlets, obstacles);

  /* Visualization */
#ifdef VISUAL
    if (tt % 1000 == 0) {
      sprintf(buf, "%s/visual/state_%d.dat", out_dir , tt / 1000);
      write_state(buf, params, cells, obstacles);
    }
#endif
  }


  /* Compute time stops here */
  auto t_comp = std::chrono::high_resolution_clock::now();
  auto comp_duration = std::chrono::duration_cast<std::chrono::milliseconds>(t_comp - t_start);
  
  /* write final state and free memory */
  sprintf(buf, "%s/final_state.dat", out_dir);
  write_state(buf, params, cells, obstacles);

  /* Display Reynolds number and time */
  printf("==done==\n");
  printf("Reynolds number:\t\t\t%.12E\n", calc_reynolds(params, cells, obstacles));
  printf("Average velocity:\t\t\t%.12E\n", av_velocity(params, cells, obstacles));
  printf("Elapsed Init time:\t\t\t%llu (ms)\n",    init_duration.count());
  printf("Elapsed Compute time:\t\t\t%llu (ms)\n", comp_duration.count());
  //print_speed(params, cells);

  /* finalise */
  finalise(&params, &cells, &tmp_cells, &obstacles, &inlets);

  return EXIT_SUCCESS;
}


