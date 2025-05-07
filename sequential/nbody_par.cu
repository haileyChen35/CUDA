#include <iostream>
#include <fstream>
#include <random>
#include <cmath>
#include <cuda_runtime.h>
#include <chrono>
#include <string>
#include <cstdlib>


double G = 6.674*std::pow(10,-11);
//double G = 1;

struct simulation {
  size_t nbpart;
  
  //host 
  double* hmass;
  //position
  double* hx;
  double* hy;
  double* hz;

  //velocity
  double* hvx;
  double* hvy;
  double* hvz;

  //force
  double* hfx;
  double* hfy;
  double* hfz;

  // device
  double* dmass;
      //position
  double* dx;
  double* dy;
  double* dz;

  //velocity
  double* dvx;
  double* dvy;
  double* dvz;

  //force
  double* dfx;
  double* dfy;
  double* dfz;
  

  
  simulation(size_t nb) : nbpart(nb) {
    
    //allocate host memory
    hmass = new double[nb]();
    hx = new double[nb]();
    hy = new double[nb](); 
    hz = new double[nb]();
    hvx = new double[nb](); 
    hvy = new double[nb](); 
    hvz = new double[nb]();
    hfx = new double[nb](); 
    hfy = new double[nb](); 
    hfz = new double[nb]();

    //allocate device memory
    cudaMalloc(&dmass, nb*sizeof(double));
    cudaMalloc(&dx, nb * sizeof(double));
    cudaMalloc(&dy, nb * sizeof(double));
    cudaMalloc(&dz, nb * sizeof(double));
    cudaMalloc(&dvx, nb * sizeof(double));
    cudaMalloc(&dvy, nb * sizeof(double));
    cudaMalloc(&dvz, nb * sizeof(double));
    cudaMalloc(&dfx, nb * sizeof(double));
    cudaMalloc(&dfy, nb * sizeof(double));
    cudaMalloc(&dfz, nb * sizeof(double));

    //initialize memory 
    cudaMemset(dmass, 0, nb * sizeof(double));
    cudaMemset(dx, 0, nb * sizeof(double));
    cudaMemset(dy, 0, nb * sizeof(double));
    cudaMemset(dz, 0, nb * sizeof(double));
    cudaMemset(dvx, 0, nb * sizeof(double));
    cudaMemset(dvy, 0, nb * sizeof(double));
    cudaMemset(dvz, 0, nb * sizeof(double));
    cudaMemset(dfx, 0, nb * sizeof(double));
    cudaMemset(dfy, 0, nb * sizeof(double));

  }

    //release memory
    ~simulation(){
        delete[] hmass;
        delete[] hx; 
        delete[] hy; 
        delete[] hz;
        delete[] hvx; 
        delete[] hvy; 
        delete[] hvz;
        delete[] hfx; 
        delete[] hfy; 
        delete[] hfz;

        cudaFree(dmass);
        cudaFree(dx);
        cudaFree(dy);
        cudaFree(dz);
        cudaFree(dvx);
        cudaFree(dvy);
        cudaFree(dvz);
        cudaFree(dfx);
        cudaFree(dfy);
        cudaFree(dfz);

    }

    void resize(size_t new_nbpart) {
        if (new_nbpart == nbpart) return;
    
        this->~simulation();
    
        nbpart = new_nbpart;
        hmass = new double[nbpart]();
        hx = new double[nbpart](); 
        hy = new double[nbpart](); 
        hz = new double[nbpart]();
        hvx = new double[nbpart](); 
        hvy = new double[nbpart](); 
        hvz = new double[nbpart]();
        hfx = new double[nbpart](); 
        hfy = new double[nbpart](); 
        hfz = new double[nbpart]();
    
        cudaMalloc(&dmass, nbpart * sizeof(double));
        cudaMalloc(&dx, nbpart * sizeof(double));
        cudaMalloc(&dy, nbpart * sizeof(double));
        cudaMalloc(&dz, nbpart * sizeof(double));
        cudaMalloc(&dvx, nbpart * sizeof(double));
        cudaMalloc(&dvy, nbpart * sizeof(double));
        cudaMalloc(&dvz, nbpart * sizeof(double));
        cudaMalloc(&dfx, nbpart * sizeof(double));
        cudaMalloc(&dfy, nbpart * sizeof(double));
        cudaMalloc(&dfz, nbpart * sizeof(double));
    
        cudaMemset(dmass, 0, nbpart * sizeof(double));
        cudaMemset(dx, 0, nbpart * sizeof(double));
        cudaMemset(dy, 0, nbpart * sizeof(double));
        cudaMemset(dz, 0, nbpart * sizeof(double));
        cudaMemset(dvx, 0, nbpart * sizeof(double));
        cudaMemset(dvy, 0, nbpart * sizeof(double));
        cudaMemset(dvz, 0, nbpart * sizeof(double));
        cudaMemset(dfx, 0, nbpart * sizeof(double));
        cudaMemset(dfy, 0, nbpart * sizeof(double));
        cudaMemset(dfz, 0, nbpart * sizeof(double));
      }
    
    void host_to_device() {
        cudaMemcpy(dmass, hmass, nbpart * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dx, hx, nbpart * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dy, hy, nbpart * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dz, hz, nbpart * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dvx, hvx, nbpart * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dvy, hvy, nbpart * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dvz, hvz, nbpart * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dfx, hfx, nbpart * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dfy, hfy, nbpart * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dfz, hfz, nbpart * sizeof(double), cudaMemcpyHostToDevice);
    }

    void device_to_host() {
      cudaMemcpy(hmass, dmass, nbpart * sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(hx, dx, nbpart * sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(hy, dy, nbpart * sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(hz, dz, nbpart * sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(hvx, dvx, nbpart * sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(hvy, dvy, nbpart * sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(hvz, dvz, nbpart * sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(hfx, dfx, nbpart * sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(hfy, dfy, nbpart * sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(hfz, dfz, nbpart * sizeof(double), cudaMemcpyDeviceToHost);
  }

   
};



void random_init(simulation& s) {
  std::random_device rd;  
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dismass(0.9, 1.0);
  std::normal_distribution<double> dispos(0.0, 1.0);
  std::normal_distribution<double> disvel(0.0, 1.0);

  for (size_t i = 0; i<s.nbpart; ++i) {
    s.hmass[i] = dismass(gen);

    s.hx[i] = dispos(gen);
    s.hy[i] = dispos(gen);
    s.hz[i] = dispos(gen);
    s.hz[i] = 0.;
    
    s.hvx[i] = disvel(gen);
    s.hvy[i] = disvel(gen);
    s.hvz[i] = disvel(gen);
    s.hvz[i] = 0.;
    s.hvx[i] = s.hy[i]*1.5;
    s.hvy[i] = -s.hx[i]*1.5;
    s.hfx[i] = s.hfy[i] = s.hfz[i] = 0.0;

  }
    s.host_to_device();
}

void init_solar(simulation& s) {
  enum Planets {SUN, MERCURY, VENUS, EARTH, MARS, JUPITER, SATURN, URANUS, NEPTUNE, MOON};
  s = simulation(10);

  // Masses in kg
  s.hmass[SUN] = 1.9891 * std::pow(10, 30);
  s.hmass[MERCURY] = 3.285 * std::pow(10, 23);
  s.hmass[VENUS] = 4.867 * std::pow(10, 24);
  s.hmass[EARTH] = 5.972 * std::pow(10, 24);
  s.hmass[MARS] = 6.39 * std::pow(10, 23);
  s.hmass[JUPITER] = 1.898 * std::pow(10, 27);
  s.hmass[SATURN] = 5.683 * std::pow(10, 26);
  s.hmass[URANUS] = 8.681 * std::pow(10, 25);
  s.hmass[NEPTUNE] = 1.024 * std::pow(10, 26);
  s.hmass[MOON] = 7.342 * std::pow(10, 22);

  // Positions (in meters) and velocities (in m/s)
  double AU = 1.496 * std::pow(10, 11); // Astronomical Unit

  s.hx[SUN] = 0; s.hy[SUN] = 0; s.hz[SUN] = 0;
  s.hx[MERCURY] = 0.39*AU; s.hy[MERCURY] = 0; s.hz[MERCURY] = 0;
  s.hx[VENUS] = 0.72*AU; s.hy[VENUS] = 0; s.hz[VENUS] = 0;
  s.hx[EARTH] = 1.0*AU; s.hy[EARTH] = 0; s.hz[EARTH] = 0;
  s.hx[MARS] = 1.52*AU; s.hy[MARS] = 0; s.hz[MARS] = 0;
  s.hx[JUPITER] = 5.20*AU; s.hy[JUPITER] = 0; s.hz[JUPITER] = 0;
  s.hx[SATURN] = 9.58*AU; s.hy[SATURN] = 0; s.hz[SATURN] = 0;
  s.hx[URANUS] = 19.22*AU; s.hy[URANUS] = 0; s.hz[URANUS] = 0;
  s.hx[NEPTUNE] = 30.05*AU; s.hy[NEPTUNE] = 0; s.hz[NEPTUNE] = 0;
  s.hx[MOON] = 1.0*AU + 3.844*std::pow(10, 8); s.hy[MOON] = 0; s.hz[MOON] = 0;

  s.hvx[SUN] = 0; s.hvy[SUN] = 0; s.hvz[SUN] = 0;
  s.hvx[MERCURY] = 0; s.hvy[MERCURY] = 47870; s.hvz[MERCURY] = 0;
  s.hvx[VENUS] = 0; s.hvy[VENUS] = 35020; s.hvz[VENUS] = 0;
  s.hvx[EARTH] = 0; s.hvy[EARTH] = 29780; s.hvz[EARTH] = 0;
  s.hvx[MARS] = 0; s.hvy[MARS] = 24130; s.hvz[MARS] = 0;
  s.hvx[JUPITER] = 0; s.hvy[JUPITER] = 13070; s.hvz[JUPITER] = 0;
  s.hvx[SATURN] = 0; s.hvy[SATURN] = 9680; s.hvz[SATURN] = 0;
  s.hvx[URANUS] = 0; s.hvy[URANUS] = 6800; s.hvz[URANUS] = 0;
  s.hvx[NEPTUNE] = 0; s.hvy[NEPTUNE] = 5430; s.hvz[NEPTUNE] = 0;
  s.hvx[MOON] = 0; s.hvy[MOON] = 29780 + 1022; s.hvz[MOON] = 0;

  for (int i = 0; i < 10; i++) {
    s.hfx[i] = 0;
    s.hfy[i] = 0;
    s.hfz[i] = 0;
    }   

    s.host_to_device();
}

//cuda kernel 
__global__ void update_particles_kernel(double* dx, double* dy, double* dz,
    double* dvx, double* dvy, double* dvz,
    double* dfx, double* dfy, double* dfz,
    double* dmass, size_t nbpart, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nbpart) return;

    dvx[i] += dfx[i] / dmass[i] * dt;
    dvy[i] += dfy[i] / dmass[i] * dt;
    dvz[i] += dfz[i] / dmass[i] * dt;

    dx[i] += dvx[i] * dt;
    dy[i] += dvy[i] * dt;
    dz[i] += dvz[i] * dt;
}

__global__ void reset_forces_kernel(double* dfx, double* dfy, double* dfz, size_t nbpart) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nbpart) return;

  dfx[i] = 0.;
  dfy[i] = 0.;
  dfz[i] = 0.;
}

__global__ void compute_forces_kernel(double* dmass, double* dx, double* dy, double* dz, 
    double* dfx, double* dfy, double* dfz, 
    size_t nbpart, double G) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nbpart) return;

    double softening = 0.1;
    double my_x = dx[i];
    double my_y = dy[i];
    double my_z = dz[i];
    double my_fx = 0.0;
    double my_fy = 0.0;
    double my_fz = 0.0;

    for (int j = 0; j < nbpart; j++) {
        if (i == j) continue;

        double delta_x = dx[j] - my_x;
        double delta_y = dy[j] - my_y;
        double delta_z = dz[j] - my_z;

        double dist_sq = delta_x*delta_x + delta_y*delta_y + delta_z*delta_z + softening;
        double inv_dist = rsqrt(dist_sq);
        double inv_dist3 = inv_dist * inv_dist * inv_dist;

        double F = G * dmass[i] * dmass[j] * inv_dist3;

        my_fx += F * delta_x;
        my_fy += F * delta_y;
        my_fz += F * delta_z;
    }

    dfx[i] = my_fx;
    dfy[i] = my_fy;
    dfz[i] = my_fz;
}


void dump_state(simulation& s) {
  std::cout<<s.nbpart<<'\t';
  for (size_t i=0; i<s.nbpart; ++i) {
    std::cout<<s.hmass[i]<<'\t';
    std::cout<<s.hx[i]<<'\t'<<s.hy[i]<<'\t'<<s.hz[i]<<'\t';
    std::cout<<s.hvx[i]<<'\t'<<s.hvy[i]<<'\t'<<s.hvz[i]<<'\t';
    std::cout<<s.hfx[i]<<'\t'<<s.hfy[i]<<'\t'<<s.hfz[i]<<'\t';
  }
  std::cout<<'\n';
}

void load_from_file(simulation& s, std::string filename) {
  std::ifstream in (filename);
  size_t nbpart;
  in>>nbpart;
  s = simulation(nbpart);
  for (size_t i=0; i<s.nbpart; ++i) {
    in>>s.hmass[i];
    in >> s.hx[i] >>  s.hy[i] >>  s.hz[i];
    in >> s.hvx[i] >> s.hvy[i] >> s.hvz[i];
    in >> s.hfx[i] >> s.hfy[i] >> s.hfz[i];
  }
  if (!in.good())
    throw "kaboom";
}

int main(int argc, char* argv[]) {
  if (argc != 6) {
    std::cerr
      <<"usage: "<<argv[0]<<" <input> <dt> <nbstep> <printevery>"<<"\n"
      <<"input can be:"<<"\n"
      <<"a number (random initialization)"<<"\n"
      <<"planet (initialize with solar system)"<<"\n"
      <<"a filename (load from file in singleline tsv)"<<"\n"
      <<"block size"<<"\n";
    return -1;
  }

  double dt = std::atof(argv[2]); //in seconds
  size_t nbstep = std::atol(argv[3]);
  size_t printevery = std::atol(argv[4]);
  int blockSize = std::atol(argv[5]);
  
  simulation s(1);

  //parse command line
  {
    size_t nbpart = std::atol(argv[1]); //return 0 if not a number
    if ( nbpart > 0) {
      s = simulation(nbpart);
      random_init(s);
    } else {
      std::string inputparam = argv[1];
      if (inputparam == "planet") {
	init_solar(s);
      } else{
	load_from_file(s, inputparam);
      }
    }    
  }

  
  int numBlocks = (s.nbpart + blockSize - 1) / blockSize;

  auto start = std::chrono::high_resolution_clock::now();
  
  for (size_t step = 0; step < nbstep; step++) {
      if (step % printevery == 0) {
          s.device_to_host();
          dump_state(s);
      }

      reset_forces_kernel<<<numBlocks, blockSize>>>(s.dfx, s.dfy, s.dfz, s.nbpart);
      
      // Compute forces on device
      compute_forces_kernel<<<numBlocks, blockSize>>>(s.dmass, s.dx, s.dy, s.dz, 
                                                    s.dfx, s.dfy, s.dfz, 
                                                    s.nbpart, G);
      
      // Update positions and velocities
      update_particles_kernel<<<numBlocks, blockSize>>>(s.dx, s.dy, s.dz,
                                                       s.dvx, s.dvy, s.dvz,
                                                       s.dfx, s.dfy, s.dfz,
                                                       s.dmass, s.nbpart, dt);
      
  }
  
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "GPU Time: " << elapsed.count() << " s" << std::endl;
}
