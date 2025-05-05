#include <iostream>
#include <fstream>
#include <random>
#include <cmath>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

double G = 6.674e-11;
//double G = 1;

struct simulation {
  size_t nbpart;
  
  double* mass;

  //position
  double* x;
  double* y;
  double* z;

  //velocity
  double* vx;
  double* vy;
  double* vz;

  //force
  double* fx;
  double* fy;
  double* fz;

  
  simulation(size_t nb) : nbpart(nb) {
    // ALLOCATE DEVICE MEMORY FOR ALL ARRAYS
    cudaMalloc(&mass, nb * sizeof(double));
    cudaMalloc(&x, nb * sizeof(double));
    cudaMalloc(&y, nb * sizeof(double));
    cudaMalloc(&z, nb * sizeof(double));
    cudaMalloc(&vx, nb * sizeof(double));
    cudaMalloc(&vy, nb * sizeof(double));
    cudaMalloc(&vz, nb * sizeof(double));
    cudaMalloc(&fx, nb * sizeof(double));
    cudaMalloc(&fy, nb * sizeof(double));
    cudaMalloc(&fz, nb * sizeof(double));
  }

  ~simulation() {
    cudaFree(mass);
    cudaFree(x); cudaFree(y); cudaFree(z);
    cudaFree(vx); cudaFree(vy); cudaFree(vz);
    cudaFree(fx); cudaFree(fy); cudaFree(fz);
  }
};


void random_init(simulation& s) {
    std::vector<double> mass_host(s.nbpart);
    std::vector<double> x_host(s.nbpart), y_host(s.nbpart), z_host(s.nbpart);
    std::vector<double> vx_host(s.nbpart), vy_host(s.nbpart), vz_host(s.nbpart);

    std::random_device rd;  
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dismass(0.9, 1.);
    std::normal_distribution<double> dispos(0., 1.);
    std::normal_distribution<double> disvel(0., 1.);

    for (size_t i = 0; i < s.nbpart; ++i) {
        mass_host[i] = dismass(gen);

        x_host[i] = dispos(gen);
        y_host[i] = dispos(gen);
        z_host[i] = dispos(gen);
        z_host[i] = 0.;
        
        vx_host[i] = disvel(gen);
        vy_host[i] = disvel(gen);
        vz_host[i] = disvel(gen);
        vz_host[i] = 0.;
        vx_host[i] = y_host[i] * 1.5;
        vy_host[i] = -x_host[i] * 1.5;
    }

    double meanmass = 0;
    double meanmassvx = 0;
    double meanmassvy = 0;
    double meanmassvz = 0;
    
    for (size_t i = 0; i < s.nbpart; ++i) {
        meanmass += mass_host[i];
        meanmassvx += mass_host[i] * vx_host[i];
        meanmassvy += mass_host[i] * vy_host[i];
        meanmassvz += mass_host[i] * vz_host[i];
    }
    
    for (size_t i = 0; i < s.nbpart; ++i) {
        vx_host[i] -= meanmassvx / meanmass;
        vy_host[i] -= meanmassvy / meanmass;
        vz_host[i] -= meanmassvz / meanmass;
    }

    cudaMemcpy(s.mass, mass_host.data(), s.nbpart * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(s.x, x_host.data(), s.nbpart * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(s.y, y_host.data(), s.nbpart * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(s.z, z_host.data(), s.nbpart * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(s.vx, vx_host.data(), s.nbpart * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(s.vy, vy_host.data(), s.nbpart * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(s.vz, vz_host.data(), s.nbpart * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemset(s.fx, 0, s.nbpart * sizeof(double));
    cudaMemset(s.fy, 0, s.nbpart * sizeof(double));
    cudaMemset(s.fz, 0, s.nbpart * sizeof(double));
}

void init_solar(simulation& s) {
    enum Planets {SUN, MERCURY, VENUS, EARTH, MARS, JUPITER, SATURN, URANUS, NEPTUNE, MOON};
    const int num_planets = 10;
    s = simulation(num_planets);

    std::vector<double> mass_host(num_planets);
    std::vector<double> x_host(num_planets), y_host(num_planets), z_host(num_planets);
    std::vector<double> vx_host(num_planets), vy_host(num_planets), vz_host(num_planets);

    mass_host[SUN] = 1.9891 * std::pow(10, 30);
    mass_host[MERCURY] = 3.285 * std::pow(10, 23);
    mass_host[VENUS] = 4.867 * std::pow(10, 24);
    mass_host[EARTH] = 5.972 * std::pow(10, 24);
    mass_host[MARS] = 6.39 * std::pow(10, 23);
    mass_host[JUPITER] = 1.898 * std::pow(10, 27);
    mass_host[SATURN] = 5.683 * std::pow(10, 26);
    mass_host[URANUS] = 8.681 * std::pow(10, 25);
    mass_host[NEPTUNE] = 1.024 * std::pow(10, 26);
    mass_host[MOON] = 7.342 * std::pow(10, 22);

    double AU = 1.496 * std::pow(10, 11); 
    x_host = {0, 0.39*AU, 0.72*AU, 1.0*AU, 1.52*AU, 5.20*AU, 9.58*AU, 19.22*AU, 30.05*AU, 1.0*AU + 3.844*std::pow(10, 8)};
    y_host = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    z_host = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    vx_host = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    vy_host = {0, 47870, 35020, 29780, 24130, 13070, 9680, 6800, 5430, 29780 + 1022};
    vz_host = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    cudaMemcpy(s.mass, mass_host.data(), num_planets * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(s.x, x_host.data(), num_planets * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(s.y, y_host.data(), num_planets * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(s.z, z_host.data(), num_planets * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(s.vx, vx_host.data(), num_planets * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(s.vy, vy_host.data(), num_planets * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(s.vz, vz_host.data(), num_planets * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemset(s.fx, 0, num_planets * sizeof(double));
    cudaMemset(s.fy, 0, num_planets * sizeof(double));
    cudaMemset(s.fz, 0, num_planets * sizeof(double));
}

__global__ void calculateForcesKernel(double* fx, double* fy, double* fz,
                                      double* x, double* y, double* z,
                                      double* mass, int nbpart) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nbpart) return;

    double fx_i = 0, fy_i = 0, fz_i = 0;
    for (int j = 0; j < nbpart; ++j) {
        if (i == j) continue;
        double dx = x[j] - x[i];
        double dy = y[j] - y[i];
        double dz = z[j] - z[i];
        double distSqr = dx*dx + dy*dy + dz*dz + 1e-9;
        double invDist = 1.0 / sqrt(distSqr);
        double invDist3 = invDist * invDist * invDist;
        double f = G * mass[i] * mass[j] * invDist3;

        fx_i += f * dx;
        fy_i += f * dy;
        fz_i += f * dz;
    }
    fx[i] = fx_i;
    fy[i] = fy_i;
    fz[i] = fz_i;
}

__global__ void applyForceKernel(double* vx, double* vy, double* vz,
                                 double* fx, double* fy, double* fz,
                                 double* mass, int nbpart, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nbpart) return;

    vx[i] += fx[i] / mass[i] * dt;
    vy[i] += fy[i] / mass[i] * dt;
    vz[i] += fz[i] / mass[i] * dt;
}

__global__ void updatePositionKernel(double* x, double* y, double* z,
                                     double* vx, double* vy, double* vz,
                                     int nbpart, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nbpart) return;

    x[i] += vx[i] * dt;
    y[i] += vy[i] * dt;
    z[i] += vz[i] * dt;
}

void calculate_forces(simulation& s) {
    int blockSize = 256;
    int gridSize = (s.nbpart + blockSize - 1) / blockSize;
    calculateForcesKernel<<<gridSize, blockSize>>>(s.fx, s.fy, s.fz, s.x, s.y, s.z, s.mass, s.nbpart);
    cudaDeviceSynchronize();
}

void apply_force(simulation& s, double dt) {
    int blockSize = 256;
    int gridSize = (s.nbpart + blockSize - 1) / blockSize;
    applyForceKernel<<<gridSize, blockSize>>>(s.vx, s.vy, s.vz, s.fx, s.fy, s.fz, s.mass, s.nbpart, dt);
    cudaDeviceSynchronize();
}

void update_position(simulation& s, double dt) {
    int blockSize = 256;
    int gridSize = (s.nbpart + blockSize - 1) / blockSize;
    updatePositionKernel<<<gridSize, blockSize>>>(s.x, s.y, s.z, s.vx, s.vy, s.vz, s.nbpart, dt);
    cudaDeviceSynchronize();
}

void dump_state(simulation& s) {
    std::vector<double> mass(s.nbpart), x(s.nbpart), y(s.nbpart), z(s.nbpart);
    std::vector<double> vx(s.nbpart), vy(s.nbpart), vz(s.nbpart);
    std::vector<double> fx(s.nbpart), fy(s.nbpart), fz(s.nbpart);

    cudaMemcpy(mass.data(), s.mass, s.nbpart * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(x.data(), s.x, s.nbpart * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(y.data(), s.y, s.nbpart * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(z.data(), s.z, s.nbpart * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(vx.data(), s.vx, s.nbpart * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(vy.data(), s.vy, s.nbpart * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(vz.data(), s.vz, s.nbpart * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(fx.data(), s.fx, s.nbpart * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(fy.data(), s.fy, s.nbpart * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(fz.data(), s.fz, s.nbpart * sizeof(double), cudaMemcpyDeviceToHost);

    std::cout << s.nbpart << '\t';
    for (size_t i = 0; i < s.nbpart; ++i) {
        std::cout << mass[i] << '\t';
        std::cout << x[i] << '\t' << y[i] << '\t' << z[i] << '\t';
        std::cout << vx[i] << '\t' << vy[i] << '\t' << vz[i] << '\t';
        std::cout << fx[i] << '\t' << fy[i] << '\t' << fz[i] << '\t';
    }
    std::cout << '\n';
}

void allocate_device(simulation& s) {
    cudaMalloc(&s.mass, s.nbpart * sizeof(double));
    cudaMalloc(&s.x, s.nbpart * sizeof(double));
    cudaMalloc(&s.y, s.nbpart * sizeof(double));
    cudaMalloc(&s.z, s.nbpart * sizeof(double));
    cudaMalloc(&s.vx, s.nbpart * sizeof(double));
    cudaMalloc(&s.vy, s.nbpart * sizeof(double));
    cudaMalloc(&s.vz, s.nbpart * sizeof(double));
    cudaMalloc(&s.fx, s.nbpart * sizeof(double));
    cudaMalloc(&s.fy, s.nbpart * sizeof(double));
    cudaMalloc(&s.fz, s.nbpart * sizeof(double));
}

void free_device(simulation& s) {
    cudaFree(s.mass);
    cudaFree(s.x); cudaFree(s.y); cudaFree(s.z);
    cudaFree(s.vx); cudaFree(s.vy); cudaFree(s.vz);
    cudaFree(s.fx); cudaFree(s.fy); cudaFree(s.fz);
}

void load_from_file(const char* filename, simulation& s) {
    std::ifstream file(filename);
    file >> s.nbpart;
    allocate_device(s);

    std::vector<double> mass(s.nbpart), x(s.nbpart), y(s.nbpart), z(s.nbpart);
    std::vector<double> vx(s.nbpart), vy(s.nbpart), vz(s.nbpart);

    for (size_t i = 0; i < s.nbpart; ++i) {
        file >> mass[i] >> x[i] >> y[i] >> z[i] >> vx[i] >> vy[i] >> vz[i];
    }

    cudaMemcpy(s.mass, mass.data(), s.nbpart * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(s.x, x.data(), s.nbpart * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(s.y, y.data(), s.nbpart * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(s.z, z.data(), s.nbpart * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(s.vx, vx.data(), s.nbpart * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(s.vy, vy.data(), s.nbpart * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(s.vz, vz.data(), s.nbpart * sizeof(double), cudaMemcpyHostToDevice);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <dt> <nbsteps>\n";
        return 1;
    }

    simulation s(std::stoul(argv[1]));
    load_from_file(argv[1], s);
    double dt = atof(argv[2]);
    int steps = atoi(argv[3]);

    for (int step = 0; step < steps; ++step) {
        calculate_forces(s);
        apply_force(s, dt);
        update_position(s, dt);
        dump_state(s);
    }

    free_device(s);
    return 0;
}
