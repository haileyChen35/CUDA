CXXFLAGS=-O3
NVCC = nvcc
CXXFLAGS = -O3 -std=c++11

Nbody = nbody
Nbody_Par = nbody_par

all: $(Nbody) $(Nbody_Par)

$(Nbody_Par): nbody_gpu.cu
	$(NVCC) $(CXXFLAGS) $< -o $@

$(Nbody): nbody_cpu.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(Nbody) $(Nbody_Par) *.out