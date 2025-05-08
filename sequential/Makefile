CXXFLAGS=-O3
NVCC = nvcc
CXXFLAGS = -O3 -std=c++11
NVCC_FLAGS = -O3 -arch=sm_61

Nbody = nbody
Nbody_Par = nbody_par

all: $(Nbody) $(Nbody_Par)

$(Nbody): nbody.cpp
	$(CXX) $(CXXFLAGS) $< -o $@
	
$(Nbody_Par): nbody_par.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -f $(Nbody) $(Nbody_Par) *.out