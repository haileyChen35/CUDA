CXXFLAGS=-O3
NVCC = nvcc
CXX := g++


# Targets
EXECUTABLES := nbody nbody_par

# Default target
all: $(EXECUTABLES)

# Serial version
nbody: nbody.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

# Parallel CUDA version
nbody_par: nbody_par.cu
	$(NVCC) $(CXXFLAGS) $< -o $@

# Cleanup
clean:
	rm -f $(EXECUTABLES) *.o *.out core* *.err *.log