CXXFLAGS=-O3

nbody: nbody_par.cu
	nvcc -O3 nbody_par.cu -o nbody_par

clean: 
	rm nbody_par *.out