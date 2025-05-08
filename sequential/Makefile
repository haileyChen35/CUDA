CXXFLAGS=-O3

nbody: nbody_par.cu
	nvcc -O3 nbody_par.cu -o nbody_par

simulation: nbody_par
	./nbody_par 1000 0.1 500 100 256
	./nbody_par 10000 0.1 500 100 256
	./nbody_par 100000 0.1 500 100 256 

clean: 
	rm nbody_par *.out