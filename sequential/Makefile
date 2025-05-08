CXXFLAGS=-O3

nbody: nbody_par.cu
	nvcc -O3 nbody.cpp -o nbody_par

solar.out: nbody_par
	date
	./nbody_par 1000 0.1 500 100 256 
	date

clean: 
	rm nbody_par *.out