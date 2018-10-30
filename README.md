
<p align="center">
  <a href="http://mathlab.github.io/ITHACA-FV/" target="_blank" >
    <img alt="BlackNUFFT" src="./docs/logo/logo.png" width="200" />
  </a>
</p>

# BlackNUFFT

This repository contains a library for the computation of the Non Uniform Fast Fourier Transform. As reference algorithm we consider the outstanding work by Greengard and Lee which is freely available, under GPL license, here http://www.cims.nyu.edu/cmcl/nufft/nufft.html.
We present a flexible and modular implementation of NUFFT of type 3 in 3D in C++. We make use of existing High Performance Computing libraries as the deal.ii library, available under LGPL license here https://www.dealii.org/. Our goal is to parallelise the C++ implementation through a multicore-multiprocessor paradigm using both Intel Threading Building Block and MPI. The modularity we have prescribed allows for extensibility of every part of the algorithm.

## BlackNUFFT Breakdown

The NUFFT algorithm can be divided in five main steps

- set up of the NUFFT
	- index set creation for the MPI and TBB parallelisations
	- setting of the tolerance and computation of the spreading constants for the gridding
	
- gridding from input array to the fine grid array, as preliminary gridding we present a hybridly parallel implementation of the Fast Guassian Gridding bt Greengard and Lee
	- FGG
	- scaling of the resulting array
-  FFT on the the distributed fine grid array
	- call of a 3D parallel existing FFT
	- circular shift of the results

- gridding from the fine grid array to the output array, as preliminary gridding we present a hybridly parallel implementation of the Fast Guassian Gridding bt Greengard and Lee
	- FGG
	- scaling of the resulting array



## Install Procedure


To install BlackNUFFT from scratch it is sufficient to install the FFTW library (if the user want to change the FFT backend library it is sufficient to change the CMakeLists.txt)

	wget http://www.fftw.org/fftw-3.3.6-pl2.tar.gz &&\
    tar xf fftw-3.3.6-pl2.tar.gz && rm -f fftw-3.3.6-pl2.tar.gz && \
    cd fftw-3.3.6-pl2 && \
    ./configure --enable-mpi --prefix=/your_installation_pwd/ --enable-shared && \
    make install && \
    
Then it is mandatory to install the deal.ii library enabling the MPI framework.

	git clone https://github.com/dealii/dealii.git
	cd dealii
	mkdir build
	cd build
	cmake ../ -DDEAL_II_WITH_MPI=ON -DCMAKE_INSTALL_PREFIX=/your_installation_pwd/
	make install
	
Finally

	git clone https://github.com/nicola-giuliani/BlackNUFFT.git
	cd BlackNUFFT
	mkdir build
	cd build
	export -DFFTW_DIR=/path_to_fftw_install_dir/
	cmake ../ 
	make 
	
at this point in the build directory you have both the shared library of BlackNUFFT and an executable ready to use

