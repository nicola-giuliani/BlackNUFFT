#NUFFT

This repository contains two different libraries for the computation of the Non Uniform Fast Fourier Transform. The algorithm has been developed by Professor Leslie Greengard and it is available, under GPL license, here http://www.cims.nyu.edu/cmcl/nufft/nufft.html.
The directory libGreengard contains the original FORTRAN 77 algorithm very optimised on a single core. The directory libNUFFT contains instead its translation in C++ via the usage of the deal.ii library, available under LGPL license here https://www.dealii.org/. Our goal is to parallelise the C++ implementation through a multicore-multiprocessor paradigm using both Intel Threading Building Block and MPI.

##BlackNUFFT Breakdown

We have divided the [original code](http://www.cims.nyu.edu/cmcl/software.html) by Leslie Greengard into the main steps. This new modularity allows for user-driven customisations since every of these functions can be easily replaced. 

- compute_ranges(): it computes the bounding box for the array of points given as input. In this way we can perform all the later steps inside the usual box around the origin.
- compute_tolerance_infos(): given the tolerance required for the computation we use Greengard's algorithm to compute the span of the finer grid and the oversampling parameters.
- create_index_set(): this function is the core of our MPI parallelisation. We need 4 different sets. 
	- fftw3_set: the automatic subdivision that fftw3_mpi builds up. It is a 1d subdivision up to now.	 
	- input_set: we use it to divide the input vector. We follow the subdivision given by fftw3_mpi 
	- fftw3_ghost: the ghost cell we need, we need a layer of n_spread cells if n_spread is the span of the convolution kernel.
	- output_set: a set that divides the output points following the same philosophy of input_set.
- fast_gaussian_gridding_on_input(): we compute the first gaussian gridding from the input array to the finer grid. Basically this is a convolution through a Gaussian kernel. We apply a mixed TBB-MPI parallelisation. We divided the input nodes following the 1d domain decomposition FFTW automatically provides. Then we perform it following the Fast Algorithm developed by Greengard. This is not a pointless operation and we have used WorkStream to deal with it. 
- deconvolution_before_fft(): a deconvolution for the second grinding we will perform on the output. This is a pointwise operation, thus we simply use TaskGroup to parallelise it in a multicore environment. We need to perform the deconvolution only on the owned elements of the fine array. This is done with the fftw3_set we have generated.
- compute_stubborn_fft(): this is a 3d pruned FFT performed using 1d FFTW complex operations. It automatically deals with the data shift. No TBB No MPI.
- compute_fft_3d(): a wrapper to execute the 3d FFT on the fine grid assembled. This function does not deal the data shift.
- shift_data_for_fftw3d(): this function needs to be called after the 3d computation. In this way we are shifting the data through a local operation.
- fast_gaussian_gridding_on_output(): same as for the input but this time we need to pass from the fine grid to the output array. We divide the output array in the same way of the input array.
- deconvolution_after_fft(): we deconvolve the first gaussian gridding and we use the ranges computed before to translate our data back to their original box. Finally we perform a reduction to assemble the entire output vector and we conclude the program.